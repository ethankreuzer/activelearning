"""Fixed features for arbitrary molecules, looked up by SMILES in a persisted store.

A MiniMol feature cache serves one exact ordered prefix of the rows it was built
from, so it cannot serve a training subset chosen afresh each round from a 10M
archive. :class:`FeatureStoreEncoder` opens the same kind of file for random
access instead: a SMILES -> row index over the base matrix, which stays
memory-mapped and read-only, plus a small run-local shard for molecules the base
store does not hold, such as newly docked acquisitions. Only those are encoded
live, once each.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, BinaryIO, Protocol, runtime_checkable

import numpy as np
import torch
from torch import Tensor

from activelearning.runtime import RuntimeContext
from activelearning.surrogate.encoder import FixedEncoder

_SHARD_FEATURES = "features.npy"
_SHARD_SMILES = "smiles.txt"
_SHARD_MANIFEST = "shard.json"


@runtime_checkable
class SupportsFeatureStore(Protocol):
    """A fixed encoder that can open a persisted feature matrix by row."""

    feature_dim: int

    def open_feature_store(
        self,
        feature_path: Path,
        values: Sequence[str],
    ) -> np.ndarray:
        """Return the matrix whose row ``i`` holds the features of ``values[i]``."""
        ...


class FeatureStoreEncoder(FixedEncoder):
    """Serve fixed features from a persisted store, encoding only what it lacks."""

    def __init__(
        self,
        inner: FixedEncoder,
        *,
        store_path: str | Path,
        smiles_path: str | Path,
        shard_dir: str | Path | None = None,
    ) -> None:
        """Initialize the store-backed encoder.

        Parameters
        ----------
        inner : FixedEncoder
            The encoder the store was built with. It validates the store and
            encodes the molecules the store does not hold.
        store_path : Path or str
            Base ``.npy`` feature matrix; its manifest is the sibling ``.json``.
        smiles_path : Path or str
            The SMILES the store was built from, one per line in row order.
        shard_dir : Path or str, optional
            Directory where features encoded live are kept, so a resumed run
            does not encode them again. ``None`` keeps them in memory only.

        Raises
        ------
        TypeError
            If ``inner`` cannot open a feature store.
        """
        if not isinstance(inner, SupportsFeatureStore):
            raise TypeError(
                f"{type(inner).__name__} cannot open a feature store; it needs an "
                "open_feature_store() method."
            )
        self.feature_dim = inner.feature_dim
        self._inner = inner
        self._store_path = Path(store_path).expanduser()
        self._smiles_path = Path(smiles_path).expanduser()
        self._shard_dir = None if shard_dir is None else Path(shard_dir).expanduser()
        self._base: np.ndarray | None = None
        self._base_index: dict[str, int] | None = None
        self._shard_features = np.empty((0, self.feature_dim), dtype=np.float32)
        self._shard_smiles: list[str] = []
        self._shard_index: dict[str, int] = {}
        if self._shard_dir is not None:
            self._load_shard()

    def bind_runtime_context(self, runtime_context: RuntimeContext) -> None:
        """Bind runtime settings to this encoder and the one it wraps."""
        super().bind_runtime_context(runtime_context)
        self._inner.bind_runtime_context(runtime_context)

    @property
    def base_features(self) -> np.ndarray:
        """The base store's read-only ``(rows, feature_dim)`` matrix."""
        self._open_base()
        assert self._base is not None
        return self._base

    def base_rows(self, values: Sequence[str]) -> np.ndarray:
        """Return each value's row in the base store, or -1 where it has none."""
        self._open_base()
        assert self._base_index is not None
        index = self._base_index
        return np.fromiter(
            (index.get(value, -1) for value in values),
            dtype=np.int64,
            count=len(values),
        )

    @property
    def shard_size(self) -> int:
        """Number of molecules encoded live and kept in the shard."""
        return len(self._shard_smiles)

    def encode(self, values: Sequence[Any], *, device: torch.device) -> Tensor:
        """Return features for ``values`` in order, from the store where possible."""
        strings = [str(value) for value in values]
        features = np.empty((len(strings), self.feature_dim), dtype=np.float32)
        rows = self.base_rows(strings)
        in_base = np.flatnonzero(rows >= 0)
        if in_base.size:
            # Reading the memmap in row order keeps the access sequential.
            order = np.argsort(rows[in_base], kind="stable")
            positions = in_base[order]
            features[positions] = self.base_features[rows[positions]]

        outside = np.flatnonzero(rows < 0)
        if outside.size:
            missing = [strings[position] for position in outside]
            self._extend_shard(missing)
            shard_rows = [self._shard_index[value] for value in missing]
            features[outside] = self._shard_features[shard_rows]
        return torch.from_numpy(features).to(device=device, dtype=torch.float32)

    def _open_base(self) -> None:
        """Open the base store and index it by SMILES on first use."""
        if self._base is not None:
            return
        smiles = self._smiles_path.read_text(encoding="utf-8").splitlines()
        base = self._inner.open_feature_store(self._store_path, smiles)
        if base.shape != (len(smiles), self.feature_dim):
            raise ValueError(
                f"Feature store {self._store_path} has shape {base.shape}; "
                f"expected {(len(smiles), self.feature_dim)}."
            )
        self._base_index = {value: row for row, value in enumerate(smiles)}
        self._base = base

    def _extend_shard(self, values: Sequence[str]) -> None:
        """Encode the values the shard lacks, add them, and persist the shard."""
        new = list(dict.fromkeys(v for v in values if v not in self._shard_index))
        if not new:
            return
        encoded = self._inner.encode(new, device=torch.device("cpu"))
        encoded_array = encoded.detach().to(dtype=torch.float32).numpy()
        if encoded_array.shape != (len(new), self.feature_dim):
            raise ValueError(
                "Inner encoder output must have shape "
                f"{(len(new), self.feature_dim)}, got {encoded_array.shape}."
            )
        start = len(self._shard_smiles)
        self._shard_features = np.concatenate([self._shard_features, encoded_array])
        self._shard_smiles.extend(new)
        self._shard_index.update(
            (value, start + offset) for offset, value in enumerate(new)
        )
        if self._shard_dir is not None:
            self._write_shard()

    def _shard_identity(self) -> dict[str, Any]:
        """Identify the encoder a shard was written by."""
        inner_type = type(self._inner)
        return {
            "encoder_class": f"{inner_type.__module__}.{inner_type.__qualname__}",
            "feature_dim": self.feature_dim,
        }

    def _load_shard(self) -> None:
        """Restore a shard written by an earlier run of the same encoder."""
        assert self._shard_dir is not None
        manifest_path = self._shard_dir / _SHARD_MANIFEST
        if not manifest_path.is_file():
            return
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("identity") != self._shard_identity():
            raise ValueError(
                f"Feature shard {self._shard_dir} was written by a different "
                f"encoder: {manifest.get('identity')}."
            )
        features = np.load(self._shard_dir / _SHARD_FEATURES, allow_pickle=False)
        smiles = (
            (self._shard_dir / _SHARD_SMILES).read_text(encoding="utf-8").splitlines()
        )
        if features.shape != (len(smiles), self.feature_dim) or len(smiles) != (
            manifest.get("row_count")
        ):
            raise ValueError(
                f"Feature shard {self._shard_dir} is inconsistent: "
                f"{features.shape} features for {len(smiles)} SMILES."
            )
        self._shard_features = features.astype(np.float32, copy=False)
        self._shard_smiles = smiles
        self._shard_index = {value: row for row, value in enumerate(smiles)}

    def _write_shard(self) -> None:
        """Rewrite the shard atomically; the manifest is published last."""
        assert self._shard_dir is not None
        self._shard_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write(
            self._shard_dir / _SHARD_FEATURES,
            lambda stream: np.save(stream, self._shard_features, allow_pickle=False),
        )
        smiles_text = "".join(f"{value}\n" for value in self._shard_smiles)
        _atomic_write(
            self._shard_dir / _SHARD_SMILES,
            lambda stream: stream.write(smiles_text.encode("utf-8")),
        )
        manifest = {
            "identity": self._shard_identity(),
            "row_count": len(self._shard_smiles),
        }
        manifest_text = json.dumps(manifest, sort_keys=True, indent=2) + "\n"
        _atomic_write(
            self._shard_dir / _SHARD_MANIFEST,
            lambda stream: stream.write(manifest_text.encode("utf-8")),
        )


def _atomic_write(path: Path, write: Callable[[BinaryIO], object]) -> None:
    """Write a file through a temporary sibling, then move it into place."""
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            write(stream)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
