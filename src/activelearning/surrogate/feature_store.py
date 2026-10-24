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

#: Rows the shard file is created with. The file is preallocated and holds more
#: rows than it uses, so an append writes only the new rows instead of rewriting
#: the whole matrix: ``shard.json`` records how many of them are real.
_INITIAL_SHARD_CAPACITY = 4096

#: Rows copied at a time when the shard outgrows its file. Copying through the
#: mapping rather than through an array keeps a doubling of a multi-gigabyte
#: shard off the heap.
_SHARD_COPY_CHUNK = 65_536


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
        # Rows of ``_shard_features`` that hold a molecule, as against the rows
        # the file is merely long enough for.
        self._shard_count = 0
        self._shard_capacity = 0
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
        return self._shard_count

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
        """Encode the values the shard lacks, add them, and persist the new rows.

        The rows are appended in place. Growing the shard by concatenation and
        rewriting the whole file, which is what this did until 2026-10-23, costs
        time proportional to the shard rather than to the molecules added: in
        job 24882282 that made every round slower than the one before it, with a
        ~1 GB rewrite per 64 molecules on a networked filesystem.
        """
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
        start = self._shard_count
        self._reserve_shard_rows(len(new))
        self._shard_features[start : start + len(new)] = encoded_array
        self._shard_count = start + len(new)
        self._shard_smiles.extend(new)
        self._shard_index.update(
            (value, start + offset) for offset, value in enumerate(new)
        )
        if self._shard_dir is not None:
            self._persist_shard_rows(new)

    def _reserve_shard_rows(self, extra: int) -> None:
        """Make room for ``extra`` more rows, doubling the file when it is full.

        Doubling is what keeps appending cheap on average: the copy below runs
        O(log n) times over a run rather than once per append.
        """
        needed = self._shard_count + extra
        if needed <= self._shard_capacity:
            return
        capacity = max(_INITIAL_SHARD_CAPACITY, self._shard_capacity * 2, needed)
        if self._shard_dir is None:
            grown = np.empty((capacity, self.feature_dim), dtype=np.float32)
            grown[: self._shard_count] = self._shard_features[: self._shard_count]
            self._shard_features = grown
        else:
            self._grow_shard_file(capacity)
        self._shard_capacity = capacity

    def _grow_shard_file(self, capacity: int) -> None:
        """Replace the shard file with a longer one holding the same rows."""
        assert self._shard_dir is not None
        self._shard_dir.mkdir(parents=True, exist_ok=True)
        path = self._shard_dir / _SHARD_FEATURES
        descriptor, name = tempfile.mkstemp(
            prefix=f".{_SHARD_FEATURES}.", suffix=".tmp", dir=self._shard_dir
        )
        os.close(descriptor)
        temporary = Path(name)
        try:
            grown = np.lib.format.open_memmap(
                temporary,
                mode="w+",
                dtype=np.float32,
                shape=(capacity, self.feature_dim),
            )
            try:
                # Chunked, and straight from one mapping to the other, so a
                # multi-gigabyte shard never has to fit in memory at once.
                for begin in range(0, self._shard_count, _SHARD_COPY_CHUNK):
                    end = min(begin + _SHARD_COPY_CHUNK, self._shard_count)
                    grown[begin:end] = self._shard_features[begin:end]
                grown.flush()
            finally:
                _close_memmap(grown)
            # Only now is the old mapping expendable.
            _close_memmap(self._shard_features)
            self._shard_features = np.empty((0, self.feature_dim), dtype=np.float32)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        self._shard_features = np.lib.format.open_memmap(path, mode="r+")

    def _persist_shard_rows(self, new: Sequence[str]) -> None:
        """Commit the rows just appended, the manifest last.

        The manifest is the commit point: it is published only once the features
        and the SMILES are on disk, so a crash can leave rows the manifest does
        not count, but never count rows that are not there. ``_load_shard``
        discards the uncounted remainder.
        """
        assert self._shard_dir is not None
        features = self._shard_features
        if isinstance(features, np.memmap):
            features.flush()
        smiles_path = self._shard_dir / _SHARD_SMILES
        with smiles_path.open("a", encoding="utf-8") as stream:
            stream.write("".join(f"{value}\n" for value in new))
            stream.flush()
            os.fsync(stream.fileno())
        self._write_manifest()

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
        row_count = manifest.get("row_count")
        if not isinstance(row_count, int) or isinstance(row_count, bool):
            raise ValueError(
                f"Feature shard {self._shard_dir} has a non-integer row_count: "
                f"{row_count!r}."
            )
        if row_count < 0:
            raise ValueError(
                f"Feature shard {self._shard_dir} has a negative row_count: "
                f"{row_count}."
            )
        features = np.lib.format.open_memmap(
            self._shard_dir / _SHARD_FEATURES, mode="r+"
        )
        smiles_path = self._shard_dir / _SHARD_SMILES
        smiles = smiles_path.read_text(encoding="utf-8").splitlines()
        # The file is preallocated, so it is long enough rather than exact. Too
        # short, or too few SMILES, means rows the manifest counts are missing.
        if (
            features.ndim != 2
            or features.shape[1] != self.feature_dim
            or features.shape[0] < row_count
            or len(smiles) < row_count
        ):
            raise ValueError(
                f"Feature shard {self._shard_dir} is inconsistent: "
                f"{features.shape} features and {len(smiles)} SMILES for "
                f"{row_count} rows."
            )
        if len(smiles) > row_count:
            # A crash between appending the SMILES and publishing the manifest.
            # The surplus names rows that were never committed; leaving them
            # would offset every row appended from here on.
            smiles = smiles[:row_count]
            smiles_path.write_text(
                "".join(f"{value}\n" for value in smiles), encoding="utf-8"
            )
        self._shard_features = features
        self._shard_capacity = int(features.shape[0])
        self._shard_count = row_count
        self._shard_smiles = smiles
        self._shard_index = {value: row for row, value in enumerate(smiles)}

    def _write_manifest(self) -> None:
        """Publish the row count atomically, which commits the rows below it."""
        assert self._shard_dir is not None
        self._shard_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "identity": self._shard_identity(),
            "row_count": self._shard_count,
        }
        manifest_text = json.dumps(manifest, sort_keys=True, indent=2) + "\n"
        _atomic_write(
            self._shard_dir / _SHARD_MANIFEST,
            lambda stream: stream.write(manifest_text.encode("utf-8")),
        )


def _close_memmap(features: np.ndarray) -> None:
    """Release a memory mapping now rather than when the garbage collector runs.

    Growing the shard replaces the file the old mapping refers to, and a
    deterministic close keeps the superseded file from outliving the rename.
    """
    if isinstance(features, np.memmap):
        features.flush()
        mapping = getattr(features, "_mmap", None)
        if mapping is not None:
            mapping.close()


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
