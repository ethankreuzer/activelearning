"""Fixed features for arbitrary molecules, looked up by SMILES in a persisted store.

A MiniMol feature cache serves one exact ordered prefix of the rows it was built
from, so it cannot serve a training subset chosen afresh each round from a 10M
archive. :class:`FeatureStoreEncoder` opens the same kind of file for random
access instead: a SMILES -> row index over the base matrix, which stays
memory-mapped and read-only. Molecules the store does not hold are encoded live
and not retained here.

Not retaining them is deliberate. A run-local shard persisted every molecule
this encoder ever saw, which in an active-learning loop is dominated by what the
policy samples while training -- ~240,000 molecules a round in job 24882282,
against ~1,700 rows of genuine cross-round reuse, since a GFlowNet samples novel
structures rather than revisiting the last round's. The molecules that are needed
again, the docked acquisitions in the archive and a fixed held-out set, come to a
few thousand a round and are cheaper to encode again than to store. Within a
round, repeats are the inner encoder's bounded cache to absorb.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch
from torch import Tensor

from activelearning.runtime import RuntimeContext
from activelearning.surrogate.encoder import FixedEncoder


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
    ) -> None:
        """Initialize the store-backed encoder.

        Parameters
        ----------
        inner : FixedEncoder
            The encoder the store was built with. It validates the store and
            encodes the molecules the store does not hold. Its own cache is what
            absorbs repeats within a round, so give it a non-zero ``cache_size``.
        store_path : Path or str
            Base ``.npy`` feature matrix; its manifest is the sibling ``.json``.
        smiles_path : Path or str
            The SMILES the store was built from, one per line in row order.

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
        self._base: np.ndarray | None = None
        self._base_index: dict[str, int] | None = None

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
            features[outside] = self._encode_live(
                [strings[position] for position in outside]
            )
        return torch.from_numpy(features).to(device=device, dtype=torch.float32)

    def _encode_live(self, missing: Sequence[str]) -> np.ndarray:
        """Encode the molecules the store lacks, each distinct one once."""
        distinct = list(dict.fromkeys(missing))
        encoded = self._inner.encode(distinct, device=torch.device("cpu"))
        encoded_array = encoded.detach().to(dtype=torch.float32).numpy()
        if encoded_array.shape != (len(distinct), self.feature_dim):
            raise ValueError(
                "Inner encoder output must have shape "
                f"{(len(distinct), self.feature_dim)}, got {encoded_array.shape}."
            )
        if len(distinct) == len(missing):
            return encoded_array
        row_of = {value: row for row, value in enumerate(distinct)}
        return encoded_array[[row_of[value] for value in missing]]

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
