"""Tests for fixed features looked up by SMILES in a persisted store."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import Tensor

from activelearning.surrogate.encoder import FixedEncoder
from activelearning.surrogate.feature_store import FeatureStoreEncoder

BASE = ["C", "CC", "CCO", "CCN"]


def _features(value: str) -> list[float]:
    """Three features that tell every test molecule apart."""
    return [float(len(value)), float(value.count("O")), float(value.count("N"))]


class _StoreEncoder(FixedEncoder):
    """Encode live by character counts; open a store saved with ``np.save``."""

    feature_dim = 3

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, values: Sequence[Any], *, device: torch.device) -> Tensor:
        self.calls.append(list(values))
        return torch.tensor([_features(v) for v in values], device=device)

    def open_feature_store(self, feature_path: Path, values: Sequence[str]) -> Any:
        features = np.load(feature_path, mmap_mode="r")
        if len(features) != len(values):
            raise ValueError("store was not built from these SMILES")
        return features


class _OtherEncoder(_StoreEncoder):
    """The same features from a different encoder class."""


def _store(tmp_path: Path) -> tuple[Path, Path]:
    """Write a base store whose features differ from live encoding."""
    store_path = tmp_path / "store.npy"
    smiles_path = tmp_path / "store.smiles.txt"
    # Offset by 100 so a row served from the store is told apart from one
    # encoded live.
    np.save(store_path, np.array([_features(v) for v in BASE], np.float32) + 100)
    smiles_path.write_text("".join(f"{v}\n" for v in BASE), encoding="utf-8")
    return store_path, smiles_path


def test_serves_store_rows_in_request_order(tmp_path: Path) -> None:
    """Hits come from the store, in the order asked, with no live encoding."""
    store_path, smiles_path = _store(tmp_path)
    inner = _StoreEncoder()
    encoder = FeatureStoreEncoder(
        inner, store_path=store_path, smiles_path=smiles_path
    )

    features = encoder.encode(["CCN", "C", "CCO"], device=torch.device("cpu"))

    expected = torch.tensor([_features(v) for v in ["CCN", "C", "CCO"]]) + 100
    assert torch.equal(features, expected)
    assert inner.calls == []
    assert encoder.base_rows(["CC", "OCCO"]).tolist() == [1, -1]


def test_encodes_misses_once_and_keeps_them_in_the_shard(tmp_path: Path) -> None:
    """A molecule outside the store is encoded live once, then reused."""
    store_path, smiles_path = _store(tmp_path)
    inner = _StoreEncoder()
    encoder = FeatureStoreEncoder(
        inner,
        store_path=store_path,
        smiles_path=smiles_path,
        shard_dir=tmp_path / "shard",
    )
    cpu = torch.device("cpu")

    first = encoder.encode(["OCCO", "CC", "NCCN", "OCCO"], device=cpu)
    second = encoder.encode(["NCCN"], device=cpu)

    assert torch.equal(first[0], torch.tensor(_features("OCCO")))
    assert torch.equal(first[1], torch.tensor(_features("CC")) + 100)
    assert torch.equal(first[3], first[0])
    assert torch.equal(second[0], first[2])
    assert inner.calls == [["OCCO", "NCCN"]]
    assert encoder.shard_size == 2

    # A new encoder on the same shard picks up where the first left off.
    resumed_inner = _StoreEncoder()
    resumed = FeatureStoreEncoder(
        resumed_inner,
        store_path=store_path,
        smiles_path=smiles_path,
        shard_dir=tmp_path / "shard",
    )
    assert torch.equal(resumed.encode(["NCCN"], device=cpu), second)
    assert resumed_inner.calls == []


def test_rejects_a_shard_from_another_encoder(tmp_path: Path) -> None:
    """Features written by one encoder are never served for another."""
    store_path, smiles_path = _store(tmp_path)
    FeatureStoreEncoder(
        _StoreEncoder(),
        store_path=store_path,
        smiles_path=smiles_path,
        shard_dir=tmp_path / "shard",
    ).encode(["OCCO"], device=torch.device("cpu"))

    with pytest.raises(ValueError, match="different encoder"):
        FeatureStoreEncoder(
            _OtherEncoder(),
            store_path=store_path,
            smiles_path=smiles_path,
            shard_dir=tmp_path / "shard",
        )


def test_propagates_a_store_that_does_not_match_its_smiles(tmp_path: Path) -> None:
    """The inner encoder's validation of the store is not bypassed."""
    store_path, smiles_path = _store(tmp_path)
    smiles_path.write_text("C\nCC\n", encoding="utf-8")
    encoder = FeatureStoreEncoder(
        _StoreEncoder(), store_path=store_path, smiles_path=smiles_path
    )

    with pytest.raises(ValueError, match="not built from these SMILES"):
        encoder.encode(["C"], device=torch.device("cpu"))


def test_requires_an_inner_encoder_that_opens_stores() -> None:
    """An encoder with no store support is rejected up front."""

    class _Plain(FixedEncoder):
        feature_dim = 3

        def encode(self, values: Sequence[Any], *, device: torch.device) -> Tensor:
            raise AssertionError("never called")

    with pytest.raises(TypeError, match="open_feature_store"):
        FeatureStoreEncoder(_Plain(), store_path="a.npy", smiles_path="a.txt")


def _encoder(tmp_path: Path, inner: FixedEncoder | None = None) -> FeatureStoreEncoder:
    """A shard-backed encoder over the usual base store."""
    store_path, smiles_path = _store(tmp_path)
    return FeatureStoreEncoder(
        inner if inner is not None else _StoreEncoder(),
        store_path=store_path,
        smiles_path=smiles_path,
        shard_dir=tmp_path / "shard",
    )


def _manifest(tmp_path: Path) -> dict[str, Any]:
    """The shard's published manifest."""
    import json

    return json.loads((tmp_path / "shard" / "shard.json").read_text(encoding="utf-8"))


def _novel(count: int, *, offset: int = 0) -> list[str]:
    """Distinct molecules no base store in these tests holds."""
    return [f"O{'C' * (index + offset + 2)}O" for index in range(count)]


def test_the_shard_file_is_preallocated_rather_than_rewritten(
    tmp_path: Path,
) -> None:
    """The fix for job 24882282: appending must not rewrite the whole matrix.

    A file longer than its row count is the whole mechanism -- the manifest says
    how many rows are real, so an append writes only the rows it adds.
    """
    encoder = _encoder(tmp_path)

    encoder.encode(_novel(3), device=torch.device("cpu"))

    features = np.load(tmp_path / "shard" / "features.npy", mmap_mode="r")
    assert features.shape[0] > 3
    assert _manifest(tmp_path)["row_count"] == 3
    assert encoder.shard_size == 3


def test_appending_does_not_touch_the_rows_already_there(tmp_path: Path) -> None:
    """Rows must be written in place, leaving their predecessors untouched."""
    encoder = _encoder(tmp_path)
    cpu = torch.device("cpu")
    first = encoder.encode(_novel(2), device=cpu)

    encoder.encode(_novel(2, offset=10), device=cpu)

    assert torch.equal(encoder.encode(_novel(2), device=cpu), first)


def test_the_shard_grows_past_its_initial_capacity(tmp_path: Path) -> None:
    """Crossing the capacity boundary is the one path that copies."""
    from activelearning.surrogate import feature_store

    monkey = feature_store._INITIAL_SHARD_CAPACITY
    feature_store._INITIAL_SHARD_CAPACITY = 2
    try:
        encoder = _encoder(tmp_path)
        cpu = torch.device("cpu")
        # Two appends either side of the boundary, then one that leaps it.
        kept = encoder.encode(_novel(2), device=cpu)
        encoder.encode(_novel(5, offset=20), device=cpu)

        assert encoder.shard_size == 7
        assert _manifest(tmp_path)["row_count"] == 7
        # The copy preserved what was already written.
        assert torch.equal(encoder.encode(_novel(2), device=cpu), kept)
        features = np.load(tmp_path / "shard" / "features.npy", mmap_mode="r")
        assert features.shape[0] >= 7
    finally:
        feature_store._INITIAL_SHARD_CAPACITY = monkey


def test_a_grown_shard_survives_a_restart(tmp_path: Path) -> None:
    """The capacity boundary must not corrupt what a later run reads back."""
    from activelearning.surrogate import feature_store

    monkey = feature_store._INITIAL_SHARD_CAPACITY
    feature_store._INITIAL_SHARD_CAPACITY = 2
    try:
        cpu = torch.device("cpu")
        encoder = _encoder(tmp_path)
        expected = encoder.encode(_novel(6), device=cpu)
    finally:
        feature_store._INITIAL_SHARD_CAPACITY = monkey

    resumed_inner = _StoreEncoder()
    resumed = _encoder(tmp_path, resumed_inner)

    assert resumed.shard_size == 6
    assert torch.equal(resumed.encode(_novel(6), device=torch.device("cpu")), expected)
    assert resumed_inner.calls == []


def test_smiles_written_but_never_committed_are_discarded(tmp_path: Path) -> None:
    """A crash between appending SMILES and publishing the manifest.

    The surplus names rows that were never committed. Keeping them would offset
    every row appended afterwards, which would silently serve one molecule's
    features for another -- the one failure here that announces nothing.
    """
    cpu = torch.device("cpu")
    encoder = _encoder(tmp_path)
    expected = encoder.encode(_novel(2), device=cpu)
    smiles_path = tmp_path / "shard" / "smiles.txt"
    with smiles_path.open("a", encoding="utf-8") as stream:
        stream.write("NEVER_COMMITTED\n")

    resumed = _encoder(tmp_path)

    assert resumed.shard_size == 2
    assert smiles_path.read_text(encoding="utf-8").splitlines() == _novel(2)
    # The next append lands on row 2, not row 3.
    assert torch.equal(resumed.encode(_novel(2), device=cpu), expected)
    resumed.encode(_novel(1, offset=30), device=cpu)
    assert resumed.shard_size == 3


def test_a_shard_shorter_than_its_row_count_is_refused(tmp_path: Path) -> None:
    """Too few rows means committed features are missing, which is corruption."""
    _encoder(tmp_path).encode(_novel(2), device=torch.device("cpu"))
    manifest_path = tmp_path / "shard" / "shard.json"
    import json

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["row_count"] = 10_000
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="inconsistent"):
        _encoder(tmp_path)


def test_a_non_integer_row_count_is_refused(tmp_path: Path) -> None:
    """The row count decides which rows are real; it cannot be guessed at."""
    _encoder(tmp_path).encode(_novel(2), device=torch.device("cpu"))
    manifest_path = tmp_path / "shard" / "shard.json"
    import json

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["row_count"] = "2"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="non-integer row_count"):
        _encoder(tmp_path)


def test_an_exact_length_shard_from_the_old_format_still_loads(
    tmp_path: Path,
) -> None:
    """Shards written before preallocation have capacity equal to row count."""
    shard = tmp_path / "shard"
    shard.mkdir()
    import json

    np.save(
        shard / "features.npy",
        np.array([_features(v) for v in _novel(2)], dtype=np.float32),
    )
    (shard / "smiles.txt").write_text(
        "".join(f"{v}\n" for v in _novel(2)), encoding="utf-8"
    )
    identity = {
        "encoder_class": (
            f"{_StoreEncoder.__module__}.{_StoreEncoder.__qualname__}"
        ),
        "feature_dim": 3,
    }
    (shard / "shard.json").write_text(
        json.dumps({"identity": identity, "row_count": 2}), encoding="utf-8"
    )
    inner = _StoreEncoder()
    encoder = _encoder(tmp_path, inner)

    assert encoder.shard_size == 2
    # Served from the shard, not re-encoded, and appending still works.
    encoder.encode(_novel(2), device=torch.device("cpu"))
    assert inner.calls == []
    encoder.encode(_novel(1, offset=40), device=torch.device("cpu"))
    assert encoder.shard_size == 3
