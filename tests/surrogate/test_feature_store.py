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
