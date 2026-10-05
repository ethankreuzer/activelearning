"""Tests for the exact-DKL prep script.

The prep job's contract is that an arm can read a resolved set CSV and its
feature cache and be certain they describe the same ordered rows. These tests
pin the row selection, the file formats and the staleness check that enforces
it.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import exact_dkl_prepare as prep
from scripts.surrogate_eval_io import select_train_eval_rows


class _FakeEncoder:
    """Stand-in encoder that publishes a minimal cache and manifest."""

    def __init__(self, feature_cache_path: Path) -> None:
        self.feature_cache_path = Path(feature_cache_path)
        self.encoded: list[str] = []

    def encode(self, values: list[str], *, device: object) -> torch.Tensor:
        self.encoded = list(values)
        features = torch.zeros((len(values), 512), dtype=torch.float32)
        np.save(self.feature_cache_path, features.numpy())
        manifest = {
            "complete": True,
            "row_count": len(values),
            "input_sha256": prep.hash_ordered_strings(values),
        }
        self.feature_cache_path.with_name(
            self.feature_cache_path.name + ".json"
        ).write_text(json.dumps(manifest))
        return features


def _write_training_csv(path: Path, rows: list[tuple[str, float]]) -> None:
    """Write a tiny stand-in for the 10M training CSV."""
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["SMILE", "y", "fidelity"])
        for smiles, y in rows:
            writer.writerow([smiles, repr(y), "1"])


def _read_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV back as dict rows."""
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture
def training_csv(tmp_path: Path) -> Path:
    """A ten-row training CSV with distinct, unsorted targets."""
    path = tmp_path / "train.csv"
    _write_training_csv(
        path,
        [
            (f"C{index}", value)
            for index, value in enumerate(
                [0.5, 0.9, 0.1, 0.7, 0.3, 0.8, 0.2, 0.6, 0.4, 1.0]
            )
        ],
    )
    return path


def test_read_training_targets_preserves_file_order(training_csv: Path) -> None:
    """The row order is the cache's identity, so it must not be disturbed."""
    smiles, targets, fidelities = prep.read_training_targets(training_csv)
    assert smiles[:3] == ["C0", "C1", "C2"]
    assert targets[1] == pytest.approx(0.9)
    assert set(fidelities) == {"1"}


def test_read_training_targets_reports_a_missing_column(tmp_path: Path) -> None:
    """A wrong column name must fail loudly, not yield an empty set."""
    path = tmp_path / "bad.csv"
    path.write_text("SMILES,y\nC,0.1\n")
    with pytest.raises(ValueError, match="missing column"):
        prep.read_training_targets(path)


def test_top_sets_are_prefixes_of_each_other(training_csv: Path) -> None:
    """top-2 must be the first two rows of top-3, so the arms nest cleanly."""
    _, targets, _ = prep.read_training_targets(training_csv)
    _, top_rows = select_train_eval_rows(targets, 4, 5, 42)
    assert list(top_rows[:2]) == list(top_rows)[:2]
    assert list(top_rows[:3])[:2] == list(top_rows[:2])
    # Highest target first.
    assert targets[top_rows[0]] == pytest.approx(1.0)
    assert targets[top_rows[1]] == pytest.approx(0.9)


def test_write_training_csv_uses_the_dataset_schema(
    tmp_path: Path, training_csv: Path
) -> None:
    """The loader reads `SMILE,y,fidelity`; note the singular column name."""
    smiles, targets, fidelities = prep.read_training_targets(training_csv)
    _, top_rows = select_train_eval_rows(targets, 4, 5, 42)
    path = tmp_path / "ampc_top_3.csv"
    prep.write_training_csv(path, smiles, targets, fidelities, top_rows[:3])
    rows = _read_rows(path)
    assert list(rows[0]) == list(prep.TRAIN_CSV_COLUMNS)
    assert [float(row["y"]) for row in rows] == [1.0, 0.9, 0.8]
    assert {row["fidelity"] for row in rows} == {"1"}
    assert not list(tmp_path.glob("*.tmp"))


def test_resolve_set_filters_the_generated_set(tmp_path: Path) -> None:
    """The generated set keeps rows that docked and pass the SA filter."""
    path = tmp_path / "generated.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["SMILES", "y", "passes_sa"])
        writer.writerow(["CC", "0.4", "1"])
        writer.writerow(["CCC", "", "1"])  # failed to dock
        writer.writerow(["CCCC", "0.6", "0"])  # fails the SA filter
        writer.writerow(["CCCCC", "0.8", "1"])
    resolved = prep.resolve_set(
        prep.SetSpec(
            name="gp_molformer_set",
            source=path,
            smiles_column="SMILES",
            label_column="y",
            extra_columns=("passes_sa",),
            docked_sa_filter=True,
        )
    )
    assert resolved.smiles == ("CC", "CCCCC")
    assert resolved.counts["n_total"] == 4
    assert resolved.counts["n_docked"] == 3
    assert resolved.counts["n_rows"] == 2


def test_resolve_set_handles_a_set_with_no_targets(tmp_path: Path) -> None:
    """The in-vitro set has no `y` column at all."""
    path = tmp_path / "olivier.csv"
    path.write_text("ZINC ID,SMILES\nz1,CC\nz2,CCC\n")
    resolved = prep.resolve_set(
        prep.SetSpec(
            name="olivier_invitro",
            source=path,
            smiles_column="SMILES",
            label_column=None,
        )
    )
    assert resolved.smiles == ("CC", "CCC")
    assert np.isnan(resolved.targets).all()


def test_write_resolved_csv_round_trips_weights(tmp_path: Path) -> None:
    """The validation set's weights must survive into the file an arm reads."""
    resolved = prep.ResolvedSet(
        name="val_set",
        smiles=("CC", "CCC"),
        targets=np.array([0.3, 0.7]),
        extras={"weight": ["1.0", "2.5"]},
        source=tmp_path / "source.csv",
        counts={},
    )
    path = tmp_path / "val_set.csv"
    prep.write_resolved_csv(path, resolved)
    rows = _read_rows(path)
    assert list(rows[0]) == ["SMILES", "y", "weight"]
    assert [row["weight"] for row in rows] == ["1.0", "2.5"]
    assert not list(tmp_path.glob("*.tmp"))


def test_encode_set_publishes_a_cache_per_set(tmp_path: Path) -> None:
    """Each set gets its own cache, keyed to its own ordered input."""
    resolved = prep.ResolvedSet(
        name="train_2000",
        smiles=("CC", "CCC", "CCCC"),
        targets=np.array([0.3, 0.2, 0.1]),
        extras={},
        source=tmp_path / "source.csv",
        counts={"n_rows": 3},
    )
    cache_path = tmp_path / "train_2000.npy"
    encoders: list[_FakeEncoder] = []

    def factory(*, feature_cache_path: Path) -> _FakeEncoder:
        encoders.append(_FakeEncoder(feature_cache_path))
        return encoders[-1]

    entry = prep.encode_set(
        resolved, cache_path, encoder_factory=factory, overwrite=False
    )
    assert entry["rows"] == 3
    assert entry["rebuilt"] is True
    assert encoders[0].encoded == ["CC", "CCC", "CCCC"]
    assert cache_path.exists()

    # A second call finds the cache current and does not re-encode.
    entry = prep.encode_set(
        resolved, cache_path, encoder_factory=factory, overwrite=False
    )
    assert entry["rebuilt"] is False
    assert len(encoders) == 1


def test_encode_set_rebuilds_when_overwriting(tmp_path: Path) -> None:
    """`--overwrite` must replace a current cache rather than skip it."""
    resolved = prep.ResolvedSet(
        name="train_2000",
        smiles=("CC",),
        targets=np.array([0.3]),
        extras={},
        source=tmp_path / "source.csv",
        counts={},
    )
    cache_path = tmp_path / "train_2000.npy"
    calls: list[_FakeEncoder] = []

    def factory(*, feature_cache_path: Path) -> _FakeEncoder:
        calls.append(_FakeEncoder(feature_cache_path))
        return calls[-1]

    prep.encode_set(resolved, cache_path, encoder_factory=factory, overwrite=False)
    prep.encode_set(resolved, cache_path, encoder_factory=factory, overwrite=True)
    assert len(calls) == 2


def test_stale_cache_raises_rather_than_being_reused(tmp_path: Path) -> None:
    """A cache describing different rows is the one failure to never allow through."""
    cache_path = tmp_path / "train_2000.npy"
    np.save(cache_path, np.zeros((2, 512), dtype=np.float32))
    cache_path.with_name(cache_path.name + ".json").write_text(
        json.dumps(
            {
                "complete": True,
                "row_count": 2,
                "input_sha256": prep.hash_ordered_strings(["CC", "CCC"]),
            }
        )
    )
    resolved = prep.ResolvedSet(
        name="train_2000",
        smiles=("CC", "CCCC"),
        targets=np.array([0.3, 0.1]),
        extras={},
        source=tmp_path / "source.csv",
        counts={},
    )
    with pytest.raises(ValueError, match="does not match the train_2000 set"):
        prep.cache_is_current(cache_path, resolved)


def test_incomplete_cache_is_not_treated_as_current(tmp_path: Path) -> None:
    """A half-written cache must be rebuilt, not trusted."""
    cache_path = tmp_path / "train_2000.npy"
    np.save(cache_path, np.zeros((1, 512), dtype=np.float32))
    cache_path.with_name(cache_path.name + ".json").write_text(
        json.dumps({"complete": False, "row_count": 1})
    )
    resolved = prep.ResolvedSet(
        name="train_2000",
        smiles=("CC",),
        targets=np.array([0.3]),
        extras={},
        source=tmp_path / "source.csv",
        counts={},
    )
    assert prep.cache_is_current(cache_path, resolved) is False


def test_eval_set_specs_cover_the_four_standalone_sets(tmp_path: Path) -> None:
    """The training sets come from the 10M set; these four come from their own files."""
    specs = prep.eval_set_specs(
        val_csv=tmp_path / "val.csv",
        gp_molformer_csv=tmp_path / "gen.csv",
        ampc_331k_csv=tmp_path / "331k.csv",
        olivier_csv=tmp_path / "olivier.csv",
    )
    assert [spec.name for spec in specs] == [
        "val_set",
        "gp_molformer_set",
        "olivier_invitro",
        "ampc_331k",
    ]
    by_name = {spec.name: spec for spec in specs}
    assert by_name["val_set"].extra_columns == ("weight",)
    assert by_name["gp_molformer_set"].docked_sa_filter is True
    assert by_name["olivier_invitro"].label_column is None


def test_train_sizes_must_fit_inside_the_top_selection() -> None:
    """The top-n sets are prefixes of train_top, so n must not exceed it."""
    with pytest.raises(SystemExit, match="must cover the largest training size"):
        prep._parse_args(["--train-sizes", "20000", "--n-train-top", "10000"])


def test_train_sizes_are_parsed_as_a_list() -> None:
    """One prep run prepares both arms' training sets."""
    args = prep._parse_args(["--train-sizes", "2000,3000"])
    assert args.train_sizes == (2000, 3000)
    assert args.selected_sets is None


def test_sets_flag_restricts_the_work() -> None:
    """A timed-out prep job must be resumable set by set."""
    args = prep._parse_args(["--sets", "ampc_331k, gp_molformer_set"])
    assert args.selected_sets == {"ampc_331k", "gp_molformer_set"}
