"""Tests for the surrogate/acquisition training-set-size study script."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from activelearning.utils.types import Candidate, Observation
from scripts import surrogate_dataset_size_study as study

BAD_SMILES = "not-a-smiles"
POISON_SMILES = "CCBr"


class _FakeSurrogate:
    """Surrogate double: the mean is the SMILES length, and one SMILES raises."""

    def __init__(self) -> None:
        self.fit_count = 0
        self.predict_calls = 0

    def fit(self, observations: list[Observation]) -> None:
        """Record that a fit happened."""
        self.fit_count += 1

    def get_state_dict(self) -> None:
        """Report no state to save."""
        return None

    def predict(self, candidates: list[Candidate]) -> dict[str, list[float]]:
        """Predict ``len(smiles)`` with unit std; fail on the poison SMILES."""
        self.predict_calls += 1
        if any(candidate.x == POISON_SMILES for candidate in candidates):
            raise ValueError("MiniMol fingerprint contains non-finite values.")
        return {
            "mean": [float(len(candidate.x)) for candidate in candidates],
            "std": [1.0] * len(candidates),
        }


class _FakeAcquisition:
    """Acquisition double scoring each candidate as half its SMILES length."""

    supports_singleton_scoring = True

    def __init__(self) -> None:
        self.updated_with = None

    def update(self, surrogate: object, observations: list[Observation]) -> None:
        """Record the surrogate it was coupled to."""
        self.updated_with = surrogate

    def score(self, candidates: list[Candidate]) -> list[float]:
        """Return ``len(smiles) / 2``."""
        return [len(candidate.x) / 2 for candidate in candidates]


def _write_eval_csv(path: Path, smiles: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ID", "SMILES", "score"])
        for index, value in enumerate(smiles):
            writer.writerow([f"mol{index}", value, -index])


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture
def no_rdkit_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """Treat every SMILES except ``BAD_SMILES`` as valid, without RDKit."""
    monkeypatch.setattr(
        study,
        "valid_smiles_mask",
        lambda smiles: [value != BAD_SMILES for value in smiles],
    )


def test_parse_eval_spec_default_and_custom_column() -> None:
    """NAME=PATH uses the default SMILES column; a :COLUMN suffix overrides it."""
    assert study.parse_eval_spec("a=data/x.csv") == study.EvalSpec(
        "a", Path("data/x.csv"), "SMILES"
    )
    assert study.parse_eval_spec("b=y.csv:SMILE").smiles_column == "SMILE"
    with pytest.raises(Exception, match="NAME=PATH"):
        study.parse_eval_spec("missing-equals")


def test_run_study_writes_csvs_and_pngs(tmp_path: Path, no_rdkit_check: None) -> None:
    """Every eval set gets surrogate and acquisition CSVs and PNGs, in row order."""
    eval_path = tmp_path / "eval.csv"
    smiles = ["C", "CCO", BAD_SMILES, "c1ccccc1"]
    _write_eval_csv(eval_path, smiles)
    surrogate = _FakeSurrogate()
    acquisition = _FakeAcquisition()
    out_dir = tmp_path / "out" / "10k"

    study.run_study(
        surrogate,
        acquisition,
        [Observation(x="C", y=0.1, fidelity=1)],
        [study.EvalSpec("evalset", eval_path)],
        out_dir,
        train_label="10k",
        fidelity=1,
        chunk_size=2,
    )

    assert surrogate.fit_count == 1
    assert acquisition.updated_with is surrogate
    for kind in ("surrogate", "acquisition"):
        assert (out_dir / f"evalset_{kind}.csv").is_file()
        assert (out_dir / f"evalset_{kind}.png").stat().st_size > 0

    surrogate_rows = _read_rows(out_dir / "evalset_surrogate.csv")
    assert [row["SMILES"] for row in surrogate_rows] == smiles
    assert list(surrogate_rows[0]) == ["ID", "SMILES", "score", "pred_mean", "pred_std"]
    means = [float(row["pred_mean"]) for row in surrogate_rows]
    assert means[:2] == [1.0, 3.0]
    assert math.isnan(means[2])
    assert means[3] == 8.0

    acquisition_rows = _read_rows(out_dir / "evalset_acquisition.csv")
    scores = [float(row["acquisition_score"]) for row in acquisition_rows]
    assert scores[0] == 0.5
    assert math.isnan(scores[2])


def test_failing_chunk_is_retried_per_molecule(no_rdkit_check: None) -> None:
    """A failing chunk is rescored per molecule; only the culprit is NaN."""
    surrogate = _FakeSurrogate()
    results = study.score_eval_set(
        ["C", POISON_SMILES, "CCC"],
        study.surrogate_score_fn(surrogate),
        study.SURROGATE_COLUMNS,
        fidelity=1,
        chunk_size=3,
        label="test",
    )

    assert results["pred_mean"][0] == 1.0
    assert math.isnan(results["pred_mean"][1])
    assert math.isnan(results["pred_std"][1])
    assert results["pred_mean"][2] == 3.0
    # One failed chunk call, then three single-molecule retries.
    assert surrogate.predict_calls == 4


def test_plot_only_rebuilds_pngs_from_csvs(tmp_path: Path) -> None:
    """--plot-only draws the histograms from existing CSVs without any fitting."""
    out_dir = tmp_path / "10m"
    out_dir.mkdir()
    for kind, column in (
        ("surrogate", "pred_mean"),
        ("acquisition", "acquisition_score"),
    ):
        with (out_dir / f"evalset_{kind}.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["SMILES", column])
            writer.writerows([["C", "0.1"], ["CC", "nan"], ["CCC", "0.3"]])

    study.main(
        [
            "--plot-only",
            "--train-label",
            "10m",
            "--output-dir",
            str(tmp_path),
            "--eval-csv",
            f"evalset={tmp_path / 'unused.csv'}",
        ]
    )

    assert (out_dir / "evalset_surrogate.png").stat().st_size > 0
    assert (out_dir / "evalset_acquisition.png").stat().st_size > 0


def test_valid_smiles_mask_uses_rdkit() -> None:
    """RDKit rejects unparsable SMILES and empty strings."""
    pytest.importorskip("rdkit")
    assert study.valid_smiles_mask(["CCO", BAD_SMILES, ""]) == [True, False, False]
