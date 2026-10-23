"""Tests for the fixed evaluation sets the surrogate is scored on each round."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import Tensor

from activelearning.monitoring.diagnostics import evaluate_held_out_set
from activelearning.monitoring.held_out import (
    HeldOutSet,
    HeldOutSetConfig,
    load_held_out_set,
)
from activelearning.surrogate.encoder import FixedEncoder
from activelearning.surrogate.fixed_feature_gp import FixedFeatureGPSurrogate
from activelearning.utils.types import Candidate, Observation


class _LengthEncoder(FixedEncoder):
    """Two features per molecule, so a GP can be fitted without chemistry."""

    feature_dim = 2

    def encode(self, values: Sequence[Any], *, device: torch.device) -> Tensor:
        """Count characters and carbons."""
        return torch.tensor(
            [[len(str(value)), str(value).count("C")] for value in values],
            dtype=torch.float32,
            device=device,
        )


def _write_csv(path: Path, rows: Sequence[str]) -> Path:
    """Write a held-out CSV with the default column names."""
    path.write_text("SMILES,y\n" + "".join(rows), encoding="utf-8")
    return path


def test_load_keeps_only_rows_with_a_finite_target(tmp_path: Path) -> None:
    """A metric over a missing target is undefined, so the row is dropped."""
    path = _write_csv(
        tmp_path / "set.csv",
        ["CC,0.1\n", "CCC,\n", "CCCC,nan\n", "CCCCC,0.5\n", "CO,inf\n"],
    )

    held_out = load_held_out_set(name="fixed", path=path)

    assert [candidate.x for candidate in held_out.candidates] == ["CC", "CCCCC"]
    assert held_out.targets.tolist() == [0.1, 0.5]
    assert {candidate.fidelity for candidate in held_out.candidates} == {1}


def test_load_subsamples_evenly_rather_than_from_the_front(tmp_path: Path) -> None:
    """These files are often target-sorted, so a prefix would be one end only."""
    path = _write_csv(
        tmp_path / "set.csv", [f"C{index},{index / 10}\n" for index in range(10)]
    )

    held_out = load_held_out_set(name="fixed", path=path, max_rows=3)

    assert held_out.targets.tolist() == [0.0, 0.4, 0.9]


def test_load_rejects_a_missing_column(tmp_path: Path) -> None:
    """A typo in a column name must fail loudly, not score zero rows."""
    path = _write_csv(tmp_path / "set.csv", ["CC,0.1\n"])

    with pytest.raises(ValueError, match="has no 'target' column"):
        load_held_out_set(name="fixed", path=path, y_column="target")


def test_load_rejects_a_missing_file(tmp_path: Path) -> None:
    """Checked before the first round, not after hours of docking."""
    with pytest.raises(FileNotFoundError, match="is not a file"):
        load_held_out_set(name="fixed", path=tmp_path / "absent.csv")


def test_load_rejects_a_set_with_no_finite_target(tmp_path: Path) -> None:
    """Nothing to score is a configuration error."""
    path = _write_csv(tmp_path / "set.csv", ["CC,\n", "CCC,nan\n"])

    with pytest.raises(ValueError, match="no row with a finite target"):
        load_held_out_set(name="fixed", path=path)


def test_config_rejects_a_name_that_is_not_a_metric_segment() -> None:
    """The name becomes a segment of a monitoring key."""
    with pytest.raises(ValueError):
        HeldOutSetConfig(name="AmpC 331k", path="set.csv")


def _fitted_surrogate() -> tuple[FixedFeatureGPSurrogate, list[str], list[float]]:
    """Fit a GP on a set its length feature explains exactly."""
    molecules = ["C" * (1 + index % 7) + "O" * (index % 3) for index in range(40)]
    targets = [0.1 * len(molecule) for molecule in molecules]
    surrogate = FixedFeatureGPSurrogate(_LengthEncoder(), ard=False)
    surrogate.fit(
        [
            Observation(x=molecule, y=target, fidelity=1)
            for molecule, target in zip(molecules, targets)
        ]
    )
    return surrogate, molecules, targets


def _held_out(
    name: str, molecules: Sequence[str], targets: Sequence[float]
) -> HeldOutSet:
    """Build a held-out set from molecules and their targets."""
    return HeldOutSet(
        name=name,
        candidates=tuple(Candidate(x=molecule, fidelity=1) for molecule in molecules),
        targets=np.asarray(targets, dtype=np.float64),
    )


def test_evaluation_reports_the_offline_metric_set_and_the_figure() -> None:
    """The numbers the offline arms report, so rounds compare against them."""
    surrogate, molecules, targets = _fitted_surrogate()
    held_out = _held_out("ampc_331k", molecules, targets)

    metrics, figures = evaluate_held_out_set(
        surrogate, held_out, include_figure=True, chunk_size=7
    )

    prefix = "surrogate/eval_set/ampc_331k"
    for name in ("pearson", "r2", "coverage_1std", "coverage_2std"):
        assert f"{prefix}/{name}" in metrics
    assert metrics[f"{prefix}/count"] == len(molecules)
    # The length feature explains this target exactly, so the fit must rank it.
    assert metrics[f"{prefix}/pearson"] > 0.9
    assert 0.0 <= metrics[f"{prefix}/coverage_1std"] <= 1.0
    assert list(figures) == [f"{prefix}/predicted_vs_observed"]


def test_evaluation_chunking_does_not_change_the_metrics() -> None:
    """The set is scored in chunks; the result must not depend on the size."""
    surrogate, molecules, targets = _fitted_surrogate()
    held_out = _held_out("fixed", molecules, targets)

    few, _ = evaluate_held_out_set(
        surrogate, held_out, include_figure=False, chunk_size=3
    )
    many, _ = evaluate_held_out_set(
        surrogate, held_out, include_figure=False, chunk_size=1000
    )

    assert few == pytest.approx(many)


def test_evaluation_without_a_figure_still_reports_metrics() -> None:
    """Figures are rendered on an interval; the scalars are every round."""
    surrogate, molecules, targets = _fitted_surrogate()
    held_out = _held_out("fixed", molecules, targets)

    metrics, figures = evaluate_held_out_set(surrogate, held_out, include_figure=False)

    assert figures == {}
    assert "surrogate/eval_set/fixed/pearson" in metrics


def test_evaluation_of_an_unfitted_surrogate_reports_nothing() -> None:
    """A surrogate that cannot predict costs the round no other diagnostics."""
    surrogate = FixedFeatureGPSurrogate(_LengthEncoder(), ard=False)
    held_out = _held_out("fixed", ["CC"], [0.2])

    metrics, figures = evaluate_held_out_set(surrogate, held_out, include_figure=True)

    assert metrics == {}
    assert figures == {}
