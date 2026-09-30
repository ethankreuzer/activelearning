"""Tests for the surrogate evaluation fit script."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from activelearning.utils.types import Observation
from scripts import surrogate_eval_fit as fit_script


def test_select_train_eval_rows_is_seeded_and_sorted() -> None:
    """The random rows are reproducible, unique and in ascending order."""
    targets = np.linspace(0.0, 1.0, 50)

    first, _ = fit_script.select_train_eval_rows(targets, 10, 0, seed=3)
    second, _ = fit_script.select_train_eval_rows(targets, 10, 0, seed=3)
    other, _ = fit_script.select_train_eval_rows(targets, 10, 0, seed=4)

    assert first.tolist() == second.tolist()
    assert first.tolist() != other.tolist()
    assert first.tolist() == sorted(set(first.tolist()))
    assert len(first) == 10


def test_select_train_eval_rows_top_is_highest_target_first() -> None:
    """The top rows are the largest targets, highest first."""
    targets = [0.1, 0.9, 0.4, 0.8, 0.2]

    _, top = fit_script.select_train_eval_rows(targets, 0, 3, seed=0)

    assert top.tolist() == [1, 3, 2]


def test_select_train_eval_rows_breaks_ties_by_row_order() -> None:
    """Equal targets at the cut-off keep the earliest rows."""
    targets = [0.5, 0.9, 0.5, 0.5, 0.1]

    _, top = fit_script.select_train_eval_rows(targets, 0, 3, seed=0)

    assert top.tolist() == [1, 0, 2]


def test_select_train_eval_rows_clamps_to_dataset_size() -> None:
    """Asking for more rows than exist returns every row once."""
    targets = [0.3, 0.2, 0.1]

    random_rows, top = fit_script.select_train_eval_rows(targets, 10, 10, seed=1)

    assert random_rows.tolist() == [0, 1, 2]
    assert top.tolist() == [0, 1, 2]


@pytest.mark.parametrize(
    ("targets", "n_random", "n_top"),
    [([], 1, 1), ([[0.1]], 1, 1), ([0.1], -1, 1), ([0.1], 1, -1)],
)
def test_select_train_eval_rows_rejects_invalid_input(
    targets: list, n_random: int, n_top: int
) -> None:
    """Empty, non-1D or negative-count inputs are rejected."""
    with pytest.raises(ValueError):
        fit_script.select_train_eval_rows(targets, n_random, n_top, seed=0)


def test_write_eval_csv_writes_selected_rows_in_order(tmp_path: Path) -> None:
    """The CSV holds the selected SMILES and targets, in the given order."""
    observations = [
        Observation(x="CCO", y=0.25),
        Observation(x="CCN", y=0.5),
        Observation(x="CCC", y=0.75),
    ]
    path = tmp_path / "out" / "rows.csv"

    fit_script.write_eval_csv(path, observations, np.array([2, 0]))

    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows == [["SMILE", "y"], ["CCC", "0.75"], ["CCO", "0.25"]]
    assert not path.with_name("rows.csv.tmp").exists()


def test_write_eval_csv_rejects_non_string_inputs(tmp_path: Path) -> None:
    """Only SMILES-string inputs can be written."""
    observations = [Observation(x=[1.0, 2.0], y=0.5)]

    with pytest.raises(ValueError, match="non-string"):
        fit_script.write_eval_csv(tmp_path / "rows.csv", observations, [0])


def test_learned_hyperparameters_is_empty_without_a_gp() -> None:
    """A surrogate that is not a fitted variational GP reports nothing."""

    class _Unfitted:
        def get_state_dict(self) -> None:
            return None

    assert fit_script.learned_hyperparameters(_Unfitted()) == {}
