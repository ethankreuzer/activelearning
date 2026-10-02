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


def test_learned_hyperparameters_reports_the_prior_std_and_covariance_spread() -> None:
    """A fresh GP has an identity inducing covariance: both eigenvalues are 1."""
    import gpytorch
    import torch

    from activelearning.surrogate.variational_gp import _VariationalGP

    class _Fitted:
        def __init__(self) -> None:
            self._gp_model = _VariationalGP(input_dim=2, num_inducing=4)
            self._likelihood = gpytorch.likelihoods.GaussianLikelihood()

        def get_state_dict(self) -> dict[str, torch.Tensor]:
            return {
                "outcome_mean": torch.tensor(0.5),
                "outcome_std": torch.tensor(2.0),
            }

    surrogate = _Fitted()

    result = fit_script.learned_hyperparameters(surrogate)

    outputscale = float(surrogate._gp_model.covar_module.outputscale)
    assert result["variational_covar_eig_min"] == pytest.approx(1.0)
    assert result["variational_covar_eig_max"] == pytest.approx(1.0)
    assert result["prior_std_original_scale"] == pytest.approx(outputscale**0.5 * 2.0)
    assert result["noise_std_original_scale"] == pytest.approx(
        result["noise"] ** 0.5 * 2.0
    )
    assert result["y_std"] == 2.0


def _args(*extra: str) -> list[str]:
    """Return a minimal argument list with the given extra options."""
    return ["config.yaml", "--output-dir", "out", *extra]


def test_parse_args_accepts_a_warm_start() -> None:
    """The three warm-start options parse together."""
    args, config_args = fit_script._parse_args(
        _args("--init-state", "s.pt", "--trainable", "variance", "--epoch-offset", "50")
    )

    assert args.init_state == Path("s.pt")
    assert args.trainable == "variance"
    assert args.epoch_offset == 50
    assert config_args == ["config.yaml"]


def test_parse_args_defaults_to_a_fit_from_scratch() -> None:
    """Without the options the run is an ordinary fit logged from step 0."""
    args, _ = fit_script._parse_args(_args())

    assert args.init_state is None
    assert args.trainable == "all"
    assert args.epoch_offset == 0


@pytest.mark.parametrize(
    "extra",
    [
        ("--trainable", "variance"),
        ("--init-state", "s.pt"),
        ("--init-state", "s.pt", "--epoch-offset", "0"),
        ("--epoch-offset", "-1"),
        ("--init-state", "s.pt", "--epoch-offset", "50", "--trainable", "mean"),
    ],
)
def test_parse_args_rejects_inconsistent_warm_start_options(
    extra: tuple[str, ...],
) -> None:
    """Freezing without a state, or a start that would log at a negative step, fails."""
    with pytest.raises(SystemExit):
        fit_script._parse_args(_args(*extra))


def test_check_init_state_accepts_no_state_and_an_existing_one(tmp_path: Path) -> None:
    """No state, or a state outside the output directory, passes."""
    state = tmp_path / "elbo" / fit_script.STATE_FILE
    state.parent.mkdir()
    state.write_bytes(b"")

    fit_script.check_init_state(None, tmp_path / "out")
    fit_script.check_init_state(state, tmp_path / "out")


def test_check_init_state_rejects_a_missing_file(tmp_path: Path) -> None:
    """A wrong path fails before any data is loaded."""
    with pytest.raises(SystemExit, match="does not exist"):
        fit_script.check_init_state(tmp_path / "missing.pt", tmp_path / "out")


def test_check_init_state_rejects_the_runs_own_output(tmp_path: Path) -> None:
    """Starting from the file the run will overwrite would destroy its own input."""
    state = tmp_path / fit_script.STATE_FILE
    state.write_bytes(b"")

    with pytest.raises(SystemExit, match="own output"):
        fit_script.check_init_state(state, tmp_path)
