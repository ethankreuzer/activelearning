"""Tests for the docked set's validity and synthesizability profile."""

import pytest

from activelearning.monitoring.diagnostics import docked_chemistry_diagnostics
from activelearning.monitoring.diagnostics_config import DiagnosticsConfig
from activelearning.utils.types import Candidate

pytest.importorskip("rdkit", reason="The docked chemistry profile needs RDKit.")

#: Benzene: trivially synthesizable, SA score well under any usable threshold.
_EASY = "c1ccccc1"
#: Aspirin: a second easy molecule, so an all-pass case is not one row wide.
_EASY_TOO = "CC(=O)Oc1ccccc1C(=O)O"
#: Not parseable by RDKit, so it must count as invalid rather than raise.
_INVALID = "this is not a molecule"

_PREFIX = "active_learning/docked"


def _candidates(*smiles: str) -> list[Candidate]:
    """Wrap SMILES as the loop's selected candidates."""
    return [Candidate(x=value, fidelity=1) for value in smiles]


def test_an_all_valid_set_reports_full_validity() -> None:
    """Molecules RDKit parses count as valid, and the rate is their fraction."""
    metrics = docked_chemistry_diagnostics(
        _candidates(_EASY, _EASY_TOO), sa_threshold=4.0
    )

    assert metrics[f"{_PREFIX}/total"] == 2.0
    assert metrics[f"{_PREFIX}/valid_count"] == 2.0
    assert metrics[f"{_PREFIX}/validity_rate"] == 1.0


def test_an_unparseable_smiles_counts_as_invalid_without_raising() -> None:
    """A bad row lowers the rate; it does not take the round down with it."""
    metrics = docked_chemistry_diagnostics(
        _candidates(_EASY, _INVALID), sa_threshold=4.0
    )

    assert metrics[f"{_PREFIX}/validity_rate"] == 0.5
    assert metrics[f"{_PREFIX}/valid_count"] == 1.0
    assert metrics[f"{_PREFIX}/total"] == 2.0


def test_a_non_string_candidate_counts_as_invalid() -> None:
    """Enabling this on a non-molecular run must not break the loop."""
    metrics = docked_chemistry_diagnostics(
        [Candidate(x=(0.0, 1.0), fidelity=1)], sa_threshold=4.0
    )

    assert metrics[f"{_PREFIX}/validity_rate"] == 0.0
    assert metrics[f"{_PREFIX}/synthesizable_rate"] == 0.0
    # No SA score was computed, so no distribution is reported for one.
    assert f"{_PREFIX}/sa_score_mean" not in metrics


def test_easy_molecules_are_synthesizable_at_the_sampler_threshold() -> None:
    """The 4.0 the sampler trains against should pass simple ring systems."""
    metrics = docked_chemistry_diagnostics(
        _candidates(_EASY, _EASY_TOO), sa_threshold=4.0
    )

    assert metrics[f"{_PREFIX}/synthesizable_rate"] == 1.0
    assert metrics[f"{_PREFIX}/sa_threshold"] == 4.0


def test_the_threshold_is_strict_and_is_what_moves_the_rate() -> None:
    """A threshold below every score admits nothing, which proves the cut."""
    candidates = _candidates(_EASY, _EASY_TOO)
    passing = docked_chemistry_diagnostics(candidates, sa_threshold=4.0)
    # Every SA score is positive, so no molecule scores strictly below a
    # threshold at the bottom of the scale.
    none_pass = docked_chemistry_diagnostics(candidates, sa_threshold=1e-6)

    assert passing[f"{_PREFIX}/synthesizable_rate"] == 1.0
    assert none_pass[f"{_PREFIX}/synthesizable_rate"] == 0.0
    # Validity is a property of the molecule, not of the threshold.
    assert none_pass[f"{_PREFIX}/validity_rate"] == 1.0


def test_the_sa_distribution_accompanies_the_rate() -> None:
    """A flat rate can hide a drift toward the cut, so report the scores too."""
    metrics = docked_chemistry_diagnostics(
        _candidates(_EASY, _EASY_TOO), sa_threshold=4.0
    )

    mean = metrics[f"{_PREFIX}/sa_score_mean"]
    median = metrics[f"{_PREFIX}/sa_score_median"]
    maximum = metrics[f"{_PREFIX}/sa_score_max"]
    assert 0.0 < mean < 4.0
    assert mean == pytest.approx(median)
    assert maximum >= mean


def test_an_empty_selection_reports_nothing() -> None:
    """A round that selected nothing must not log a zero-of-zero rate."""
    assert docked_chemistry_diagnostics([], sa_threshold=4.0) == {}


def test_the_profile_is_off_unless_a_threshold_is_configured() -> None:
    """Opt-in, because it is the one diagnostic that requires RDKit."""
    assert DiagnosticsConfig().docked_sa_threshold is None


@pytest.mark.parametrize("threshold", [0.0, -1.0, float("nan"), float("inf")])
def test_a_nonsensical_threshold_is_refused_at_config_time(
    threshold: float,
) -> None:
    """Catch it when the config loads, not hours later in the first round."""
    with pytest.raises(ValueError, match="positive finite SA score"):
        DiagnosticsConfig(docked_sa_threshold=threshold)
