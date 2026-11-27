"""Derivation of beta and the floor from a constant K."""

import csv
import json
import math
from pathlib import Path

import pytest

from activelearning.sampler.reward_calibration import (
    derive_reward_scale,
    log_scores_from_values,
    percentile,
)

REPO = Path(__file__).resolve().parents[2]
LADDER = REPO / "outputs/ampc/k_ladder"


def _ramp(n=20_000, lo=-100.0, hi=-50.0):
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]


def test_percentile_matches_numpy_default():
    np = pytest.importorskip("numpy")
    values = sorted(_ramp(10_001))
    for q in (0.0, 5.0, 50.0, 99.0, 100.0):
        assert percentile(values, q) == pytest.approx(float(np.percentile(values, q)))


def test_beta_is_ln_k_over_spread_and_k_is_recovered():
    scores = _ramp()
    scale = derive_reward_scale(scores, k=250.0)
    assert scale.spread == pytest.approx(scale.p99 - scale.median)
    assert scale.beta == pytest.approx(math.log(250.0) / scale.spread)
    # The defining property: the reward ratio p99/median is K again.
    assert math.exp(scale.beta * scale.spread) == pytest.approx(250.0)
    assert scale.floor == pytest.approx(percentile(sorted(scores), 5.0))
    assert scale.n == 20_000


def test_floor_percentile_is_respected():
    scores = _ramp()
    a = derive_reward_scale(scores, k=100.0, floor_percentile=5.0)
    b = derive_reward_scale(scores, k=100.0, floor_percentile=20.0)
    assert b.floor > a.floor
    assert b.beta == a.beta


def test_wider_spread_gives_smaller_beta_for_the_same_k():
    narrow = derive_reward_scale(_ramp(hi=-60.0), k=300.0)
    wide = derive_reward_scale(_ramp(hi=-40.0), k=300.0)
    assert wide.beta < narrow.beta


@pytest.mark.parametrize("bad", [1.0, 0.5, 0.0, -3.0, float("inf"), float("nan")])
def test_rejects_k_not_above_one(bad):
    with pytest.raises(ValueError, match="K must be"):
        derive_reward_scale(_ramp(), k=bad)


@pytest.mark.parametrize("bad", [-1.0, 50.0, 75.0])
def test_rejects_floor_percentile_not_below_median(bad):
    with pytest.raises(ValueError, match="floor_percentile"):
        derive_reward_scale(_ramp(), k=10.0, floor_percentile=bad)


def test_rejects_too_few_scores():
    with pytest.raises(ValueError, match="at least"):
        derive_reward_scale(_ramp(n=500), k=10.0)


def test_rejects_unranked_prior_sample():
    """An unfitted acquisition scores everything the same, so there is no spread."""
    with pytest.raises(ValueError, match="does not rank"):
        derive_reward_scale([-5.0] * 20_000, k=10.0)


def test_rejects_nan_and_plus_inf():
    for bad in (float("nan"), math.inf):
        with pytest.raises(ValueError, match="nan or"):
            derive_reward_scale(_ramp() + [bad], k=10.0)


def test_underflowed_tail_is_kept_as_minus_inf_not_dropped():
    """A zero score is a real molecule in the lower tail: it must move percentiles."""
    scores = _ramp()
    with_tail = [-math.inf] * 400 + scores  # 2% underflowed, below the 5th pct
    kept = derive_reward_scale(with_tail, k=10.0)
    dropped = derive_reward_scale(scores, k=10.0)
    assert kept.median != dropped.median
    assert math.isfinite(kept.floor)


def test_floor_inside_the_underflowed_tail_is_an_error():
    scores = [-math.inf] * 2_000 + _ramp()  # ~9% underflowed > 5%
    with pytest.raises(ValueError, match="no floor to derive"):
        derive_reward_scale(scores, k=10.0)


def test_median_inside_the_underflowed_tail_is_an_error():
    scores = [-math.inf] * 15_000 + _ramp(n=10_000)
    with pytest.raises(ValueError, match="median"):
        derive_reward_scale(scores, k=10.0)


def test_log_scores_from_values():
    logs = log_scores_from_values([1.0, math.e, 0.0])
    assert logs[0] == 0.0 and logs[1] == pytest.approx(1.0) and logs[2] == -math.inf
    for bad in (-1.0, float("nan"), math.inf):
        with pytest.raises(ValueError):
            log_scores_from_values([bad])


def _recorded_arms():
    arms = []
    if LADDER.is_dir():
        for path in sorted(LADDER.glob("kladder_*_s4[23]_steps5000/arm_params.json")):
            params = json.loads(path.read_text())
            csv_path = REPO / params.get("dist_csv", "")
            if csv_path.is_file():
                arms.append((params["arm"], params, csv_path))
    return arms


@pytest.mark.parametrize(
    "arm,params,csv_path", _recorded_arms() or [pytest.param(None, None, None, marks=pytest.mark.skip("no K-ladder outputs here"))]
)
def test_reproduces_recorded_k_ladder_arms(arm, params, csv_path):
    """The in-loop derivation matches the betas and floors the K ladder recorded."""
    with open(csv_path, newline="") as handle:
        scores = [float(row["score_gibbon_log"]) for row in csv.DictReader(handle)]
    scale = derive_reward_scale(scores, k=float(params["K"]))
    assert scale.n == params["dist_n"]
    assert scale.beta == pytest.approx(params["beta"], abs=1e-6)
    assert scale.median == pytest.approx(params["dist_median"], abs=1e-3)
    assert scale.p99 == pytest.approx(params["dist_p99"], abs=1e-3)
    # arm_params.json stores the floor rounded to one decimal.
    assert scale.floor == pytest.approx(params["log_score_floor"], abs=0.05)
