"""The AmpC multi-round config's training-set settings agree with each other."""

import csv
from pathlib import Path

import numpy as np
import pytest

from activelearning.applications.molecules.hit_rate import DEFAULT_HIT_THRESHOLD
from activelearning.utils.config_loader import load_config

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "config/ampc/subset_fixed_gp.yaml"
QUANTILES = (50.0, 75.0, 90.0, 95.0, 99.0, 99.75)


@pytest.fixture(scope="module")
def cfg():
    from omegaconf import OmegaConf

    return OmegaConf.to_container(load_config(CONFIG), resolve=True)


def test_fits_the_whole_archive_up_to_40k_then_subsets(cfg):
    subset = cfg["surrogate"]["training_subset"]
    assert subset["type"] == "StratifiedFPS"
    assert subset["size"] == 40000


def test_a_hit_is_pprop_5(cfg):
    """`keep_above` is the hit threshold: pProp >= 5, not the older y >= 0.5."""
    assert cfg["surrogate"]["training_subset"]["keep_above"] == pytest.approx(
        DEFAULT_HIT_THRESHOLD
    )


def test_exact_cholesky_covers_the_largest_fit(cfg):
    """Above max_cholesky_size gpytorch silently falls back to approximate CG."""
    assert cfg["surrogate"]["max_cholesky_size"] >= cfg["surrogate"]["training_subset"]["size"]




@pytest.fixture(scope="module")
def start_targets(cfg):
    path = REPO / cfg["dataset"]["initial_data"]["path"]
    if not path.is_file():
        pytest.skip(f"{path} is not on this machine")
    column = cfg["dataset"]["initial_data"]["y_column"]
    with path.open(newline="") as handle:
        return np.array([float(row[column]) for row in csv.DictReader(handle)])


def test_the_start_set_fits_and_holds_every_hit(cfg, start_targets):
    size = cfg["surrogate"]["training_subset"]["size"]
    hits = int((start_targets >= cfg["surrogate"]["training_subset"]["keep_above"]).sum())
    assert len(start_targets) <= size  # round 1 fits the whole start set
    assert 0 < hits < size  # a hit tier that could not exceed the budget


def test_pinned_band_edges_are_the_start_sets_round_zero_quantiles(cfg, start_targets):
    pinned = cfg["surrogate"]["training_subset"]["band_edges"]
    derived = np.percentile(start_targets, QUANTILES)
    assert pinned == pytest.approx(derived.tolist(), abs=6e-6)
