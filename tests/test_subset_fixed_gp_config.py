"""The AmpC multi-round config's training-set settings agree with each other."""

import csv
from pathlib import Path

import numpy as np
import pytest

from activelearning.applications.molecules.hit_rate import DEFAULT_HIT_THRESHOLD
from activelearning.utils.config_loader import load_config

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "config/ampc/subset_fixed_gp.yaml"


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


def test_band_edges_are_rederived_from_the_archive_every_round(cfg):
    """Unpinned and refreshed: a function of the archive being cut down each round."""
    subset = cfg["surrogate"]["training_subset"]
    assert subset.get("band_edges") is None
    assert subset["refresh_band_edges"] is True


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



def test_the_top_edge_is_pinned_below_the_hit_cutoff(cfg):
    subset = cfg["surrogate"]["training_subset"]
    assert subset["top_band_edge"] < subset["keep_above"]
    # pProp 4.5 on the y scale, per the fitted hit-rate model (4.0 -> 0.349461,
    # 5.0 -> 0.524739 are the anchors it reproduces).
    assert subset["top_band_edge"] == pytest.approx(0.4389967546609654)


def test_on_the_real_start_set_the_top_band_is_the_near_hit_tier(cfg, start_targets):
    """With the shipped settings the top band holds rows, and only near-hit ones."""
    import torch

    from activelearning.surrogate.encoder import FixedEncoder
    from activelearning.surrogate.training_subset import StratifiedFPSSubsetSelectorConfig
    from activelearning.utils.types import Observation

    class _Index(FixedEncoder):
        feature_dim = 1

        def encode(self, values, *, device):
            return torch.tensor([[float(str(v)[1:])] for v in values], device=device)

    subset = dict(cfg["surrogate"]["training_subset"])
    subset["size"] = 3000  # forces a real subsetting pass on the 20k start set
    selector = StratifiedFPSSubsetSelectorConfig(**subset).build()
    archive = [Observation(x=f"m{i}", y=float(y)) for i, y in enumerate(start_targets)]

    chosen = selector.select(archive, encoder=_Index())

    keep, edge = subset["keep_above"], subset["top_band_edge"]
    near_hit = int(((start_targets >= edge) & (start_targets < keep)).sum())
    assert selector.last_diagnostics["bands"][-1]["n_band"] == near_hit > 0
    assert len(chosen.indices) == 3000
    assert chosen.top_count == int((start_targets >= keep).sum())
