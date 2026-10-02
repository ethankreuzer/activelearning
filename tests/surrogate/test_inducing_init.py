"""Tests for stratified and k-means inducing-point initialization."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
import torch

from activelearning.surrogate.config import (
    VariationalGPSurrogateConfig,
    VariationalGPTrainingConfig,
)
from activelearning.surrogate.encoder import FixedEncoder
from activelearning.surrogate.inducing_init import (
    kmeans_centers,
    select_kmeans,
    select_stratified,
    stratum_counts,
    stratum_rows,
    validate_strata,
)
from activelearning.surrogate.variational_gp import VariationalGPSurrogate
from activelearning.utils.types import Observation


class _NumericFixedEncoder(FixedEncoder):
    """Return two-dimensional numeric inputs as fixed features."""

    feature_dim = 2

    def encode(
        self,
        values: Sequence[Any],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        """Convert numeric rows to a feature tensor."""
        return torch.as_tensor(values, dtype=torch.float32, device=device)


QUANTILES = (0.9, 0.99, 0.999)
FRACTIONS = (0.375, 0.25, 0.1875, 0.1875)


def _rows(n: int = 10_000) -> tuple[torch.Tensor, torch.Tensor]:
    """Return features equal to the row's target, so a row identifies its stratum."""
    targets = torch.arange(n, dtype=torch.float32)
    return targets[:, None].repeat(1, 3), targets[:, None]


def test_stratum_counts_sum_and_match_default_split() -> None:
    """The default shares give 24 / 16 / 12 / 12 of 64 and always sum to the total."""
    assert stratum_counts(64, FRACTIONS) == [24, 16, 12, 12]
    for total in (1, 7, 63, 100):
        assert sum(stratum_counts(total, FRACTIONS)) == total


def test_stratum_rows_cut_by_rank_with_deterministic_ties() -> None:
    """Strata partition the rows by target rank, ties broken by row order."""
    targets = torch.zeros(100)
    strata = stratum_rows(targets, (0.5,))
    assert [len(rows) for rows in strata] == [50, 50]
    assert strata[0].tolist() == list(range(50))
    ranked = stratum_rows(torch.arange(100.0).flip(0), (0.9,))
    assert sorted(ranked[1].tolist()) == list(range(10))


def test_select_stratified_draws_from_each_stratum() -> None:
    """Each stratum supplies its share of distinct rows."""
    features, targets = _rows()
    points = select_stratified(features, targets, 64, QUANTILES, FRACTIONS, seed=1)
    values = points[:, 0]
    assert points.shape == (64, 3)
    assert len(torch.unique(values)) == 64
    assert int((values < 9000).sum()) == 24
    assert int(((values >= 9000) & (values < 9900)).sum()) == 16
    assert int(((values >= 9900) & (values < 9990)).sum()) == 12
    assert int((values >= 9990).sum()) == 12


def test_select_stratified_is_seeded() -> None:
    """The same seed gives the same points, a different seed does not."""
    features, targets = _rows()
    first = select_stratified(features, targets, 64, QUANTILES, FRACTIONS, seed=3)
    again = select_stratified(features, targets, 64, QUANTILES, FRACTIONS, seed=3)
    other = select_stratified(features, targets, 64, QUANTILES, FRACTIONS, seed=4)
    assert torch.equal(first, again)
    assert not torch.equal(first, other)


def test_select_rejects_stratum_smaller_than_its_allotment() -> None:
    """A stratum with fewer rows than points is an error, not a silent shortfall."""
    features, targets = _rows(200)
    with pytest.raises(ValueError, match="allotted"):
        select_stratified(features, targets, 64, QUANTILES, FRACTIONS)


def test_kmeans_centers_find_separated_clusters() -> None:
    """Well separated blobs each receive one centre near their mean."""
    generator = torch.Generator().manual_seed(0)
    blobs = torch.cat(
        [
            torch.randn(200, 2, generator=generator) * 0.1 + offset
            for offset in (0.0, 10.0, -10.0)
        ]
    )
    centers = kmeans_centers(blobs, 3, generator=torch.Generator().manual_seed(1))
    truth = torch.tensor([[0.0, 0.0], [10.0, 10.0], [-10.0, -10.0]])
    nearest = torch.cdist(centers, truth)
    assert sorted(nearest.argmin(dim=1).tolist()) == [0, 1, 2]
    assert float(nearest.min(dim=1).values.max()) < 1.5


def test_select_kmeans_respects_strata_and_subsampling() -> None:
    """Centres stay inside their stratum's range and the count matches."""
    features, targets = _rows()
    points = select_kmeans(
        features, targets, 64, QUANTILES, FRACTIONS, max_rows=500, seed=2
    )
    values = points[:, 0]
    assert points.shape == (64, 3)
    assert int((values >= 9990).sum()) == 12
    assert int((values < 9000).sum()) == 24


def test_validate_strata_rejects_bad_input() -> None:
    """Boundaries and shares must be consistent."""
    with pytest.raises(ValueError, match="increasing"):
        validate_strata((0.9, 0.5), (0.3, 0.3, 0.4))
    with pytest.raises(ValueError, match="one value per stratum"):
        validate_strata((0.9,), (1.0,))
    with pytest.raises(ValueError, match="sum to 1"):
        validate_strata((0.9,), (0.5, 0.2))


def test_config_defaults_and_validation() -> None:
    """The default stays uniform random and bad strata are rejected at validation."""
    base = {
        "encoder": {
            "type": "MiniMolAmpcSmilesFixedEncoder",
            "checkpoint_path": "minimol_resources/model/final.pt",
        },
    }
    config = VariationalGPSurrogateConfig.model_validate(base)
    assert config.inducing_init == "random"
    with pytest.raises(ValueError):
        VariationalGPSurrogateConfig.model_validate(
            {**base, "inducing_strata_fractions": [0.5, 0.5]}
        )
    with pytest.raises(ValueError):
        VariationalGPSurrogateConfig.model_validate({**base, "inducing_init": "grid"})


@pytest.mark.parametrize("mode", ["stratified", "kmeans"])
def test_surrogate_fits_with_each_init(mode: str) -> None:
    """A fit with either new init runs and keeps the inducing-point shape."""
    torch.manual_seed(0)
    surrogate = VariationalGPSurrogate(
        encoder=_NumericFixedEncoder(),
        training_params=VariationalGPTrainingConfig(epochs=2, lr=1e-2),
        num_inducing=4,
        inducing_init=mode,
        inducing_strata_quantiles=(0.5,),
        inducing_strata_fractions=(0.5, 0.5),
    )
    surrogate.fit([Observation(x=[float(i), float(i)], y=float(i)) for i in range(20)])
    assert surrogate.is_fitted()
    assert surrogate._gp_model is not None
    assert surrogate._gp_model.variational_strategy.inducing_points.shape == (4, 2)


def test_surrogate_rejects_bad_init_arguments() -> None:
    """Unknown modes and multi-fidelity use of the new modes fail at construction."""
    kwargs: dict[str, Any] = {
        "encoder": _NumericFixedEncoder(),
        "training_params": VariationalGPTrainingConfig(),
    }
    with pytest.raises(ValueError, match="inducing_init"):
        VariationalGPSurrogate(**kwargs, inducing_init="grid")
    with pytest.raises(ValueError, match="single-fidelity"):
        VariationalGPSurrogate(
            **kwargs, inducing_init="kmeans", is_multi_fidelity=True, target_fidelity=1
        )
