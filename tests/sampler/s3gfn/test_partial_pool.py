"""A round that hits the attempt cap carries a big-enough partial pool forward."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from activelearning.sampler.config import S3GFNSamplerConfig
from tests.sampler.s3gfn.conftest import FakeChem


class StallingModel:
    """Yields ``unique`` distinct molecules in total, then only repeats of them."""

    pad_token_id = 0

    def __init__(self, unique: int) -> None:
        self.policy = nn.Linear(1, 1)
        self.prior = nn.Linear(1, 1)
        self.unique = unique
        self.next = 0
        self.calls: list[int] = []

    def generate(self, count, max_length, temperature):
        del max_length, temperature
        self.calls.append(count)
        smiles = tuple(f"C{min(self.next + i, self.unique - 1)}" for i in range(count))
        self.next += count
        return SimpleNamespace(
            smiles=smiles, fidelity_indices=torch.zeros(count, dtype=torch.long)
        )


def _sampler(make_sampler, **overrides):
    settings = dict(
        n_samples=10,
        fidelities=[1],
        batch_size=99,
        generation_batch_size=4,
        max_generation_attempts=20,
    )
    settings.update(overrides)
    return make_sampler(**settings)


def test_partial_pool_at_or_above_the_floor_is_carried_forward(make_sampler, caplog):
    sampler = _sampler(make_sampler, min_pool_size=5)
    model = StallingModel(unique=7)

    with caplog.at_level(logging.WARNING):
        candidates = sampler._generate_final_candidates(
            model=model, molecule_chem=FakeChem
        )

    assert [c.x for c in candidates] == [f"C{i}" for i in range(7)]
    assert sum(model.calls) == 20  # ran to the cap before giving up
    assert sampler.round_metrics.final_candidates == tuple(candidates)
    assert any("carrying the partial pool forward" in r.message for r in caplog.records)


def test_a_pool_exactly_at_the_floor_is_carried_forward(make_sampler):
    sampler = _sampler(make_sampler, min_pool_size=7)
    candidates = sampler._generate_final_candidates(
        model=StallingModel(unique=7), molecule_chem=FakeChem
    )
    assert len(candidates) == 7


def test_a_pool_below_the_floor_still_fails_the_round(make_sampler):
    sampler = _sampler(make_sampler, min_pool_size=8)
    with pytest.raises(RuntimeError, match=r"min_pool_size=8"):
        sampler._generate_final_candidates(
            model=StallingModel(unique=7), molecule_chem=FakeChem
        )
    assert sampler.round_metrics.final_candidates == ()


def test_without_a_floor_any_shortfall_still_fails_the_round(make_sampler):
    sampler = _sampler(make_sampler)  # min_pool_size unset: strict
    with pytest.raises(RuntimeError, match="after 20 attempts"):
        sampler._generate_final_candidates(
            model=StallingModel(unique=9), molecule_chem=FakeChem
        )


def test_generation_still_stops_the_moment_the_pool_is_full(make_sampler, caplog):
    sampler = _sampler(make_sampler, min_pool_size=5, max_generation_attempts=500)
    model = StallingModel(unique=1000)

    with caplog.at_level(logging.WARNING):
        candidates = sampler._generate_final_candidates(
            model=model, molecule_chem=FakeChem
        )

    assert len(candidates) == 10
    assert sum(model.calls) == 12  # three batches of four, not 500 attempts
    assert not any("partial pool" in r.message for r in caplog.records)


def test_the_round_records_that_it_was_partial(
    make_sampler, fake_acquisition, patch_molecule_dependencies, tmp_path
):
    sampler = _sampler(
        make_sampler,
        min_pool_size=5,
        generation_summary_path=str(tmp_path / "gen_{round}.json"),
    )
    sampler._new_round_model = lambda: StallingModel(unique=7)
    sampler._train_round = lambda **kwargs: None

    candidates = sampler.sample(
        acquisition=fake_acquisition, cost_fn=lambda cs: [1.0] * len(cs)
    )

    assert len(candidates) == 7
    (path,) = tmp_path.glob("gen_*.json")
    summary = json.loads(path.read_text(encoding="utf-8"))
    assert summary["reached_target"] is False
    assert summary["partial_pool"] is True
    assert summary["generation"]["cap_reached"] is True
    assert summary["generation"]["accepted_count"] == 7

    metrics, _ = sampler.drain_round_diagnostics(include_figures=False, max_points=10)
    assert metrics["sampler/s3gfn/generation/pool_size"] == 7
    assert metrics["sampler/s3gfn/generation/partial_pool"] == 1


def test_a_full_pool_is_not_reported_as_partial(
    make_sampler, fake_acquisition, patch_molecule_dependencies, tmp_path
):
    sampler = _sampler(
        make_sampler,
        min_pool_size=5,
        max_generation_attempts=500,
        generation_summary_path=str(tmp_path / "gen_{round}.json"),
    )
    sampler._new_round_model = lambda: StallingModel(unique=1000)
    sampler._train_round = lambda **kwargs: None

    sampler.sample(acquisition=fake_acquisition, cost_fn=lambda cs: [1.0] * len(cs))

    (path,) = tmp_path.glob("gen_*.json")
    summary = json.loads(path.read_text(encoding="utf-8"))
    assert summary["reached_target"] is True and summary["partial_pool"] is False
    metrics, _ = sampler.drain_round_diagnostics(include_figures=False, max_points=10)
    assert metrics["sampler/s3gfn/generation/pool_size"] == 10
    assert metrics["sampler/s3gfn/generation/partial_pool"] == 0


@pytest.mark.parametrize("bad", [0, -1, 11])
def test_constructor_rejects_a_floor_outside_one_to_n_samples(make_sampler, bad):
    with pytest.raises(ValueError, match="min_pool_size"):
        _sampler(make_sampler, min_pool_size=bad)


def test_config_passes_the_floor_through_and_checks_it():
    assert S3GFNSamplerConfig(n_samples=100).min_pool_size is None
    assert S3GFNSamplerConfig(n_samples=100, min_pool_size=10).min_pool_size == 10
    with pytest.raises(ValueError, match="cannot exceed"):
        S3GFNSamplerConfig(n_samples=100, min_pool_size=101)
    with pytest.raises(ValueError):
        S3GFNSamplerConfig(n_samples=100, min_pool_size=0)
