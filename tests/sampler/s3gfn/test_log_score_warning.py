"""S3-GFN warns when log-scale acquisition scores meet a large beta."""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import pytest

import activelearning.sampler.s3gfn.sampler as sampler_module


def _acquisition(scale: str) -> SimpleNamespace:
    return SimpleNamespace(supports_singleton_scoring=True, score_scale=scale)


@pytest.mark.parametrize(
    ("scale", "beta", "warns"),
    [("log", 100.0, True), ("log", 1.0, False), ("value", 100.0, False)],
)
def test_warns_only_for_log_scores_with_large_beta(scale: str, beta: float, warns: bool) -> None:
    sampler = SimpleNamespace(beta=beta)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sampler_module.S3GFNSampler._validate_acquisition(sampler, _acquisition(scale))
    messages = [str(w.message) for w in caught if "log-scale scores" in str(w.message)]
    assert bool(messages) is warns


def test_acquisition_without_score_scale_is_treated_as_value() -> None:
    sampler = SimpleNamespace(beta=100.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sampler_module.S3GFNSampler._validate_acquisition(
            sampler, SimpleNamespace(supports_singleton_scoring=True)
        )
