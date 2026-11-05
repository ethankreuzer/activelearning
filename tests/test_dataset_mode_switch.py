"""Tests for switching how labeled data is managed over active-learning rounds.

The loop can either subset a growing archive to a fixed size each round
(``StratifiedFPS``) or fit the whole thing (``FullArchive``). The two selectors
are covered in ``tests/surrogate/test_training_subset.py``; what is covered here
is the *switch* -- the config mechanics by which a job selects one, which have a
sharp edge worth pinning down.

OmegaConf merges mappings recursively, so an override cannot replace a node, only
add to it. The two regimes have disjoint schemas and
``FullArchiveSubsetSelectorConfig`` sets ``extra="forbid"``, so overriding
``surrogate.training_subset`` in place leaves the StratifiedFPS keys beside the
new ``type`` and the config is refused. The switch therefore takes two pieces:
``config/ampc/overrides/full_archive.yaml`` nulls the node, and the override then
merges onto nothing and replaces it.

These tests exist because that is easy to get wrong, invisible when read, and
would otherwise be discovered by a job that dies at config load after queueing.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf
from pydantic import TypeAdapter, ValidationError

from activelearning.surrogate.training_subset import (
    FullArchiveSubsetSelector,
    FullArchiveSubsetSelectorConfig,
    StratifiedFPSSubsetSelectorConfig,
    TrainingSubsetSelectorConfig,
)
from activelearning.utils.config_loader import load_config

BASE_CONFIG = Path("config/ampc/subset_fixed_gp.yaml")
CLEAR_OVERLAY = Path("config/ampc/overrides/full_archive.yaml")
#: Exactly the override jobs/single_fidelity_al_nibi.sh passes for `full`.
FULL_ARCHIVE_OVERRIDE = (
    "surrogate.training_subset={type: FullArchive, top_threshold: 0.5, max_size: 46000}"
)

_ADAPTER = TypeAdapter(TrainingSubsetSelectorConfig)


def _training_subset(paths: list[Path], overrides: list[str] | None = None) -> object:
    """Return the merged config's training_subset node as a plain container.

    ``to_container`` raises on a null node rather than returning ``None``, and a
    null node is one of the states under test, so it is handled here.
    """
    cfg = load_config([str(path) for path in paths], overrides=overrides)
    node = cfg.surrogate.training_subset
    return None if node is None else OmegaConf.to_container(node, resolve=True)


def test_the_shipped_config_is_the_subset_regime() -> None:
    """Unswitched, the config is what the completed 10-round run ran."""
    parsed = _ADAPTER.validate_python(_training_subset([BASE_CONFIG]))

    assert isinstance(parsed, StratifiedFPSSubsetSelectorConfig)
    assert parsed.keep_above == 0.5


def test_the_two_piece_switch_replaces_the_node() -> None:
    """Overlay plus override yields a FullArchive node and nothing else.

    The assertion that matters is the absence of `size`: it is what tells us the
    node was replaced rather than merged onto.
    """
    node = _training_subset(
        [BASE_CONFIG, CLEAR_OVERLAY], overrides=[FULL_ARCHIVE_OVERRIDE]
    )

    assert node == {"type": "FullArchive", "top_threshold": 0.5, "max_size": 46000}

    parsed = _ADAPTER.validate_python(node)
    selector = parsed.build()
    assert isinstance(parsed, FullArchiveSubsetSelectorConfig)
    assert isinstance(selector, FullArchiveSubsetSelector)
    assert selector.top_threshold == 0.5
    assert selector.max_size == 46000


def test_the_override_alone_is_refused() -> None:
    """Without the overlay the StratifiedFPS keys survive the merge.

    This is the failure the overlay exists to prevent, and it must stay loud: a
    config carrying both a FullArchive type and a subset size does not say which
    regime was intended.
    """
    node = _training_subset([BASE_CONFIG], overrides=[FULL_ARCHIVE_OVERRIDE])

    assert node["type"] == "FullArchive"
    assert node["size"] == 25000  # the merge left it behind

    with pytest.raises(ValidationError):
        _ADAPTER.validate_python(node)


def test_the_overlay_alone_is_refused() -> None:
    """Without the override the node is null, which no selector validates."""
    node = _training_subset([BASE_CONFIG, CLEAR_OVERLAY])

    assert node is None

    with pytest.raises(ValidationError):
        _ADAPTER.validate_python(node)
