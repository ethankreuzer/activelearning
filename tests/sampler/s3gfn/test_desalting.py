"""Desalting: keep the largest fragment of a multi-fragment generation.

These use the real RDKit, because the behaviour under test is RDKit's own
fragment handling. ``FakeChem`` treats a SMILES as its own molecule and cannot
tell a salt from a parent.
"""

from __future__ import annotations

import pytest
import torch

import activelearning.sampler.s3gfn.sampler as sampler_module
from tests.sampler.s3gfn.conftest import (
    FakeAcquisition,
    FakeModel,
    FakeSynthesizability,
)

Chem = pytest.importorskip("rdkit.Chem")


def _canon(smiles: str, desalt: bool = True):
    return sampler_module._canonicalize_with_reason(
        smiles, molecule_chem=Chem, desalt=desalt
    )


def test_desalting_keeps_the_parent_and_drops_the_counter_ion() -> None:
    assert _canon("CCO.Cl") == ("CCO", "desalted")
    assert _canon("Cl.CCO") == ("CCO", "desalted")
    assert _canon("CCO.[Cl-].[Na+]") == ("CCO", "desalted")


def test_desalting_off_still_rejects_every_multi_fragment_sample() -> None:
    assert _canon("CCO.Cl", desalt=False) == (None, "disconnected")


def test_a_ring_closure_split_by_a_dot_is_one_molecule_not_a_desalt() -> None:
    # RDKit canonicalizes C1.C1 to CC, so there is nothing to strip.
    assert _canon("C1.C1") == ("CC", None)


def test_largest_fragment_is_chosen_by_heavy_atoms_not_string_length() -> None:
    # The first fragment is the longer string but has fewer heavy atoms: the
    # explicit hydrogens inflate the text, not the atom count.
    parent, reason = _canon("[H][H].[H]C([H])([H])C([H])([H])O")
    assert (parent, reason) == ("CCO", "desalted")


def test_equal_fragments_resolve_the_same_way_in_either_order() -> None:
    assert _canon("CC.CO") == _canon("CO.CC")


def test_a_molecule_with_no_heavy_atom_left_is_still_rejected() -> None:
    assert _canon("[H].[H]") == (None, "disconnected")


def test_parse_errors_and_empty_strings_are_unaffected_by_desalting() -> None:
    assert _canon("C1CC") == (None, "parse_error")
    assert _canon("") == (None, "empty")


def test_a_clean_molecule_is_not_labelled_desalted() -> None:
    assert _canon("OCC") == ("CCO", None)


def _prepare(sampler, smiles, input_ids):
    return sampler._prepare_batch(
        model=FakeModel(),
        smiles=smiles,
        fidelity_indices=torch.tensor([1] * len(smiles)),
        synthesizability=FakeSynthesizability(),
        molecule_chem=Chem,
        acquisition=FakeAcquisition(),
        cost_fn=None,
        generated_input_ids=input_ids,
    )


def test_a_desalted_row_is_scored_as_its_parent_and_its_salted_form_is_a_negative(
    make_sampler,
) -> None:
    sampler = make_sampler(desalt_largest_fragment=True)
    input_ids = torch.tensor([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])

    prepared = _prepare(
        sampler,
        ("CCO", "CCO.Cl", "invalid", "CC(C)O"),
        input_ids,
    )

    # Positives: the parent, trained on as the stripped SMILES.
    assert prepared.smiles == ("CCO", "CCO", "CC(C)O")
    assert prepared.desalted_count == 1
    # Negatives: the salted trajectory and the unparseable one, in batch order,
    # as the policy's own token ids.
    assert prepared.invalid_smiles == ("CCO.Cl", "invalid")
    assert prepared.invalid_input_ids.tolist() == [[2, 2, 2], [3, 3, 3]]


def test_without_desalting_a_salted_row_is_only_a_negative(make_sampler) -> None:
    sampler = make_sampler(desalt_largest_fragment=False)
    input_ids = torch.tensor([[1, 1, 1], [2, 2, 2]])

    prepared = _prepare(sampler, ("CCO", "CCO.Cl"), input_ids)

    assert prepared.smiles == ("CCO",)
    assert prepared.desalted_count == 0
    assert prepared.invalid_smiles == ("CCO.Cl",)


def test_desalted_rows_are_counted_by_the_reject_log_but_not_as_invalid(
    make_sampler,
) -> None:
    sampler = make_sampler(desalt_largest_fragment=True)
    _prepare(sampler, ("CCO.Cl", "CCO.Cl", "invalid"), torch.ones((3, 3)).long())

    counts = sampler.round_metrics.reject_counts
    assert counts["train/desalted"] == 2
    assert counts["train/parse_error"] == 1
    assert "train/disconnected" not in counts


def test_final_generation_desalts_and_dedupes_on_the_parent(make_sampler) -> None:
    sampler = make_sampler(n_samples=3, desalt_largest_fragment=True)
    candidates: list = []
    seen: set[str] = set()

    valid, invalid, duplicate = sampler._process_candidate_batch(
        smiles=("CCO", "CCO.Cl", "CC(C)O.[Na+]", "invalid"),
        fidelity_indices=torch.tensor([1, 1, 1, 1]),
        candidates=candidates,
        seen_smiles=seen,
        molecule_chem=Chem,
    )

    # CCO.Cl strips to a parent already seen, so it is a duplicate, not a new
    # candidate and not an invalid sample.
    assert [candidate.x for candidate in candidates] == ["CCO", "CC(C)O"]
    assert (valid, invalid, duplicate) == (2, 1, 1)


def test_the_step_log_gains_a_desalted_series_only_when_desalting_is_on(
    make_sampler,
) -> None:
    on = make_sampler(desalt_largest_fragment=True)
    _prepare(on, ("CCO.Cl",), torch.ones((1, 3)).long())
    off = make_sampler(desalt_largest_fragment=False)

    on.round_metrics.record_training_step(
        generated_count=1,
        valid_count=1,
        synthesizable_count=1,
        online_loss=None,
        replay_loss=None,
        auxiliary_loss=None,
        log_z=0.0,
        raw_reward_scores=[1.0],
        desalted_count=1,
    )
    off.round_metrics.record_training_step(
        generated_count=1,
        valid_count=1,
        synthesizable_count=1,
        online_loss=None,
        replay_loss=None,
        auxiliary_loss=None,
        log_z=0.0,
        raw_reward_scores=[1.0],
    )

    assert on.round_metrics.desalted_counts == [1]
    assert off.round_metrics.desalted_counts == []


def test_the_sampler_config_passes_the_flag_through() -> None:
    from activelearning.sampler.config import S3GFNSamplerConfig

    default = S3GFNSamplerConfig(n_samples=1, fidelities=[1])
    assert default.desalt_largest_fragment is False
    enabled = S3GFNSamplerConfig(
        n_samples=1, fidelities=[1], desalt_largest_fragment=True
    )
    assert enabled.desalt_largest_fragment is True
