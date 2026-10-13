from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from torch import nn
import pytest
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from activelearning.runtime import RuntimeContext
import activelearning.sampler.s3gfn.logging as logging_module
import activelearning.sampler.s3gfn.sampler as sampler_module
from activelearning.utils.types import Candidate

from tests.sampler.s3gfn.conftest import (
    FakeAcquisition,
    FakeChem,
    FakeModel,
    FakeSynthesizability,
    FakeTokenizer,
)


class _FakeBatchAcquisition:
    """Acquisition that cannot score candidates one at a time."""

    supports_singleton_scoring = False


class _GradientTrainingModel:
    def __init__(self) -> None:
        self.policy = nn.Linear(1, 1, bias=False)
        self.fidelity_head = nn.Linear(1, 1, bias=False)
        self.log_z = nn.Parameter(torch.tensor(0.5))
        self.pad_token_id = 0
        self.device = torch.device("cpu")
        self.last_auxiliary_loss = None

    def generate(self, count, max_length, temperature):
        assert count == 1
        return SimpleNamespace(
            smiles=("CC",),
            input_ids=torch.ones((1, 3), dtype=torch.long),
            fidelity_indices=torch.tensor([0], dtype=torch.long),
        )

    def encode_smiles(self, smiles):
        return torch.ones((len(smiles), 3), dtype=torch.long)

    def on_policy_loss(self, input_ids, reward_scores, beta, fidelity_indices=None):
        return (
            self.policy.weight.square().sum()
            + self.fidelity_head.weight.square().sum()
            + self.log_z.square()
        )


def test_sampler_keeps_reward_precision_separate_from_model_dtype(
    make_sampler,
    fake_model,
):
    class _SmallRewardAcquisition:
        supports_singleton_scoring = True

        @staticmethod
        def score(candidates, cost_weighting=None):
            assert cost_weighting is None
            return [0.001] * len(candidates)

    sampler = make_sampler(model_dtype="bfloat16")

    prepared = sampler._prepare_batch(
        model=fake_model,
        smiles=("CC",),
        fidelity_indices=torch.tensor([0]),
        synthesizability=FakeSynthesizability(),
        molecule_chem=FakeChem,
        acquisition=_SmallRewardAcquisition(),
        cost_fn=None,
    )

    assert prepared.reward_scores.dtype is torch.float32
    assert prepared.reward_scores.tolist() == pytest.approx([0.001])


def test_training_uses_batch_size_for_generation(make_sampler):
    class _TrainingModel(_GradientTrainingModel):
        def __init__(self) -> None:
            super().__init__()
            self.generate_calls = []

        def generate(self, count, max_length, temperature):
            self.generate_calls.append(count)
            return SimpleNamespace(
                smiles=tuple("CC" for _ in range(count)),
                input_ids=torch.ones((count, 3), dtype=torch.long),
                fidelity_indices=torch.zeros(count, dtype=torch.long),
            )

    model = _TrainingModel()
    sampler = make_sampler(
        n_samples=1,
        batch_size=3,
        replay_batch_size=4,
        model_dtype="float32",
    )
    optimizer = torch.optim.SGD(
        [
            *model.policy.parameters(),
            *model.fidelity_head.parameters(),
            model.log_z,
        ],
        lr=0.1,
    )

    result = sampler._train_step(
        model=model,
        synthesizability=FakeSynthesizability(),
        positive_buffer=sampler_module.ReplayBuffer(pad_token_id=0, capacity=4),
        negative_buffer=None,
        molecule_chem=FakeChem,
        acquisition=FakeAcquisition(),
        cost_fn=None,
        optimizer=optimizer,
    )

    assert model.generate_calls == [3]
    assert result[0] == 3


def test_training_counts_requested_rows_when_generation_filters_rows(make_sampler):
    class _FilteredTrainingModel(_GradientTrainingModel):
        def generate(self, count, max_length, temperature):
            del max_length, temperature
            assert count == 3
            return SimpleNamespace(
                smiles=("CC",),
                input_ids=torch.ones((1, 3), dtype=torch.long),
                fidelity_indices=torch.tensor([0], dtype=torch.long),
            )

    model = _FilteredTrainingModel()
    sampler = make_sampler(
        n_samples=1,
        batch_size=3,
        replay_batch_size=4,
        num_warmup_steps=0,
        learning_rate=0.1,
        log_z_learning_rate=0.1,
        model_dtype="float32",
    )
    positive_buffer = sampler_module.ReplayBuffer(pad_token_id=0, capacity=4)
    optimizer = torch.optim.SGD(
        [
            *model.policy.parameters(),
            *model.fidelity_head.parameters(),
            model.log_z,
        ],
        lr=0.1,
    )

    result = sampler._train_step(
        model=model,
        synthesizability=FakeSynthesizability(),
        positive_buffer=positive_buffer,
        negative_buffer=None,
        molecule_chem=FakeChem,
        acquisition=FakeAcquisition(),
        cost_fn=None,
        optimizer=optimizer,
    )

    assert result[0] == 3
    assert sampler.round_metrics.generated_counts == [3]


def test_replay_uses_replay_batch_size(make_sampler):
    class _RecordingBuffer:
        def __init__(self, size: int) -> None:
            self.size = size
            self.calls: list[dict] = []

        def __len__(self) -> int:
            return self.size

        def sample(self, **kwargs):
            self.calls.append(kwargs)
            count = kwargs["count"]
            return SimpleNamespace(
                input_ids=torch.ones((count, 3), dtype=torch.long),
                reward_scores=torch.ones(count, dtype=kwargs["dtype"]),
                fidelity_indices=None,
            )

    sampler = make_sampler(n_samples=1, replay_batch_size=3)
    positive_buffer = _RecordingBuffer(size=5)
    negative_buffer = _RecordingBuffer(size=5)

    class _ReplayModel:
        def replay_loss(self, **kwargs):
            del kwargs
            return None

    model = _ReplayModel()

    sampler._update_replay_batch(
        model=model,
        positive_buffer=positive_buffer,
        negative_buffer=negative_buffer,
        optimizer=None,
    )

    assert positive_buffer.calls[0]["count"] == 3
    assert positive_buffer.calls[0]["dtype"] is torch.float32
    assert negative_buffer.calls[0]["count"] == 3
    assert negative_buffer.calls[0]["dtype"] is torch.float32


def test_final_generation_uses_independent_batch_size_and_strict_cap(
    make_sampler,
):
    class _FinalGenerationModel:
        def __init__(self) -> None:
            self.policy = nn.Linear(1, 1)
            self.prior = nn.Linear(1, 1)
            self.calls: list[int] = []
            self.next_smiles = 0

        def generate(self, count, max_length, temperature):
            del max_length, temperature
            self.calls.append(count)
            smiles = tuple(f"C{self.next_smiles + index}" for index in range(count))
            self.next_smiles += count
            return SimpleNamespace(
                smiles=smiles,
                fidelity_indices=torch.zeros(count, dtype=torch.long),
            )

    sampler = make_sampler(
        n_samples=4,
        batch_size=99,
        generation_batch_size=3,
        max_generation_attempts=5,
    )
    model = _FinalGenerationModel()

    candidates = sampler._generate_final_candidates(
        model=model,
        molecule_chem=FakeChem,
    )

    assert len(candidates) == 4
    assert model.calls == [3, 2]


def test_final_generation_stops_at_strict_attempt_cap(make_sampler):
    class _EmptyGenerationModel:
        def __init__(self) -> None:
            self.policy = nn.Linear(1, 1)
            self.prior = nn.Linear(1, 1)
            self.calls: list[int] = []

        def generate(self, count, max_length, temperature):
            del max_length, temperature
            self.calls.append(count)
            return SimpleNamespace(smiles=(), fidelity_indices=None)

    sampler = make_sampler(
        n_samples=1,
        batch_size=99,
        generation_batch_size=8,
        max_generation_attempts=3,
    )
    model = _EmptyGenerationModel()

    with pytest.raises(RuntimeError, match="after 3 attempts"):
        sampler._generate_final_candidates(model=model, molecule_chem=FakeChem)

    assert model.calls == [3]
    assert sampler.round_metrics.generation_invalid == 3


def test_sampler_returns_canonical_smiles_with_conditionally_sampled_fidelities(
    make_sampler,
    fake_model,
    fake_acquisition,
    patch_molecule_dependencies,
):
    sampler = make_sampler()
    sampler._new_round_model = lambda: fake_model
    sampler._train_round = lambda **kwargs: None

    candidates = sampler.sample(
        acquisition=fake_acquisition,
        cost_fn=lambda candidates: [
            float(candidate.fidelity) for candidate in candidates
        ],
    )

    assert len(candidates) == 2
    assert {candidate.x for candidate in candidates} == {"CC", "CO"}
    assert [candidate.fidelity for candidate in candidates] == [2, 1]
    assert fake_acquisition.seen_fidelities == []
    assert fake_model.generate_calls[0] == {
        "count": 2,
        "max_length": 8,
        "temperature": 1.0,
    }


def test_zero_training_steps_samples_without_acquisition_or_training(
    make_sampler,
    fake_model,
    patch_molecule_dependencies,
) -> None:
    """Zero steps generate directly from a fresh pretrained policy."""
    sampler = make_sampler(n_train_steps=0)
    sampler._new_round_model = lambda: fake_model
    sampler._train_round = Mock(side_effect=AssertionError("training must not run"))

    candidates = sampler.sample()

    assert len(candidates) == 2
    assert sampler._train_round.call_count == 0
    assert sampler.round_metrics.training_duration_s == 0.0


def test_sampler_compiles_policy_before_training(
    make_sampler,
    patch_molecule_dependencies,
) -> None:
    """Compilation must accelerate policy forwards in training and generation."""
    events: list[tuple[str, object]] = []

    class CompileAwareFakeModel(FakeModel):
        """Record compilation without invoking TorchInductor."""

        def compile_policy(
            self,
            *,
            mode: str,
            dynamic: bool | None,
            training_only: bool,
        ) -> None:
            """Record the requested compilation mode."""
            events.append(("compile", mode, dynamic, training_only))

    model = CompileAwareFakeModel()
    model.policy.train()
    sampler = make_sampler(
        compile_strategy="training_and_generation",
        torch_compile_mode="max-autotune",
        torch_compile_dynamic=None,
    )
    sampler._new_round_model = lambda: model
    sampler._train_round = lambda **kwargs: events.append(
        ("train", kwargs["model"].policy.training)
    )

    sampler.sample(acquisition=FakeAcquisition())

    assert events == [
        ("compile", "max-autotune", None, False),
        ("train", True),
    ]


def test_round_model_enables_attention_adapter_after_copy(
    make_sampler,
    monkeypatch,
) -> None:
    """The adapter must bind to the fresh round policy, not its CPU template."""
    sampler = make_sampler(attention_mask_adapter=True)
    template = SimpleNamespace(
        policy=nn.Linear(1, 1),
        prior=nn.Linear(1, 1),
        tokenizer=FakeTokenizer(),
        fidelity_head=None,
    )
    sampler._pretrained_model = template
    sampler._keep_pretrained_template_on_cpu = lambda: None
    enabled_policies = []

    def record_enable(model):
        enabled_policies.append(model.policy)
        return 1

    monkeypatch.setattr(
        sampler_module.S3GFNModel,
        "enable_attention_mask_adapter",
        record_enable,
    )
    round_model = sampler._new_round_model()

    assert enabled_policies == [round_model.policy]
    assert round_model.policy is not template.policy


def test_sampler_advances_round_state_across_consecutive_samples(
    make_sampler,
    patch_molecule_dependencies,
) -> None:
    models: list[FakeModel] = []
    sampler = make_sampler()

    def build_model() -> FakeModel:
        model = FakeModel()
        models.append(model)
        return model

    sampler._new_round_model = build_model
    sampler._train_round = lambda **kwargs: None

    sampler.sample(acquisition=FakeAcquisition())
    sampler.sample(acquisition=FakeAcquisition())

    assert len(models) == 2
    assert sampler._round_index == 2


def test_sampler_resets_round_metrics_at_sample_start(
    make_sampler,
    patch_molecule_dependencies,
) -> None:
    sampler = make_sampler()
    previous_metrics = sampler.round_metrics
    previous_metrics.record_training_step(
        generated_count=1,
        valid_count=1,
        synthesizable_count=1,
        online_loss=1.0,
        replay_loss=None,
        auxiliary_loss=None,
        log_z=0.0,
        raw_reward_scores=[1.0],
    )
    observed_metrics = []
    sampler._new_round_model = lambda: FakeModel()
    sampler._train_round = lambda **kwargs: observed_metrics.append(
        sampler.round_metrics
    )

    sampler.sample(acquisition=FakeAcquisition())

    assert observed_metrics == [sampler.round_metrics]
    assert observed_metrics[0] is not previous_metrics
    assert observed_metrics[0].generated_counts == []


def test_prepare_batch_scores_the_selected_fidelity(
    make_sampler,
    fake_model,
    fake_acquisition,
    patch_molecule_dependencies,
):
    sampler = make_sampler()

    prepared = sampler._prepare_batch(
        model=fake_model,
        smiles=("CC", "CO"),
        fidelity_indices=torch.tensor([1, 0]),
        synthesizability=FakeSynthesizability(),
        molecule_chem=FakeChem,
        acquisition=fake_acquisition,
        cost_fn=None,
    )

    assert prepared.reward_scores.tolist() == [2.0, 1.0]
    assert fake_acquisition.seen_fidelities == [2, 1]


def test_single_fidelity_preparation_omits_terminal_action(
    make_sampler,
    fake_model,
    patch_molecule_dependencies,
):
    sampler = make_sampler(n_samples=1, fidelities=[7], batch_size=1)

    prepared = sampler._prepare_batch(
        model=fake_model,
        smiles=("CC",),
        fidelity_indices=None,
        synthesizability=FakeSynthesizability(),
        molecule_chem=FakeChem,
        acquisition=FakeAcquisition(),
        cost_fn=None,
    )

    assert prepared.fidelity_indices is None
    assert prepared.reward_scores.tolist() == [7.0]


def test_sampler_rejects_batch_only_acquisitions(make_sampler):
    sampler = make_sampler(n_samples=1, fidelities=[1])

    with pytest.raises(ValueError, match="singleton scoring"):
        sampler.sample(acquisition=_FakeBatchAcquisition())


def test_canonicalization_rejects_disconnected_molecules():
    assert (
        sampler_module._canonicalize_to_smiles(
            "C.C",
            molecule_chem=FakeChem,
        )
        is None
    )


@pytest.mark.parametrize(
    ("smiles", "expected_reason"),
    (
        ("CCO", None),
        ("", "empty"),
        ("   ", "empty"),
        ("invalid", "parse_error"),
        ("C.C", "disconnected"),
    ),
)
def test_canonicalize_with_reason_labels_each_rejection_path(
    smiles: str,
    expected_reason: str | None,
) -> None:
    canonical, reason = sampler_module._canonicalize_with_reason(
        smiles,
        molecule_chem=FakeChem,
    )
    assert reason == expected_reason
    # Exactly one of the two is set, whichever path was taken.
    assert (canonical is None) is (reason is not None)


def test_reject_counts_separate_phases_and_reasons(make_sampler) -> None:
    sampler = make_sampler(n_samples=2)
    sampler._record_rejects(
        [("parse_error", "C1CC"), ("disconnected", "C.C")],
        phase="generation",
    )
    sampler._record_rejects([("parse_error", "c1cc")], phase="train")
    sampler.round_metrics.record_reject_count(
        "unterminated",
        3,
        phase="generation",
    )

    assert sampler.round_metrics.reject_counts == {
        "generation/parse_error": 1,
        "generation/disconnected": 1,
        "generation/unterminated": 3,
        "train/parse_error": 1,
    }
    # Unterminated generations carry no string, so only the three with one are
    # eligible for the reservoir.
    assert len(sampler.round_metrics.reject_examples) == 3


def test_reject_reservoir_is_bounded_but_counts_stay_exact(make_sampler) -> None:
    sampler = make_sampler(n_samples=2)
    sampler.reject_sample_size = 5
    sampler._record_rejects(
        [("parse_error", f"C{index}") for index in range(500)],
        phase="generation",
    )

    assert sampler.round_metrics.reject_counts["generation/parse_error"] == 500
    assert len(sampler.round_metrics.reject_examples) == 5
    assert all(
        phase == "generation" and reason == "parse_error"
        for phase, reason, _ in sampler.round_metrics.reject_examples
    )


def test_zero_reject_sample_size_keeps_counts_only(make_sampler) -> None:
    sampler = make_sampler(n_samples=2)
    sampler.reject_sample_size = 0
    sampler._record_rejects([("parse_error", "C1CC")], phase="generation")

    assert sampler.round_metrics.reject_counts == {"generation/parse_error": 1}
    assert sampler.round_metrics.reject_examples == []


def test_write_reject_log_writes_sampled_rows(make_sampler, tmp_path) -> None:
    sampler = make_sampler(n_samples=2)
    sampler._record_rejects(
        [("parse_error", "C1CC"), ("disconnected", "C.C")],
        phase="generation",
    )
    path = str(tmp_path / "rejects_round_{round}.csv")

    written = logging_module.write_reject_log(
        sampler.round_metrics,
        path,
        round_index=3,
    )

    assert written is not None
    written_path = Path(written)
    # `{round}` is substituted so successive rounds do not overwrite.
    assert written_path.name == "rejects_round_0003.csv"
    rows = list(csv.DictReader(written_path.open(encoding="utf-8")))
    assert [row["reason"] for row in rows] == ["parse_error", "disconnected"]
    assert [row["smiles"] for row in rows] == ["C1CC", "C.C"]
    assert [row["length"] for row in rows] == ["4", "3"]


def test_write_reject_log_skips_an_empty_reservoir(make_sampler, tmp_path) -> None:
    sampler = make_sampler(n_samples=2)
    path = tmp_path / "rejects.csv"

    assert (
        logging_module.write_reject_log(
            sampler.round_metrics,
            str(path),
            round_index=0,
        )
        is None
    )
    assert not path.exists()


def test_training_updates_policy_and_fidelity_parameters(
    make_sampler,
    make_replay_buffer,
    fake_acquisition,
) -> None:
    model = _GradientTrainingModel()
    policy_before = model.policy.weight.detach().clone()
    fidelity_before = model.fidelity_head.weight.detach().clone()
    sampler = make_sampler(
        n_samples=1,
        batch_size=1,
        replay_batch_size=2,
        num_warmup_steps=0,
        learning_rate=0.1,
        log_z_learning_rate=0.1,
    )
    positive_buffer = make_replay_buffer()

    sampler._train_round(
        model=model,
        synthesizability=FakeSynthesizability(),
        positive_buffer=positive_buffer,
        negative_buffer=None,
        molecule_chem=FakeChem,
        acquisition=FakeAcquisition(),
        cost_fn=None,
    )

    assert not torch.equal(model.policy.weight, policy_before)
    assert not torch.equal(model.fidelity_head.weight, fidelity_before)


def test_replay_updates_wait_for_a_full_replay_batch(
    make_sampler,
    make_replay_buffer,
) -> None:
    class _ReplayModel:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def replay_loss(self, **kwargs):
            self.calls.append(kwargs)
            return None

    sampler = make_sampler(n_samples=1, fidelities=[1], replay_batch_size=2)
    positive_buffer = make_replay_buffer()
    negative_buffer = make_replay_buffer()
    tokens = torch.tensor([[1, 2, 0], [1, 3, 0]])
    positive_buffer.add_batch(tokens[:1], ["positive-a"], [0.5])
    model = _ReplayModel()
    sampler._update_replay_batch(
        model=model,
        positive_buffer=positive_buffer,
        negative_buffer=negative_buffer,
        optimizer=None,
    )
    assert model.calls == []

    positive_buffer.add_batch(tokens[1:], ["positive-b"], [1.0])
    negative_buffer.add_batch(tokens[:1], ["negative-a"], [0.0])
    sampler._update_replay_batch(
        model=model,
        positive_buffer=positive_buffer,
        negative_buffer=negative_buffer,
        optimizer=None,
    )
    assert model.calls[-1]["negative_input_ids"] is None

    negative_buffer.add_batch(tokens[1:], ["negative-b"], [0.0])
    sampler._update_replay_batch(
        model=model,
        positive_buffer=positive_buffer,
        negative_buffer=negative_buffer,
        optimizer=None,
    )
    assert model.calls[-1]["negative_input_ids"] is not None


def test_generated_batch_routes_synthesizable_and_negative_replays(
    make_sampler,
    make_replay_buffer,
) -> None:
    class _OnPolicyModel:
        def on_policy_loss(self, *args, **kwargs):
            return None

    sampler = make_sampler(n_samples=1, fidelities=[1], aux_coefficient=1.0e-4)
    positive_buffer = make_replay_buffer()
    negative_buffer = make_replay_buffer()
    prepared = sampler_module._PreparedMoleculeBatch(
        smiles=("positive", "negative"),
        input_ids=torch.tensor([[1, 2, 0], [1, 3, 0]]),
        reward_scores=torch.tensor([1.0, 0.0]),
        acq_scores=(1.0, 0.0),
        synthesizable=(True, False),
    )

    sampler._update_generated_batch(
        model=_OnPolicyModel(),
        prepared=prepared,
        positive_buffer=positive_buffer,
        negative_buffer=negative_buffer,
        optimizer=None,
    )

    assert [entry.smiles for entry in positive_buffer.entries] == ["positive"]
    assert [entry.smiles for entry in negative_buffer.entries] == ["negative"]


def test_optimizer_clips_fidelity_head_gradients(monkeypatch, make_sampler) -> None:
    policy = nn.Linear(1, 1)
    fidelity_head = nn.Linear(1, 1)
    model = SimpleNamespace(
        policy=policy,
        fidelity_head=fidelity_head,
        log_z=nn.Parameter(torch.zeros(())),
    )
    sampler = make_sampler(n_samples=1)
    optimizer = torch.optim.SGD(
        [
            *policy.parameters(),
            *fidelity_head.parameters(),
            model.log_z,
        ],
        lr=0.1,
    )
    captured_parameters = []

    def capture_clip(parameters, max_norm):
        captured_parameters.extend(parameters)
        return torch.tensor(0.0)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", capture_clip)

    sampler._optimize(
        optimizer,
        sum(parameter.sum() for parameter in model.policy.parameters())
        + sum(parameter.sum() for parameter in model.fidelity_head.parameters())
        + model.log_z,
        model,
    )

    assert any(
        parameter is list(fidelity_head.parameters())[0]
        for parameter in captured_parameters
    )


def test_round_metrics_aggregate_scalars_and_figures(
    make_sampler,
    make_replay_buffer,
) -> None:
    sampler = make_sampler(n_train_steps=2)
    metrics = sampler.round_metrics
    metrics.record_training_step(
        generated_count=4,
        valid_count=3,
        synthesizable_count=2,
        online_loss=2.0,
        replay_loss=None,
        auxiliary_loss=None,
        log_z=0.1,
        raw_reward_scores=[2.0, 4.0, 6.0],
    )
    metrics.record_training_step(
        generated_count=2,
        valid_count=2,
        synthesizable_count=1,
        online_loss=4.0,
        replay_loss=3.0,
        auxiliary_loss=0.5,
        log_z=0.2,
        raw_reward_scores=[8.0, 10.0],
    )
    metrics.record_generation_batch(
        attempts=5,
        invalid_count=1,
        duplicate_count=1,
    )
    metrics.record_final_candidates(
        [
            Candidate(x="CC", fidelity=1),
            Candidate(x="CO", fidelity=2),
        ]
    )
    metrics.training_duration_s = 1.25
    metrics.generation_duration_s = 0.75
    metrics.positive_buffer_size = 3

    logged, figure_calls = sampler.drain_round_diagnostics(
        include_figures=True,
        max_points=1000,
    )

    assert logged["sampler/s3gfn/train/generated_total"] == 6
    assert logged["sampler/s3gfn/train/valid_total"] == 5
    assert logged["sampler/s3gfn/train/synthesizable_total"] == 3
    assert logged["sampler/s3gfn/train/validity_rate"] == pytest.approx(5 / 6)
    assert logged["sampler/s3gfn/train/synthesizable_rate"] == pytest.approx(3 / 5)
    assert logged["sampler/s3gfn/train/online_updates"] == 2
    assert logged["sampler/s3gfn/train/replay_updates"] == 1
    assert logged["sampler/s3gfn/train/online_rtb_loss_mean"] == pytest.approx(3.0)
    assert logged["sampler/s3gfn/train/online_rtb_loss_final"] == pytest.approx(4.0)
    assert logged["sampler/s3gfn/train/replay_loss_mean"] == pytest.approx(3.0)
    assert logged["sampler/s3gfn/train/contrastive_loss_mean"] == pytest.approx(0.5)
    assert logged["sampler/s3gfn/train/contrastive_loss_final"] == pytest.approx(0.5)
    assert logged["sampler/s3gfn/train/log_z_final"] == pytest.approx(0.2)
    assert logged["sampler/s3gfn/reward/raw_mean"] == pytest.approx(6.0)
    assert logged["sampler/s3gfn/reward/raw_max"] == pytest.approx(10.0)
    assert logged["sampler/s3gfn/generation/yield"] == pytest.approx(0.4)
    assert logged["sampler/s3gfn/generation/invalid_rate"] == pytest.approx(0.2)
    assert logged["sampler/s3gfn/generation/duplicate_rate"] == pytest.approx(0.2)
    assert all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in logged.values()
    )
    assert set(figure_calls) == {
        "sampler/s3gfn/training_losses",
        "sampler/s3gfn/log_z",
        "sampler/s3gfn/reward/trajectory",
        "sampler/s3gfn/reward/concentration",
        "sampler/s3gfn/reward/effective_support",
        "sampler/s3gfn/train/batch_health",
        # No acq/trajectory: these steps recorded no acq_scores, and that figure
        # plots acq itself rather than the post-transform score.
    }
    assert all(isinstance(figure, Figure) for figure in figure_calls.values())
    training_axis = figure_calls["sampler/s3gfn/training_losses"].axes[0]
    assert training_axis.get_ylabel() == "S3-GFN training loss"
    for figure in figure_calls.values():
        plt.close(figure)


def test_sampler_drains_round_metrics_without_touching_runtime_logger(
    make_sampler,
    fake_model,
    patch_molecule_dependencies,
):
    runtime_logger = Mock()
    sampler = make_sampler()
    sampler.bind_runtime_context(RuntimeContext(logger=runtime_logger))
    sampler._new_round_model = lambda: fake_model
    sampler._train_round = lambda **kwargs: None

    sampler.sample(acquisition=FakeAcquisition())

    metrics, figures = sampler.drain_round_diagnostics(
        include_figures=False,
        max_points=1000,
    )

    assert metrics["sampler/s3gfn/generation/yield"] == pytest.approx(1.0)
    assert figures == {}
    runtime_logger.log_metric.assert_not_called()
    runtime_logger.log_step.assert_not_called()
    runtime_logger.end.assert_not_called()


def test_round_metrics_drain_without_runtime_logger(
    make_sampler,
    make_replay_buffer,
) -> None:
    """Metric emission must be skipped, not crash, when no logger is bound."""
    sampler = make_sampler(n_samples=1, fidelities=[1])
    assert sampler.logger is None
    metrics = sampler.round_metrics
    metrics.record_training_step(
        generated_count=1,
        valid_count=1,
        synthesizable_count=1,
        online_loss=1.0,
        replay_loss=None,
        auxiliary_loss=None,
        log_z=0.0,
        raw_reward_scores=[1.0],
    )

    logged, figures = sampler.drain_round_diagnostics(
        include_figures=False,
        max_points=1000,
    )

    assert logged["sampler/s3gfn/train/generated_total"] == 1
    assert figures == {}
    assert sampler.drain_round_diagnostics(
        include_figures=False,
        max_points=1000,
    ) == ({}, {})


def test_prepare_batch_retains_acquisition_reward_scores(
    make_sampler,
    fake_model,
    patch_molecule_dependencies,
):
    sampler = make_sampler()

    prepared = sampler._prepare_batch(
        model=fake_model,
        smiles=("CC", "CO"),
        fidelity_indices=torch.tensor([1, 0]),
        synthesizability=FakeSynthesizability(),
        molecule_chem=FakeChem,
        acquisition=FakeAcquisition(),
        cost_fn=None,
    )

    assert prepared.reward_scores.tolist() == [2.0, 1.0]


def test_optimizer_does_not_decay_log_z(make_sampler) -> None:
    """log_z is a normalizer estimate; decay would pull it back toward zero."""
    model = SimpleNamespace(
        policy=nn.Linear(1, 1),
        fidelity_head=None,
        log_z=nn.Parameter(torch.tensor(-8.0)),
    )
    sampler = make_sampler(log_z_learning_rate=0.1)

    policy_group, log_z_group = sampler._build_training_optimizer(model).param_groups

    assert log_z_group["params"][0] is model.log_z
    assert log_z_group["weight_decay"] == 0.0
    assert log_z_group["lr"] == 0.1
    # The policy keeps AdamW's default decay; only the scalar is exempt.
    assert policy_group["weight_decay"] > 0.0


def test_round_model_starts_log_z_at_the_configured_value(make_sampler) -> None:
    """Every round's fresh model opens at initial_log_z, not at zero."""
    sampler = make_sampler(initial_log_z=-8.0)
    sampler._pretrained_model = SimpleNamespace(
        policy=nn.Linear(1, 1),
        prior=nn.Linear(1, 1),
        tokenizer=FakeTokenizer(),
        fidelity_head=None,
    )
    sampler._keep_pretrained_template_on_cpu = lambda: None

    round_model = sampler._new_round_model()

    assert float(round_model.log_z.detach()) == pytest.approx(-8.0)
    assert round_model.log_z.requires_grad


@pytest.mark.parametrize("setting", ["initial_log_z", "log_score_floor"])
def test_sampler_rejects_non_finite_log_z_and_floor(make_sampler, setting) -> None:
    with pytest.raises(ValueError, match=setting):
        make_sampler(**{setting: float("nan")})


def test_optimizer_records_the_gradient_norm_before_clipping(make_sampler) -> None:
    """The recorded norm is the unclipped one, so it can exceed the threshold."""
    model = SimpleNamespace(
        policy=nn.Linear(1, 1, bias=False),
        fidelity_head=None,
        log_z=nn.Parameter(torch.zeros(())),
    )
    sampler = make_sampler(n_samples=1, gradient_clip_norm=1.0)
    optimizer = torch.optim.SGD([*model.policy.parameters(), model.log_z], lr=0.1)

    # d/dw = 30 and d/dlog_z = 40, a global norm of 50 against a clip of 1.
    sampler._optimize(
        optimizer, 30.0 * model.policy.weight.sum() + 40.0 * model.log_z, model
    )

    assert sampler._last_gradient_norm == pytest.approx(50.0)


def test_training_step_records_gradient_norms_for_the_updates_that_ran(
    make_sampler,
    make_replay_buffer,
) -> None:
    """One on-policy update and no replay: one norm, on the step it happened."""
    sampler = make_sampler(
        n_samples=1,
        batch_size=1,
        replay_batch_size=2,
        num_warmup_steps=0,
    )

    sampler._train_round(
        model=_GradientTrainingModel(),
        synthesizability=FakeSynthesizability(),
        positive_buffer=make_replay_buffer(),
        negative_buffer=None,
        molecule_chem=FakeChem,
        acquisition=FakeAcquisition(),
        cost_fn=None,
    )

    metrics = sampler.round_metrics
    assert metrics.online_gradient_norm_steps == [1]
    assert metrics.online_gradient_norms[0] > 0.0
    assert metrics.replay_gradient_norms == []
    assert metrics.gradient_clip_norm == sampler.gradient_clip_norm
    # This stand-in model exposes no residual diagnostics, and none are invented.
    assert metrics.residual_steps == []


def test_write_pool_log_writes_smiles_and_fidelity(tmp_path) -> None:
    candidates = [
        Candidate(x="CCO", fidelity=1),
        Candidate(x="c1ccccc1", fidelity=None),
    ]
    path = str(tmp_path / "pool_round_{round}.csv")

    written = logging_module.write_pool_log(candidates, path, round_index=2)

    assert written is not None
    written_path = Path(written)
    assert written_path.name == "pool_round_0002.csv"
    rows = list(csv.DictReader(written_path.open(encoding="utf-8")))
    # The column is SMILES so the file feeds the evaluation pipeline directly.
    assert [row["SMILES"] for row in rows] == ["CCO", "c1ccccc1"]
    assert [row["fidelity"] for row in rows] == ["1", ""]


def test_write_pool_log_skips_an_empty_pool(tmp_path) -> None:
    path = tmp_path / "pool.csv"

    assert logging_module.write_pool_log([], str(path), round_index=0) is None
    assert not path.exists()


def test_pool_log_is_written_even_though_the_round_has_not_completed(
    make_sampler, tmp_path
) -> None:
    """The pool must survive a round that dies later, in the oracle."""
    sampler = make_sampler(n_samples=2)
    sampler.pool_log_path = str(tmp_path / "pool_round_{round}.csv")

    sampler._write_pool_log(1, [Candidate(x="CCO", fidelity=1)])

    assert (tmp_path / "pool_round_0001.csv").exists()


def test_pool_log_is_disabled_by_default(make_sampler, tmp_path) -> None:
    sampler = make_sampler(n_samples=2)

    assert sampler.pool_log_path is None
    # No path configured: nothing is written and nothing raises.
    sampler._write_pool_log(1, [Candidate(x="CCO", fidelity=1)])
    assert list(tmp_path.iterdir()) == []
