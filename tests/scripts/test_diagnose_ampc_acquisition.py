"""Focused tests for the standalone AmpC acquisition diagnostic."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from activelearning.utils.types import Observation
from scripts import diagnose_ampc_acquisition as diagnostic


class _FakeSynthesizability:
    """Small SA-score double with configurable rejected molecules."""

    def __init__(self, scores: dict[str, float] | None = None) -> None:
        self.threshold = 4.0
        self.scores = scores or {}

    def score(self, smiles: str) -> float:
        """Return a finite configured score."""
        return self.scores.get(smiles, 1.0)


class _FakePosterior:
    """Posterior double with one mean and variance per encoded row."""

    def posterior(self, encoded: torch.Tensor) -> SimpleNamespace:
        """Return deterministic posterior tensors."""
        count = encoded.shape[0]
        return SimpleNamespace(
            mean=torch.arange(count, dtype=encoded.dtype).reshape(-1, 1),
            variance=torch.ones((count, 1), dtype=encoded.dtype),
        )


class _FakeSurrogate:
    """Fixed-shape encoder and posterior model for evaluator tests."""

    def encode_candidates(self, candidates: list[object]) -> torch.Tensor:
        """Encode candidates as a deterministic two-dimensional tensor."""
        return torch.arange(
            len(candidates) * 2,
            dtype=torch.float64,
        ).reshape(len(candidates), 2)

    def get_model(self) -> _FakePosterior:
        """Return the fake posterior model."""
        return _FakePosterior()


class _FakeRawAcquisition:
    """Raw acquisition double with a negative and positive value."""

    def __call__(self, encoded: torch.Tensor) -> torch.Tensor:
        """Return values aligned to the encoded batch."""
        values = torch.tensor(
            [-1.0, 0.25, 0.0, 0.5],
            dtype=encoded.dtype,
            device=encoded.device,
        )
        return values[: encoded.shape[0]]


def _evaluator(
    *,
    method: str = "test",
    budget: int = 4,
    scores: dict[str, float] | None = None,
) -> diagnostic.ProductionEvaluator:
    """Build a production evaluator backed by small deterministic doubles."""
    return diagnostic.ProductionEvaluator(
        method=method,
        surrogate=_FakeSurrogate(),
        acquisition=SimpleNamespace(_botorch_acqf=_FakeRawAcquisition()),
        fidelity=1,
        sa_threshold=4.0,
        evaluation_budget=budget,
        initial_smiles={"CC"},
        synthesizability=_FakeSynthesizability(scores),
    )


def test_parse_args_rejects_nonpositive_evaluation_budget() -> None:
    """The CLI requires at least one real acquisition evaluation."""
    with pytest.raises(SystemExit):
        diagnostic.parse_args(["--evaluation-budget", "0"])


def test_parse_args_merges_seed_and_chunk_overrides() -> None:
    """Seed and score chunk CLI values become OmegaConf overrides."""
    args = diagnostic.parse_args(
        [
            "--seed",
            "7",
            "--score-chunk-size",
            "11",
            "--override",
            "runtime.device=cpu",
        ]
    )

    assert diagnostic._resolved_overrides(args) == [
        "runtime.device=cpu",
        "runtime.seed=7",
        "acquisition.score_chunk_size=11",
    ]


def test_constant_acquisition_returns_one_per_candidate() -> None:
    """Uniform S3-GFN reward has one constant score per candidate."""
    acquisition = diagnostic.ConstantAcquisition()
    candidates = [
        diagnostic.Candidate("CC", fidelity=1),
        diagnostic.Candidate("CO", fidelity=1),
    ]

    assert acquisition.score(candidates) == [1.0, 1.0]


def test_uniform_s3gfn_uses_constant_training_acquisition() -> None:
    """S3-GFN receives the constant adapter and not the production acquisition."""

    class _FakeSampler:
        """Sampler double that records its training acquisition."""

        def __init__(self) -> None:
            self.training_acquisition = None

        def sample(self, *, acquisition, observations, cost_fn):
            """Record arguments and return one valid candidate."""
            self.training_acquisition = acquisition
            assert observations == []
            assert cost_fn is None
            return [diagnostic.Candidate("CC", fidelity=1)]

    class _FakeSamplerConfig:
        """Pydantic-like sampler-config double."""

        def __init__(self, sampler: _FakeSampler) -> None:
            self.sampler = sampler
            self.updated: dict[str, object] = {}

        def model_copy(self, *, update):
            """Record the copied sampler settings."""
            self.updated = dict(update)
            return self

        def build(self):
            """Return the fake sampler."""
            return self.sampler

    sampler = _FakeSampler()
    sampler_config = _FakeSamplerConfig(sampler)
    state = SimpleNamespace(
        config=SimpleNamespace(sampler=sampler_config),
        runtime_context=SimpleNamespace(),
        observations=[],
    )

    class _FakeEvaluator:
        """Evaluator double for the post-generation rescore boundary."""

        def __init__(self) -> None:
            self.evaluation_budget = 2
            self.counters = diagnostic.SearchCounters(evaluated=1)
            self.records: list[diagnostic.EvaluatedMolecule] = []

        def evaluate(self, proposals):
            """Record final proposals without scoring them."""
            self.proposals = proposals
            return []

    evaluator = _FakeEvaluator()
    result = diagnostic.run_uniform_s3gfn(
        evaluator=evaluator,
        state=state,
        seed=7,
    )

    assert sampler_config.updated == {"n_samples": 2, "seed": 7}
    assert sampler.training_acquisition.score(
        [diagnostic.Candidate("CC", fidelity=1)]
    ) == [1.0]
    assert evaluator.proposals[0][0] == "CC"
    assert result.parameters["real_acquisition_used_during_training"] is False


def test_evaluator_filters_and_preserves_raw_negative_values() -> None:
    """Invalid, disconnected, duplicate, and high-SA proposals are counted."""
    evaluator = _evaluator(
        budget=3,
        scores={"N": 5.0},
    )

    records = evaluator.evaluate(
        [
            ("CC", {"source": "initial"}),
            ("C.C", {}),
            ("CC", {}),
            ("N", {}),
            ("CO", {}),
            ("CN", {}),
        ]
    )

    assert [record.smiles for record in records] == ["CC", "CO", "CN"]
    assert records[0].raw_acquisition == -1.0
    assert records[0].acquisition == 0.0
    assert records[0].is_initial_molecule is True
    assert evaluator.counters.proposed == 6
    assert evaluator.counters.disconnected == 1
    assert evaluator.counters.duplicate == 1
    assert evaluator.counters.not_synthesizable == 1
    assert evaluator.counters.evaluated == 3


def test_evaluator_does_not_exceed_budget() -> None:
    """A large proposal batch is truncated before real scoring."""
    evaluator = _evaluator(budget=2)

    records = evaluator.evaluate([(smiles, {}) for smiles in ("CC", "CO", "CN", "CCC")])

    assert len(records) == 2
    assert evaluator.counters.evaluated == 2
    assert evaluator.remaining == 0


def test_selfies_sa_is_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    """The same seed produces the same accepted molecular trajectory."""
    observations = [Observation("CC", 1.0), Observation("CO", 1.0)]

    monkeypatch.setattr(
        diagnostic,
        "_selfies_alphabet",
        lambda _: ["[C]", "[O]"],
    )

    def run() -> list[tuple[str, float]]:
        """Run the deterministic small annealing instance."""
        result = diagnostic.run_selfies_sa(
            evaluator=_evaluator(method="selfies-sa", budget=5),
            observations=observations,
            seed=13,
            chains=2,
            cooling_ratio=0.1,
        )
        return [(record.smiles, record.acquisition) for record in result.records]

    assert run() == run()


def test_rdkit_ga_is_deterministic_with_injected_operators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Search control remains deterministic when graph operators are injected."""
    monkeypatch.setattr(
        diagnostic,
        "_crossover_smiles",
        lambda left, right, rng: "CCC",
    )
    monkeypatch.setattr(
        diagnostic,
        "_mutate_smiles",
        lambda smiles, rng: "CCO" if smiles == "CCC" else "CCC",
    )
    observations = [Observation("CC", 1.0), Observation("CO", 1.0)]

    def run() -> list[str]:
        """Run one deterministic genetic-search instance."""
        result = diagnostic.run_rdkit_ga(
            evaluator=_evaluator(method="rdkit-ga", budget=5),
            observations=observations,
            seed=19,
            population_size=2,
            elite_fraction=0.5,
            mutation_rate=1.0,
        )
        return [record.smiles for record in result.records]

    assert run() == run()


def test_summary_distinguishes_negative_clamping() -> None:
    """Summary metadata reports negative raw values separately from zeros."""
    evaluator = _evaluator(budget=2)
    evaluator.acquisition._botorch_acqf = lambda encoded: torch.tensor(
        [-1.0, 0.0],
        dtype=encoded.dtype,
    )
    records = evaluator.evaluate([("CC", {}), ("CO", {})])
    result = diagnostic.SearchResult(
        method="test",
        records=records,
        counters=evaluator.counters,
        parameters={},
        duration_s=0.1,
        completed_budget=True,
    )

    summary = diagnostic.summarize_method(
        result,
        evaluation_budget=2,
        top_k=2,
        zero_tolerance=0.0,
    )

    assert summary["score_diagnosis"] == "all_nonpositive_clamped"
    assert summary["raw_negative_count"] == 1
    assert summary["clamped_positive_count"] == 0


def test_summary_reports_earliest_positive_evaluation() -> None:
    """First-positive metadata uses acquisition order, not ranking order."""
    records = [
        diagnostic.EvaluatedMolecule(
            smiles="CCO",
            method="test",
            evaluation_index=2,
            raw_acquisition=0.5,
            acquisition=0.5,
            posterior_mean=0.0,
            posterior_std=1.0,
            sa_score=1.0,
            is_initial_molecule=False,
            provenance={},
        ),
        diagnostic.EvaluatedMolecule(
            smiles="CC",
            method="test",
            evaluation_index=1,
            raw_acquisition=0.1,
            acquisition=0.1,
            posterior_mean=0.0,
            posterior_std=1.0,
            sa_score=1.0,
            is_initial_molecule=False,
            provenance={},
        ),
    ]
    result = diagnostic.SearchResult(
        method="test",
        records=records,
        counters=diagnostic.SearchCounters(evaluated=2),
        parameters={},
        duration_s=0.1,
        completed_budget=True,
    )

    summary = diagnostic.summarize_method(
        result,
        evaluation_budget=2,
        top_k=2,
        zero_tolerance=0.0,
    )

    assert summary["first_positive_evaluation_index"] == 1


def test_ranked_csv_and_plots_are_written(tmp_path: Path) -> None:
    """Artifact writers create the documented CSV and three PNG outputs."""
    evaluator = _evaluator(budget=2)
    records = evaluator.evaluate([("CC", {}), ("CO", {})])

    csv_path = tmp_path / "ranked.csv"
    diagnostic.write_ranked_csv(csv_path, records)
    diagnostic.write_plots(tmp_path, "test", records)

    assert csv_path.is_file()
    assert (tmp_path / "test_score_distribution.png").is_file()
    assert (tmp_path / "test_search_trajectory.png").is_file()
    assert (tmp_path / "test_posterior_relationship.png").is_file()
