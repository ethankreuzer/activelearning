"""Fake-backed tests for the MiniMol SMILES DKL encoder."""

from __future__ import annotations

import json
import multiprocessing
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import Tensor, nn

import activelearning.applications.molecules.minimol_encoder as minimol_module
from activelearning.acquisition.botorch.botorch_analytic import (
    UpperConfidenceBound,
)
from activelearning.active_learning import active_learning
from activelearning.budget.budget import Budget
from activelearning.dataset.list_dataset import ListDataset
from activelearning.oracle.multi_fidelity_oracle import MultiFidelityOracle
from activelearning.sampler.pool_score_sampler import PoolScoreSampler
from activelearning.selector.score_selector import TopKAcquisitionSelector
from activelearning.surrogate.dkl import ExactDKLSurrogate, VariationalDKLSurrogate
from activelearning.surrogate.dkl.config import DKLTrainingConfig
from activelearning.utils.types import Candidate, Observation


class _FakeMiniMol:
    """Deterministic MiniMol replacement returning 512-value fingerprints."""

    instances: list["_FakeMiniMol"] = []

    def __init__(
        self,
        *,
        batch_size: int,
        checkpoint_path: object = None,
    ) -> None:
        self.batch_size = batch_size
        self.checkpoint_path = checkpoint_path
        self.calls: list[list[str]] = []
        self.grad_enabled: list[bool] = []
        self.__class__.instances.append(self)

    def __call__(self, smiles: list[str]) -> list[Tensor]:
        self.calls.append(list(smiles))
        self.grad_enabled.append(torch.is_grad_enabled())
        return [
            torch.arange(512, dtype=torch.float64) + sum(map(ord, value))
            for value in smiles
        ]


@pytest.fixture
def fake_minimol(monkeypatch: pytest.MonkeyPatch) -> list[_FakeMiniMol]:
    """Replace the lazy MiniMol loader and return constructed fake models."""
    _FakeMiniMol.instances = []
    monkeypatch.setattr(minimol_module, "_load_minimol", lambda: _FakeMiniMol)
    return _FakeMiniMol.instances


def _encode_feature_cache_in_process(
    cache_path: str,
    ready_queue: Any,
    start_event: Any,
    call_log_path: str,
    result_queue: Any,
) -> None:
    """Create or reuse one feature cache from a forked worker."""

    class _ProcessMiniMol:
        def __call__(self, smiles: list[str]) -> list[Tensor]:
            with open(call_log_path, "a", encoding="utf-8") as stream:
                stream.write("call\n")
            return [
                torch.arange(512, dtype=torch.float64) + sum(map(ord, value))
                for value in smiles
            ]

    minimol_module._bundled_minimol_checkpoint = lambda: None

    encoder = minimol_module.MiniMolSmilesFixedEncoder(
        feature_cache_path=cache_path,
    )
    encoder._build_minimol = lambda checkpoint_path: _ProcessMiniMol()
    ready_queue.put(True)
    start_event.wait(timeout=30)
    try:
        features = encoder.encode(
            ["CC", "CO"],
            device=torch.device("cpu"),
        )
        result_queue.put(tuple(features.shape))
    except BaseException as error:
        result_queue.put(repr(error))
        raise


def test_minimol_encoder_validates_constructor_arguments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Invalid extraction and projection sizes should fail early."""
    monkeypatch.setattr(minimol_module, "_load_minimol", lambda: _FakeMiniMol)

    with pytest.raises(ValueError, match="batch_size"):
        minimol_module.MiniMolSmilesEncoder(batch_size=0)
    with pytest.raises(ValueError, match="latent_dim"):
        minimol_module.MiniMolSmilesEncoder(latent_dim=0)
    with pytest.raises(ValueError, match="cache_size"):
        minimol_module.MiniMolSmilesEncoder(cache_size=-1)
    with pytest.raises(FileNotFoundError, match="checkpoint_path"):
        minimol_module.MiniMolSmilesEncoder(
            checkpoint_path=tmp_path / "missing.pth",
        )


def test_minimol_encoder_passes_checkpoint_path_to_loader(
    fake_minimol: list[_FakeMiniMol],
    tmp_path: Path,
) -> None:
    """The configured checkpoint path reaches the MiniMol loader."""
    checkpoint_path = tmp_path / "minimol.pth"
    checkpoint_path.touch()

    encoder = minimol_module.MiniMolSmilesEncoder(
        checkpoint_path=checkpoint_path,
    )

    assert encoder.checkpoint_path == checkpoint_path
    assert fake_minimol == []
    encoder.prepare_inputs(["CC"], device=torch.device("cpu"))
    assert fake_minimol[0].checkpoint_path == checkpoint_path


@pytest.mark.parametrize("wrapped", [False, True])
def test_minimol_checkpoint_loads_predictor_state_dict(
    wrapped: bool,
    tmp_path: Path,
) -> None:
    """Raw and wrapped predictor state dicts replace MiniMol weights."""
    source = nn.Linear(2, 1)
    with torch.no_grad():
        source.weight.fill_(3.0)
        source.bias.fill_(-2.0)
    checkpoint: object = source.state_dict()
    if wrapped:
        checkpoint = {"state_dict": checkpoint}
    checkpoint_path = tmp_path / "minimol.pth"
    torch.save(checkpoint, checkpoint_path)

    target = SimpleNamespace(
        predictor=SimpleNamespace(predictor=nn.Linear(2, 1)),
    )
    minimol_module._load_minimol_checkpoint(target, checkpoint_path)

    assert torch.equal(target.predictor.predictor.weight, source.weight)
    assert torch.equal(target.predictor.predictor.bias, source.bias)


def test_prepare_inputs_extracts_ordered_fingerprints(
    fake_minimol: list[_FakeMiniMol],
) -> None:
    """Raw SMILES become ordered float32 fingerprint rows."""
    encoder = minimol_module.MiniMolSmilesEncoder(batch_size=2, latent_dim=4)

    prepared = encoder.prepare_inputs(["CC", "CO"], device=torch.device("cpu"))

    assert prepared.shape == (2, 512)
    assert prepared.dtype == torch.float32
    assert prepared[:, 0].tolist() == pytest.approx(
        [float(sum(map(ord, value))) for value in ["CC", "CO"]]
    )
    assert fake_minimol[0].batch_size == 2
    assert fake_minimol[0].calls == [["CC", "CO"]]
    assert fake_minimol[0].grad_enabled == [False]


def test_fixed_encoder_returns_raw_fingerprints(
    fake_minimol: list[_FakeMiniMol],
) -> None:
    """The fixed encoder returns pooled512 without a projection module."""
    encoder = minimol_module.MiniMolSmilesFixedEncoder(
        batch_size=2,
        cache_size=8,
    )

    features = encoder.encode(["CC", "CO"], device=torch.device("cpu"))

    assert encoder.feature_dim == 512
    assert features.shape == (2, 512)
    assert not hasattr(encoder, "projection")
    assert fake_minimol[0].calls == [["CC", "CO"]]


def test_cache_identity_does_not_import_minimol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inspecting stock cache identity must not load the MiniMol package."""
    monkeypatch.delitem(sys.modules, "minimol", raising=False)
    encoder = minimol_module.MiniMolSmilesFixedEncoder()

    identity = encoder._cache_encoder_identity()

    assert identity["backend"] == "minimol.Minimol"
    assert "minimol" not in sys.modules


def test_prepare_inputs_handles_empty_batches_without_model_call(
    fake_minimol: list[_FakeMiniMol],
) -> None:
    """An empty candidate batch should preserve its shape without extraction."""
    encoder = minimol_module.MiniMolSmilesEncoder()

    prepared = encoder.prepare_inputs([], device=torch.device("cpu"))

    assert encoder.latent_dim == 32
    assert prepared.shape == (0, 512)
    assert prepared.dtype == torch.float32
    assert fake_minimol == []


def test_configured_feature_cache_ignores_empty_batches(
    fake_minimol: list[_FakeMiniMol],
    tmp_path: Path,
) -> None:
    """An empty request does not create cache coordination or data files."""
    feature_path = tmp_path / "minimol-features.npy"
    encoder = minimol_module.MiniMolSmilesFixedEncoder(
        feature_cache_path=feature_path,
    )

    features = encoder.encode([], device=torch.device("cpu"))

    assert features.shape == (0, 512)
    assert not feature_path.exists()
    assert not Path(f"{feature_path}.json").exists()
    assert not Path(f"{feature_path}.lock").exists()
    assert fake_minimol == []


def test_prepare_inputs_rejects_non_string_values(
    fake_minimol: list[_FakeMiniMol],
) -> None:
    """MiniMol's raw SMILES boundary should reject non-string values."""
    encoder = minimol_module.MiniMolSmilesEncoder()

    with pytest.raises(ValueError, match="string inputs"):
        encoder.prepare_inputs(["CC", 1], device=torch.device("cpu"))

    assert fake_minimol == []


def test_fingerprint_cache_deduplicates_requests_and_evicts_lru_entries(
    fake_minimol: list[_FakeMiniMol],
) -> None:
    """The bounded cache should deduplicate inputs and evict least-recent rows."""
    encoder = minimol_module.MiniMolSmilesEncoder(cache_size=2)

    encoder.prepare_inputs(["CC", "CC", "CO"], device=torch.device("cpu"))
    encoder.prepare_inputs(["CC", "CN"], device=torch.device("cpu"))
    encoder.prepare_inputs(["CO"], device=torch.device("cpu"))

    assert fake_minimol[0].calls == [["CC", "CO"], ["CN"], ["CO"]]


def test_persistent_feature_cache_is_reused_without_a_minimol_call(
    fake_minimol: list[_FakeMiniMol],
    tmp_path: Path,
) -> None:
    """A fresh encoder can reuse the verified feature matrix directly."""
    feature_path = tmp_path / "minimol-features.npy"
    values = ["CC", "CC", "CO"]
    first_encoder = minimol_module.MiniMolSmilesFixedEncoder(
        batch_size=2,
        feature_cache_path=feature_path,
    )
    expected = first_encoder.encode(
        values,
        device=torch.device("cpu"),
    )

    second_encoder = minimol_module.MiniMolSmilesFixedEncoder(
        batch_size=2,
        feature_cache_path=feature_path,
    )
    actual = second_encoder.encode(
        values,
        device=torch.device("cpu"),
    )

    assert feature_path.is_file()
    assert Path(f"{feature_path}.json").is_file()
    assert torch.equal(actual, expected)
    manifest = json.loads(Path(f"{feature_path}.json").read_text(encoding="utf-8"))
    assert manifest["format_version"] == 2
    assert "sha256" not in manifest["feature_file"]
    assert fake_minimol[0].calls == [["CC", "CO"]]
    assert len(fake_minimol) == 1


def test_candidate_encoding_does_not_replace_feature_cache(
    fake_minimol: list[_FakeMiniMol],
    tmp_path: Path,
) -> None:
    """Prediction batches remain ordinary live/LRU requests."""
    feature_path = tmp_path / "minimol-features.npy"
    encoder = minimol_module.MiniMolSmilesFixedEncoder(
        feature_cache_path=feature_path,
    )
    encoder.encode(["CC", "CO"], device=torch.device("cpu"))
    feature_bytes = feature_path.read_bytes()
    manifest_bytes = Path(f"{feature_path}.json").read_bytes()
    encoder.encode(["CCC"], device=torch.device("cpu"))

    fresh_encoder = minimol_module.MiniMolSmilesFixedEncoder(
        feature_cache_path=feature_path,
    )
    combined = fresh_encoder.encode(["CC", "CO", "CN"], device=torch.device("cpu"))

    manifest = json.loads(Path(f"{feature_path}.json").read_text(encoding="utf-8"))
    assert manifest["row_count"] == 2
    assert combined.shape == (3, 512)
    assert combined[2, 0].item() == pytest.approx(float(sum(map(ord, "CN"))))
    assert fake_minimol[0].calls == [["CC", "CO"], ["CCC"]]
    assert fake_minimol[1].calls == [["CN"]]
    assert feature_path.read_bytes() == feature_bytes
    assert Path(f"{feature_path}.json").read_bytes() == manifest_bytes


def test_minimol_dkl_feature_cache_does_not_constrain_predictions(
    fake_minimol: list[_FakeMiniMol],
    tmp_path: Path,
) -> None:
    """DKL fitting uses persistence while prediction remains live encoding."""
    feature_path = tmp_path / "minimol-features.npy"
    encoder = minimol_module.MiniMolSmilesEncoder(
        latent_dim=4,
        feature_cache_path=feature_path,
    )
    surrogate = ExactDKLSurrogate(
        encoder=encoder,
        training_params=DKLTrainingConfig(epochs=1, lr=1e-2),
        standardize_outputs=False,
    )

    surrogate.fit(
        [
            Observation(x="CC", y=1.0),
            Observation(x="CO", y=2.0),
        ]
    )
    prediction = surrogate.predict([Candidate(x="CCC")])

    assert len(prediction["mean"]) == 1
    assert fake_minimol[0].calls == [["CC", "CO"], ["CCC"]]


def test_feature_cache_uses_one_extraction_across_processes(
    tmp_path: Path,
) -> None:
    """Concurrent first fits serialize and publish one valid artifact."""
    context = multiprocessing.get_context("fork")
    feature_path = tmp_path / "minimol-features.npy"
    call_log_path = tmp_path / "backend-calls.log"
    ready_queue = context.Queue()
    result_queue = context.Queue()
    start_event = context.Event()
    processes = [
        context.Process(
            target=_encode_feature_cache_in_process,
            args=(
                str(feature_path),
                ready_queue,
                start_event,
                str(call_log_path),
                result_queue,
            ),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    for _ in processes:
        ready_queue.get(timeout=30)
    start_event.set()
    for process in processes:
        process.join(timeout=30)

    assert all(process.exitcode == 0 for process in processes)
    assert [result_queue.get(timeout=5) for _ in processes] == [
        (2, 512),
        (2, 512),
    ]
    assert call_log_path.read_text(encoding="utf-8").splitlines() == ["call"]
    assert feature_path.is_file()
    assert Path(f"{feature_path}.json").is_file()
    assert Path(f"{feature_path}.lock").is_file()


def test_persistent_feature_cache_encodes_only_a_verified_suffix(
    fake_minimol: list[_FakeMiniMol],
    tmp_path: Path,
) -> None:
    """Inputs appended after the cached prefix are encoded live."""
    feature_path = tmp_path / "minimol-features.npy"
    first_encoder = minimol_module.MiniMolSmilesFixedEncoder(
        feature_cache_path=feature_path,
    )
    prefix = first_encoder.encode(
        ["CC", "CO"],
        device=torch.device("cpu"),
    )

    second_encoder = minimol_module.MiniMolSmilesFixedEncoder(
        feature_cache_path=feature_path,
    )
    combined = second_encoder.encode(
        ["CC", "CO", "CN"],
        device=torch.device("cpu"),
    )

    assert torch.equal(combined[:2], prefix)
    assert combined[2, 0].item() == pytest.approx(float(sum(map(ord, "CN"))))
    assert fake_minimol[0].calls == [["CC", "CO"]]
    assert fake_minimol[1].calls == [["CN"]]


def test_persistent_feature_cache_falls_back_for_reordered_inputs(
    fake_minimol: list[_FakeMiniMol],
    tmp_path: Path,
) -> None:
    """Reordering cached inputs uses live encoding without replacing the cache."""
    feature_path = tmp_path / "minimol-features.npy"
    encoder = minimol_module.MiniMolSmilesFixedEncoder(
        feature_cache_path=feature_path,
    )
    encoder.encode(
        ["CC", "CO", "CC"],
        device=torch.device("cpu"),
    )
    feature_bytes = feature_path.read_bytes()
    manifest_bytes = Path(f"{feature_path}.json").read_bytes()

    fresh_encoder = minimol_module.MiniMolSmilesFixedEncoder(
        feature_cache_path=feature_path,
    )
    features = fresh_encoder.encode(
        ["CO", "CC", "CC"],
        device=torch.device("cpu"),
    )

    assert features[:, 0].tolist() == pytest.approx(
        [float(sum(map(ord, value))) for value in ["CO", "CC", "CC"]]
    )
    assert fake_minimol[1].calls == [["CO", "CC"]]
    assert feature_path.read_bytes() == feature_bytes
    assert Path(f"{feature_path}.json").read_bytes() == manifest_bytes


def test_persistent_feature_cache_falls_back_for_shorter_inputs(
    fake_minimol: list[_FakeMiniMol],
    tmp_path: Path,
) -> None:
    """A shorter request uses live encoding without replacing the cache."""
    feature_path = tmp_path / "minimol-features.npy"
    encoder = minimol_module.MiniMolSmilesFixedEncoder(
        feature_cache_path=feature_path,
    )
    encoder.encode(["CC", "CO"], device=torch.device("cpu"))
    feature_bytes = feature_path.read_bytes()
    manifest_bytes = Path(f"{feature_path}.json").read_bytes()

    fresh_encoder = minimol_module.MiniMolSmilesFixedEncoder(
        feature_cache_path=feature_path,
    )
    features = fresh_encoder.encode(["CC"], device=torch.device("cpu"))

    assert features[:, 0].tolist() == pytest.approx([float(sum(map(ord, "CC")))])
    assert fake_minimol[1].calls == [["CC"]]
    assert feature_path.read_bytes() == feature_bytes
    assert Path(f"{feature_path}.json").read_bytes() == manifest_bytes


def test_persistent_feature_cache_rejects_encoder_identity_changes(
    fake_minimol: list[_FakeMiniMol],
    tmp_path: Path,
) -> None:
    """A different checkpoint cannot reuse an existing feature artifact."""
    first_checkpoint = tmp_path / "first.pth"
    second_checkpoint = tmp_path / "second.pth"
    first_checkpoint.write_bytes(b"first")
    second_checkpoint.write_bytes(b"second")
    feature_path = tmp_path / "minimol-features.npy"
    first_encoder = minimol_module.MiniMolSmilesFixedEncoder(
        checkpoint_path=first_checkpoint,
        feature_cache_path=feature_path,
    )
    first_encoder.encode(["CC"], device=torch.device("cpu"))

    second_encoder = minimol_module.MiniMolSmilesFixedEncoder(
        checkpoint_path=second_checkpoint,
        feature_cache_path=feature_path,
    )
    with pytest.raises(ValueError, match="encoder identity mismatch"):
        second_encoder.encode(["CC"], device=torch.device("cpu"))


def test_persistent_feature_cache_rejects_incomplete_manifest(
    fake_minimol: list[_FakeMiniMol],
    tmp_path: Path,
) -> None:
    """A manifest without its completion metadata is never reused."""
    feature_path = tmp_path / "minimol-features.npy"
    encoder = minimol_module.MiniMolSmilesFixedEncoder(
        feature_cache_path=feature_path,
    )
    encoder.encode(["CC"], device=torch.device("cpu"))
    manifest_path = Path(f"{feature_path}.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["complete"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    fresh_encoder = minimol_module.MiniMolSmilesFixedEncoder(
        feature_cache_path=feature_path,
    )
    with pytest.raises(ValueError, match="missing fields: complete"):
        fresh_encoder.encode(["CC"], device=torch.device("cpu"))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("dtype", "float64", "dtype mismatch"),
        ("shape", [1, 511], "shape metadata"),
    ],
)
def test_persistent_feature_cache_rejects_array_metadata_changes(
    fake_minimol: list[_FakeMiniMol],
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    """Manifest dtype and shape changes cannot bypass cache verification."""
    feature_path = tmp_path / "minimol-features.npy"
    encoder = minimol_module.MiniMolSmilesFixedEncoder(
        feature_cache_path=feature_path,
    )
    encoder.encode(["CC"], device=torch.device("cpu"))
    manifest_path = Path(f"{feature_path}.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    fresh_encoder = minimol_module.MiniMolSmilesFixedEncoder(
        feature_cache_path=feature_path,
    )
    with pytest.raises(ValueError, match=message):
        fresh_encoder.encode(["CC"], device=torch.device("cpu"))


def test_persistent_feature_cache_rejects_truncated_features(
    fake_minimol: list[_FakeMiniMol],
    tmp_path: Path,
) -> None:
    """A truncated .npy file fails before any stale rows are returned."""
    feature_path = tmp_path / "minimol-features.npy"
    encoder = minimol_module.MiniMolSmilesFixedEncoder(
        feature_cache_path=feature_path,
    )
    encoder.encode(["CC"], device=torch.device("cpu"))
    feature_path.write_bytes(feature_path.read_bytes()[:-1])

    fresh_encoder = minimol_module.MiniMolSmilesFixedEncoder(
        feature_cache_path=feature_path,
    )
    with pytest.raises(ValueError, match="file size mismatch"):
        fresh_encoder.encode(["CC"], device=torch.device("cpu"))


def test_zero_cache_size_disables_only_the_lru(
    fake_minimol: list[_FakeMiniMol],
) -> None:
    """A zero cache size still permits ordinary live extraction."""
    encoder = minimol_module.MiniMolSmilesFixedEncoder(
        cache_size=0,
    )
    encoder.encode(["CC"], device=torch.device("cpu"))
    encoder.encode(["CC"], device=torch.device("cpu"))

    assert fake_minimol[0].calls == [["CC"], ["CC"]]


def test_persistent_feature_cache_cleans_up_after_atomic_write_failure(
    fake_minimol: list[_FakeMiniMol],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A failed manifest commit leaves no misleading partial artifact."""
    feature_path = tmp_path / "minimol-features.npy"
    real_replace = minimol_module.os.replace
    replace_calls = 0

    def fail_manifest_commit(source: str, destination: str) -> None:
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 2:
            raise OSError("simulated manifest commit failure")
        real_replace(source, destination)

    monkeypatch.setattr(minimol_module.os, "replace", fail_manifest_commit)
    encoder = minimol_module.MiniMolSmilesFixedEncoder(
        feature_cache_path=feature_path,
    )
    with pytest.raises(OSError, match="manifest commit"):
        encoder.encode(["CC"], device=torch.device("cpu"))

    assert not feature_path.exists()
    assert not Path(f"{feature_path}.json").exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_latent_minimol_encoder_persists_only_fixed_fingerprints(
    fake_minimol: list[_FakeMiniMol],
    tmp_path: Path,
) -> None:
    """The latent wrapper reuses fixed features without caching projections."""
    feature_path = tmp_path / "minimol-features.npy"
    first_encoder = minimol_module.MiniMolSmilesEncoder(
        latent_dim=3,
        feature_cache_path=feature_path,
    )
    prepared = first_encoder.prepare_inputs(
        ["CC", "CO"],
        device=torch.device("cpu"),
    )
    projected = first_encoder(prepared)
    projected.sum().backward()

    second_encoder = minimol_module.MiniMolSmilesEncoder(
        latent_dim=3,
        feature_cache_path=feature_path,
    )
    reused = second_encoder.prepare_inputs(
        ["CC", "CO"],
        device=torch.device("cpu"),
    )

    assert second_encoder.feature_cache_path == feature_path
    assert torch.equal(reused, prepared)
    assert first_encoder.projection.weight.grad is not None
    assert fake_minimol[0].calls == [["CC", "CO"]]
    assert len(fake_minimol) == 1


def test_projection_is_trainable_while_minimol_inference_is_frozen(
    fake_minimol: list[_FakeMiniMol],
) -> None:
    """Backbone extraction runs without gradients while projection receives them."""
    encoder = minimol_module.MiniMolSmilesEncoder(latent_dim=3)
    inputs = encoder.prepare_inputs(["CC"], device=torch.device("cpu"))

    output = encoder(inputs)
    output.sum().backward()

    assert output.shape == (1, 3)
    assert encoder.projection.weight.grad is not None
    assert encoder.projection.bias.grad is not None
    assert fake_minimol[0].grad_enabled == [False]


@pytest.mark.parametrize(
    ("output", "exception", "message"),
    [
        (torch.zeros(1, 512), TypeError, "list of fingerprint"),
        ([torch.zeros(512), torch.zeros(512)], ValueError, "fingerprints"),
        ([torch.zeros(511)], ValueError, "must have shape"),
        ([torch.full((512,), float("nan"))], ValueError, "non-finite"),
    ],
)
def test_prepare_inputs_validates_minimol_outputs(
    monkeypatch: pytest.MonkeyPatch,
    output: object,
    exception: type[Exception],
    message: str,
) -> None:
    """Malformed or non-finite upstream fingerprints should fail explicitly."""

    class _MalformedMiniMol:
        def __init__(
            self,
            *,
            batch_size: int,
            checkpoint_path: object = None,
        ) -> None:
            del batch_size
            del checkpoint_path

        def __call__(self, smiles: list[str]) -> object:
            del smiles
            return output

    monkeypatch.setattr(
        minimol_module,
        "_load_minimol",
        lambda: _MalformedMiniMol,
    )
    encoder = minimol_module.MiniMolSmilesEncoder()

    with pytest.raises(exception, match=message):
        encoder.prepare_inputs(["CC"], device=torch.device("cpu"))


@pytest.mark.parametrize("surrogate_type", [ExactDKLSurrogate, VariationalDKLSurrogate])
def test_minimol_encoder_trains_exact_and_variational_dkl(
    fake_minimol: list[_FakeMiniMol],
    surrogate_type: type[ExactDKLSurrogate] | type[VariationalDKLSurrogate],
) -> None:
    """Both DKL variants should fit and predict through MiniMol fingerprints."""
    encoder = minimol_module.MiniMolSmilesEncoder(latent_dim=4, cache_size=8)
    kwargs: dict[str, object] = {}
    if surrogate_type is VariationalDKLSurrogate:
        kwargs["num_inducing"] = 2
    surrogate = surrogate_type(
        encoder=encoder,
        training_params=DKLTrainingConfig(epochs=1, lr=1e-2),
        standardize_outputs=False,
        **kwargs,
    )

    surrogate.fit(
        [
            Observation(x="CC", y=1.0),
            Observation(x="CO", y=2.0),
            Observation(x="CN", y=3.0),
        ]
    )
    prediction = surrogate.predict(
        [Candidate(x="CC"), Candidate(x="CCC")],
    )

    assert surrogate.is_fitted()
    assert len(prediction["mean"]) == 2
    assert len(prediction["std"]) == 2
    assert torch.isfinite(torch.tensor(prediction["mean"])).all()
    assert torch.isfinite(torch.tensor(prediction["std"])).all()


def test_minimol_encoder_runs_a_smiles_active_learning_round(
    fake_minimol: list[_FakeMiniMol],
) -> None:
    """A string candidate pool can complete one active-learning round."""
    surrogate = ExactDKLSurrogate(
        encoder=minimol_module.MiniMolSmilesEncoder(latent_dim=4),
        training_params=DKLTrainingConfig(epochs=1, lr=1e-2),
        standardize_outputs=False,
    )
    dataset = ListDataset()
    dataset.add_observations(
        [
            Observation(x="CC", y=2.0, fidelity=1),
            Observation(x="CO", y=2.0, fidelity=1),
        ]
    )

    active_learning(
        dataset=dataset,
        surrogate=surrogate,
        acquisition=UpperConfidenceBound(beta=1.0),
        sampler=PoolScoreSampler(
            candidate_pool=[
                Candidate(x="CCC", fidelity=1),
                Candidate(x="CN", fidelity=1),
            ],
            num_samples=2,
        ),
        selector=TopKAcquisitionSelector(num_samples=1),
        oracle=MultiFidelityOracle(
            fidelity_configs={
                1: {
                    "cost_per_sample": 1.0,
                    "fidelity_confidence": 1.0,
                    "score_fn": lambda value: float(len(value)),
                }
            }
        ),
        budget=Budget(
            available_budget=1.0,
            schedule=lambda _: 1.0,
            max_rounds=1,
        ),
    )

    assert len(dataset.get_observations_iterable()) == 3
