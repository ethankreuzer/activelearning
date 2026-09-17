"""Tests for the collaborator's full-trunk MiniMol AmpC encoder."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

import activelearning.applications.molecules.minimol_ampc_encoder as ampc_module


class _FakeAmpcEncoder:
    """Deterministic replacement for the collaborator's pooled512 encoder."""

    def __init__(
        self,
        outputs: Any = None,
        invalid: list[str] | None = None,
    ) -> None:
        self.outputs = outputs
        self.invalid = invalid or []
        self.calls: list[tuple[list[str], int]] = []
        self.options: list[dict[str, Any]] = []

    def encode(
        self,
        smiles: list[str],
        *,
        batch_size: int,
        **kwargs: Any,
    ) -> Any:
        """Record the request and return the configured backend output."""
        self.calls.append((list(smiles), batch_size))
        self.options.append(kwargs)
        if callable(self.outputs):
            return self.outputs(smiles)
        return self.outputs

    def featurize(
        self,
        smiles: list[str],
        *,
        on_invalid: str,
    ) -> tuple[list[object], list[str], list[str]]:
        """Return the configured invalid SMILES for skip-mode featurization."""
        assert on_invalid == "skip"
        invalid = [smile for smile in smiles if smile in self.invalid]
        kept = [smile for smile in smiles if smile not in invalid]
        return [], kept, invalid


def _make_checkpoint_package(tmp_path: Path) -> tuple[Path, Path]:
    """Create a checkpoint path next to a fake bundled package."""
    package_path = tmp_path / "minimol_ampc_encoder"
    (package_path / "minimol_ampc").mkdir(parents=True)
    checkpoint_path = package_path / "model" / "final.pt"
    checkpoint_path.parent.mkdir()
    checkpoint_path.touch()
    return checkpoint_path, package_path


def test_ampc_encoder_loads_backend_and_uses_pooled512(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The adapter passes its paths/device and inherits feature caching."""
    checkpoint_path, package_path = _make_checkpoint_package(tmp_path)
    backend = _FakeAmpcEncoder(
        outputs=lambda smiles: np.asarray(
            [[float(sum(map(ord, value)))] * 512 for value in smiles],
            dtype=np.float32,
        )
    )
    loaded: dict[str, object] = {}
    contexts: list[str] = []

    @contextmanager
    def fake_graphium_compatibility() -> Any:
        """Record the scoped Graphium dtype workaround."""
        contexts.append("enter")
        try:
            yield
        finally:
            contexts.append("exit")

    def fake_loader(
        checkpoint: Path,
        package: Path | None,
        device: str | torch.device | None,
    ) -> _FakeAmpcEncoder:
        """Record the loader inputs and return the fake backend."""
        loaded.update(checkpoint=checkpoint, package=package, device=device)
        return backend

    monkeypatch.setattr(ampc_module, "_load_ampc_encoder", fake_loader)
    monkeypatch.setattr(
        ampc_module,
        "_graphium_float32_compatibility",
        fake_graphium_compatibility,
    )

    encoder = ampc_module.MiniMolAmpcSmilesEncoder(
        checkpoint_path=checkpoint_path,
        batch_size=2,
        latent_dim=4,
    )
    prepared = encoder.prepare_inputs(
        ["CC", "CC", "CO"],
        device=torch.device("cpu"),
    )

    assert loaded == {
        "checkpoint": checkpoint_path.resolve(),
        "package": package_path.resolve(),
        "device": "cpu",
    }
    assert prepared.shape == (3, 512)
    assert prepared.dtype == torch.float32
    assert prepared[:, 0].tolist() == pytest.approx(
        [float(sum(map(ord, value))) for value in ["CC", "CC", "CO"]]
    )
    assert backend.calls == [(["CC", "CO"], 2)]
    assert backend.options == [{"on_invalid": "skip"}]
    assert contexts == ["enter", "exit"]


def test_ampc_encoder_substitutes_zero_rows_for_invalid_molecules(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """MiniMol-invalid inputs retain batch alignment with zero fingerprints."""
    checkpoint_path, package_path = _make_checkpoint_package(tmp_path)

    backend = _FakeAmpcEncoder(
        outputs=lambda smiles: np.asarray(
            [
                [float(index + 1)] * 512
                for index, _ in enumerate(
                    smile for smile in smiles if smile != "unsupported"
                )
            ],
            dtype=np.float32,
        ),
        invalid=["unsupported"],
    )

    @contextmanager
    def fake_graphium_compatibility() -> Any:
        yield

    monkeypatch.setattr(
        ampc_module,
        "_load_ampc_encoder",
        lambda checkpoint, package, device: backend,
    )
    monkeypatch.setattr(
        ampc_module,
        "_graphium_float32_compatibility",
        fake_graphium_compatibility,
    )
    encoder = ampc_module.MiniMolAmpcSmilesFixedEncoder(
        checkpoint_path=checkpoint_path,
        package_path=package_path,
    )

    with caplog.at_level("WARNING", logger=ampc_module.__name__):
        features = encoder.encode(
            ["CC", "unsupported", "CO"],
            device=torch.device("cpu"),
        )

    assert features.shape == (3, 512)
    assert features[:, 0].tolist() == [1.0, 0.0, 2.0]
    assert backend.options == [{"on_invalid": "skip"}]
    assert "using zero fingerprints" in caplog.text


def test_ampc_fixed_encoder_returns_pooled512_without_projection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The fixed AmpC encoder exposes pooled512 directly."""
    checkpoint_path, package_path = _make_checkpoint_package(tmp_path)
    backend = _FakeAmpcEncoder(
        outputs=np.ones((2, 512), dtype=np.float32),
    )

    @contextmanager
    def fake_graphium_compatibility() -> Any:
        yield

    monkeypatch.setattr(
        ampc_module,
        "_load_ampc_encoder",
        lambda checkpoint, package, device: backend,
    )
    monkeypatch.setattr(
        ampc_module,
        "_graphium_float32_compatibility",
        fake_graphium_compatibility,
    )

    encoder = ampc_module.MiniMolAmpcSmilesFixedEncoder(
        checkpoint_path=checkpoint_path,
        package_path=package_path,
    )
    features = encoder.encode(["CC", "CO"], device=torch.device("cpu"))

    assert encoder.feature_dim == 512
    assert features.shape == (2, 512)
    assert not hasattr(encoder, "projection")


def test_ampc_feature_cache_reuse_is_lazy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A cache-only AmpC fit does not load a second checkpoint backend."""
    checkpoint_path, package_path = _make_checkpoint_package(tmp_path)
    backends: list[_FakeAmpcEncoder] = []

    def fake_loader(
        checkpoint: Path,
        package: Path | None,
        device: str | torch.device | None,
    ) -> _FakeAmpcEncoder:
        del checkpoint, package, device
        backend = _FakeAmpcEncoder(
            outputs=lambda smiles: np.asarray(
                [[float(sum(map(ord, value)))] * 512 for value in smiles],
                dtype=np.float32,
            )
        )
        backends.append(backend)
        return backend

    monkeypatch.setattr(ampc_module, "_load_ampc_encoder", fake_loader)
    cache_path = tmp_path / "ampc-features.npy"
    first_encoder = ampc_module.MiniMolAmpcSmilesFixedEncoder(
        checkpoint_path=checkpoint_path,
        package_path=package_path,
        feature_cache_path=cache_path,
    )
    first_encoder.encode(["CC", "CO"], device=torch.device("cpu"))

    second_encoder = ampc_module.MiniMolAmpcSmilesFixedEncoder(
        checkpoint_path=checkpoint_path,
        package_path=package_path,
        feature_cache_path=cache_path,
    )
    second_encoder.encode(["CC", "CO"], device=torch.device("cpu"))

    assert len(backends) == 1
    assert backends[0].calls == [(["CC", "CO"], 100)]


def test_ampc_backend_loader_configures_worker_safe_featurizer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The loader propagates the SciPy dtype workaround to Graphium workers."""
    checkpoint_path, package_path = _make_checkpoint_package(tmp_path)
    datamodule = SimpleNamespace(
        featurization={"explicit_H": False},
        smiles_transformer=object(),
    )
    backend = SimpleNamespace(
        model=SimpleNamespace(
            trunk=SimpleNamespace(datamodule=datamodule),
        ),
    )
    contexts: list[str] = []

    class _FakeEncoder:
        @classmethod
        def load(
            cls,
            checkpoint: Path,
            *,
            device: str | torch.device | None,
        ) -> Any:
            """Return a backend exposing the collaborator's data module."""
            assert checkpoint == checkpoint_path
            assert device == "cpu"
            return backend

    @contextmanager
    def fake_graphium_compatibility() -> Any:
        """Record the scoped dtype workaround."""
        contexts.append("enter")
        try:
            yield
        finally:
            contexts.append("exit")

    monkeypatch.setattr(
        ampc_module.importlib,
        "import_module",
        lambda name: SimpleNamespace(MiniMolAmpcEncoder=_FakeEncoder),
    )
    monkeypatch.setattr(
        ampc_module,
        "_graphium_float32_compatibility",
        fake_graphium_compatibility,
    )

    loaded = ampc_module._load_ampc_encoder(
        checkpoint_path,
        package_path,
        "cpu",
    )

    assert loaded is backend
    assert contexts == ["enter", "exit"]
    assert datamodule.smiles_transformer.func is ampc_module._minimol_molecule_transform
    assert datamodule.smiles_transformer.keywords == {"explicit_H": False}


def test_ampc_encoder_rejects_invalid_package_path(
    tmp_path: Path,
) -> None:
    """An explicitly configured package path must contain minimol_ampc."""
    checkpoint_path = tmp_path / "final.pt"
    checkpoint_path.touch()

    with pytest.raises(FileNotFoundError, match="package_path"):
        ampc_module.MiniMolAmpcSmilesEncoder(
            checkpoint_path=checkpoint_path,
            package_path=tmp_path / "missing-package",
        )


@pytest.mark.parametrize(
    ("outputs", "message"),
    [
        (np.zeros((1, 511), dtype=np.float32), "must have shape"),
        (np.full((1, 512), np.nan, dtype=np.float32), "non-finite"),
    ],
)
def test_ampc_encoder_validates_pooled512_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    outputs: np.ndarray,
    message: str,
) -> None:
    """Malformed or non-finite collaborator embeddings fail explicitly."""
    checkpoint_path, package_path = _make_checkpoint_package(tmp_path)
    backend = _FakeAmpcEncoder(outputs=outputs)
    monkeypatch.setattr(
        ampc_module,
        "_load_ampc_encoder",
        lambda checkpoint, package, device: backend,
    )
    encoder = ampc_module.MiniMolAmpcSmilesEncoder(
        checkpoint_path=checkpoint_path,
        package_path=package_path,
    )

    with pytest.raises(ValueError, match=message):
        encoder.prepare_inputs(["CC"], device=torch.device("cpu"))
