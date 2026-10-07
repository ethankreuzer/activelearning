"""MiniMol-backed SMILES encoders for molecular surrogate models."""

from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import json
import os
import tempfile
import threading
import warnings
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch import Tensor, nn

from activelearning.applications.molecules._optional import (
    missing_molecules_dependency_error,
)
from activelearning.surrogate.encoder import FixedEncoder, LatentEncoder

__all__ = [
    "MINIMOL_ACTIVATIONS",
    "MiniMolSmilesEncoder",
    "MiniMolSmilesFixedEncoder",
]

MINIMOL_FINGERPRINT_DIM = 512
_FEATURE_CACHE_FORMAT_VERSION = 2
_HASH_CHUNK_SIZE = 1024 * 1024
#: Training rows used to set the input scale of an unprojected encoder: 2000
#: rows are two million pairwise distances, plenty for a median.
_CALIBRATION_ROWS = 2000

#: Activations available on the trainable DKL projection. ``"none"`` keeps the
#: projection linear, which makes the kernel a learned Mahalanobis metric on the
#: fingerprints rather than a genuinely non-stationary one.
MINIMOL_ACTIVATIONS: Mapping[str, Callable[[], nn.Module]] = {
    "none": nn.Identity,
    "gelu": nn.GELU,
    "relu": nn.ReLU,
}


@contextmanager
def _graphium_float32_compatibility() -> Iterator[None]:
    """Use a SciPy-compatible dtype while Graphium featurizes a molecule.

    Graphium 2.4.7's graph-dict helper drops its configured dtype before
    calling the adjacency helper. The resulting ``float16`` sparse matrix is
    rejected by SciPy. The patch is scoped to the active featurization call
    and is applied independently in each process used by Graphium's
    featurizer.
    """
    import numpy as np
    from graphium.features import featurizer

    original = featurizer.mol_to_adj_and_features

    def _featurize_with_float32(*args: Any, **kwargs: Any) -> Any:
        kwargs["dtype"] = np.float32
        return original(*args, **kwargs)

    featurizer.mol_to_adj_and_features = _featurize_with_float32
    try:
        yield
    finally:
        featurizer.mol_to_adj_and_features = original


def _minimol_molecule_transform(molecule: Any, **kwargs: Any) -> Any:
    """Featurize one molecule with Graphium's sparse dtype workaround."""
    from graphium.features import featurizer

    with _graphium_float32_compatibility():
        return featurizer.mol_to_pyggraph(molecule, **kwargs)


def _load_minimol_checkpoint(model: Any, checkpoint_path: Path) -> None:
    """Load a predictor state dict into a constructed MiniMol model."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    state_dict: Any = checkpoint
    if isinstance(checkpoint, Mapping) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    if not isinstance(state_dict, Mapping) or not all(
        isinstance(key, str) for key in state_dict
    ):
        raise TypeError(
            "MiniMol checkpoint must be a state dict or a mapping containing "
            "a string-keyed 'state_dict'."
        )

    fingerprinter = getattr(model, "predictor", None)
    predictor = getattr(fingerprinter, "predictor", None)
    if not isinstance(predictor, nn.Module):
        raise TypeError(
            "MiniMol does not expose the expected Graphium predictor for "
            "checkpoint loading."
        )

    target_keys = set(predictor.state_dict())
    if not target_keys.intersection(state_dict):
        raise ValueError(
            f"MiniMol checkpoint {checkpoint_path} contains no parameters "
            "matching the constructed predictor."
        )
    predictor.load_state_dict(state_dict, strict=False)


def _hash_file(path: Path) -> str:
    """Return the SHA-256 digest of a file's exact bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_ordered_strings(values: Sequence[str]) -> str:
    """Hash an ordered string sequence with length-framed UTF-8 values."""
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    """Hash JSON data after applying a deterministic canonical encoding."""
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _package_version(package_name: str) -> str | None:
    """Return an installed package version without requiring package metadata."""
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _bundled_minimol_checkpoint() -> Path | None:
    """Find MiniMol's bundled state dict without importing the package."""
    try:
        distribution = importlib.metadata.distribution("minimol")
    except importlib.metadata.PackageNotFoundError:
        return None
    checkpoint_path = Path(
        distribution.locate_file("minimol/ckpts/minimol_v1/state_dict.pth")
    )
    return checkpoint_path if checkpoint_path.is_file() else None


def _temporary_path(path: Path) -> Path:
    """Reserve a temporary sibling path for an atomic cache-file write."""
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    return Path(name)


@contextmanager
def _exclusive_file_lock(path: Path) -> Iterator[None]:
    """Serialize cache inspection and publication across local processes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _build_compatible_minimol(
    minimol_class: type[Any],
    *,
    batch_size: int,
    checkpoint_path: Path | None = None,
) -> Any:
    """Construct MiniMol with the Graphium sparse dtype workaround."""
    with _graphium_float32_compatibility():
        model = minimol_class(batch_size=batch_size)

    if checkpoint_path is not None:
        _load_minimol_checkpoint(model, checkpoint_path)

    model.datamodule.smiles_transformer = partial(
        _minimol_molecule_transform,
        **model.datamodule.featurization,
    )
    return model


def _load_minimol() -> Callable[..., Any]:
    """Load a MiniMol constructor on demand."""
    try:
        from minimol import Minimol
    except ImportError as error:  # pragma: no cover - optional dependency
        raise missing_molecules_dependency_error(
            "MiniMol SMILES encoder",
            error,
        ) from error

    return partial(_build_compatible_minimol, Minimol)


class MiniMolSmilesFixedEncoder(FixedEncoder):
    """Encode SMILES as fixed 512-dimensional MiniMol fingerprints."""

    feature_dim = MINIMOL_FINGERPRINT_DIM
    _cache_backend_name = "minimol.Minimol"
    _cache_package_names = ("minimol",)

    def __init__(
        self,
        *,
        batch_size: int = 100,
        cache_size: int = 4096,
        feature_cache_path: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        cache_only: bool = False,
    ) -> None:
        """Initialize the fixed MiniMol encoder.

        Parameters
        ----------
        batch_size : int, default=100
            Maximum number of SMILES sent to MiniMol per extraction batch.
        cache_size : int, default=4096
            Maximum number of detached CPU fingerprints retained by the LRU
            cache. Zero disables the in-memory LRU cache.
        feature_cache_path : Path or str, optional
            Path to a persistent ``.npy`` feature matrix. A JSON manifest and
            a lock file are stored beside it. The cache represents one exact
            ordered input prefix and is created on the first non-empty encode.
        checkpoint_path : Path or str, optional
            Optional predictor state-dict checkpoint to load over MiniMol's
            bundled pretrained weights.
        cache_only : bool, default=False
            Whether to forbid live MiniMol inference entirely. With a persistent
            cache a request that is not an exact ordered prefix of the cached
            input falls back to live encoding, which is correct but silent and
            slow. When a caller depends on the cache -- in particular when it is
            timing the run -- that fallback is a defect, so this turns it into an
            error. Requires ``feature_cache_path``.
        Raises
        ------
        ValueError
            If ``batch_size`` is not positive, ``cache_size`` is negative, or
            ``cache_only`` is set without a ``feature_cache_path``.
        FileNotFoundError
            If ``checkpoint_path`` is provided but does not point to a file.
        ImportError
            If live extraction is requested without MiniMol installed.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if cache_size < 0:
            raise ValueError("cache_size must be non-negative.")
        if cache_only and feature_cache_path is None:
            raise ValueError("cache_only requires a feature_cache_path.")
        resolved_checkpoint_path = (
            Path(checkpoint_path).expanduser() if checkpoint_path is not None else None
        )
        resolved_feature_cache_path = (
            Path(feature_cache_path).expanduser()
            if feature_cache_path is not None
            else None
        )
        if (
            resolved_feature_cache_path is not None
            and resolved_feature_cache_path.exists()
            and resolved_feature_cache_path.is_dir()
        ):
            raise ValueError(
                "feature_cache_path must name a file, not a directory: "
                f"{resolved_feature_cache_path}"
            )
        if (
            resolved_checkpoint_path is not None
            and not resolved_checkpoint_path.is_file()
        ):
            raise FileNotFoundError(
                "MiniMol checkpoint_path does not point to a file: "
                f"{resolved_checkpoint_path}"
            )

        self.batch_size = batch_size
        self.cache_size = cache_size
        self.feature_cache_path = resolved_feature_cache_path
        self.checkpoint_path = resolved_checkpoint_path
        self.cache_only = cache_only
        self._minimol: Any | None = None
        self._minimol_lock = threading.Lock()
        self._fingerprint_cache: OrderedDict[str, Tensor] = OrderedDict()
        self._persistent_features: np.memmap | None = None
        self._persistent_row_count: int | None = None
        self._persistent_manifest: dict[str, Any] | None = None

    def encode(
        self,
        values: Sequence[Any],
        *,
        device: torch.device,
    ) -> Tensor:
        """Convert raw SMILES into MiniMol fingerprint tensors."""
        strings = self._validate_smiles(values)
        if not strings:
            return torch.empty(
                (0, MINIMOL_FINGERPRINT_DIM),
                dtype=torch.float32,
                device=device,
            )
        if self.feature_cache_path is None:
            return self._encode_live(strings).to(device=device, dtype=torch.float32)

        lock_path = self._feature_cache_lock_path
        if lock_path is None:
            raise RuntimeError("MiniMol feature cache lock path is unavailable.")
        with _exclusive_file_lock(lock_path):
            if not self._load_feature_cache():
                live_features = self._encode_live(strings)
                self._write_feature_cache(strings, live_features)
                return live_features.to(device=device, dtype=torch.float32)
            cached_prefix = self._load_matching_feature_prefix(strings)
            if cached_prefix is not None:
                cached_count = cached_prefix.shape[0]
                if cached_count == len(strings):
                    return cached_prefix.to(device=device, dtype=torch.float32)

        if cached_prefix is None:
            return self._encode_live(strings).to(device=device, dtype=torch.float32)

        cached_count = cached_prefix.shape[0]
        live_suffix = self._encode_live(strings[cached_count:])
        return torch.cat((cached_prefix, live_suffix), dim=0).to(
            device=device,
            dtype=torch.float32,
        )

    def _build_minimol(self, checkpoint_path: Path | None) -> Any:
        """Construct stock MiniMol and optionally load its predictor state dict."""
        return _load_minimol()(
            batch_size=self.batch_size,
            checkpoint_path=checkpoint_path,
        )

    def _ensure_minimol_loaded(self) -> Any:
        """Construct MiniMol once, only when live extraction needs it.

        Raises
        ------
        RuntimeError
            If ``cache_only`` is set, since reaching this point means the
            persistent cache did not cover the request.
        """
        if self.cache_only:
            raise RuntimeError(
                "This MiniMol encoder is cache-only, but it was asked to encode "
                f"SMILES that the feature cache {self.feature_cache_path} does not "
                "cover as an exact ordered prefix. Live inference is disabled "
                "because falling back to it silently would take hours and "
                "invalidate any timing measurement. Rebuild the cache for this "
                "exact ordered input, or construct the encoder without cache_only."
            )
        if self._minimol is None:
            with self._minimol_lock:
                if self._minimol is None:
                    self._minimol = self._build_minimol(self.checkpoint_path)
        return self._minimol

    @staticmethod
    def _validate_smiles(values: Sequence[Any]) -> list[str]:
        """Validate and materialize raw SMILES values."""
        strings: list[str] = []
        for value in values:
            if not isinstance(value, str):
                raise ValueError(
                    "MiniMol SMILES fixed encoders require string inputs, got "
                    f"{type(value).__name__}."
                )
            strings.append(value)
        return strings

    def _encode_live(self, strings: list[str]) -> Tensor:
        """Encode strings using only the bounded in-memory cache."""
        resolved: dict[str, Tensor] = {}
        missing: list[str] = []
        missing_set: set[str] = set()
        for string in strings:
            if string in resolved:
                continue
            cached = self._fingerprint_cache.get(string) if self.cache_size else None
            if cached is not None:
                self._fingerprint_cache.move_to_end(string)
                resolved[string] = cached
            elif string not in missing_set:
                missing.append(string)
                missing_set.add(string)

        if missing:
            missing_features = self._extract_fingerprints(missing)
            for string, feature in zip(missing, missing_features):
                resolved[string] = feature
                if self.cache_size:
                    self._fingerprint_cache[string] = feature
                    self._fingerprint_cache.move_to_end(string)
                    while len(self._fingerprint_cache) > self.cache_size:
                        self._fingerprint_cache.popitem(last=False)

        return torch.stack([resolved[string] for string in strings], dim=0).to(
            device="cpu",
            dtype=torch.float32,
        )

    def _extract_fingerprints(self, smiles: list[str]) -> list[Tensor]:
        """Run frozen MiniMol inference and validate its fingerprint output."""
        minimol = self._ensure_minimol_loaded()
        with torch.inference_mode():
            outputs = minimol(smiles)
        if not isinstance(outputs, (list, tuple)):
            raise TypeError("MiniMol must return a list of fingerprint tensors.")
        if len(outputs) != len(smiles):
            raise ValueError(
                "MiniMol returned "
                f"{len(outputs)} fingerprints for {len(smiles)} SMILES."
            )

        features: list[Tensor] = []
        for index, output in enumerate(outputs):
            if not isinstance(output, Tensor):
                raise TypeError(
                    f"MiniMol fingerprint at index {index} is not a tensor."
                )
            if output.ndim != 1 or output.numel() != MINIMOL_FINGERPRINT_DIM:
                raise ValueError(
                    f"MiniMol fingerprint at index {index} must have shape "
                    f"({MINIMOL_FINGERPRINT_DIM},), got {tuple(output.shape)}."
                )
            feature = output.detach().to(device="cpu", dtype=torch.float32)
            if not torch.isfinite(feature).all():
                raise ValueError(
                    f"MiniMol fingerprint at index {index} contains non-finite values."
                )
            features.append(feature)
        return features

    @property
    def _feature_cache_manifest_path(self) -> Path | None:
        """Return the manifest path paired with the feature cache file."""
        if self.feature_cache_path is None:
            return None
        return Path(f"{self.feature_cache_path}.json")

    @property
    def _feature_cache_lock_path(self) -> Path | None:
        """Return the process-shared lock path for the feature cache."""
        if self.feature_cache_path is None:
            return None
        return Path(f"{self.feature_cache_path}.lock")

    def _cache_encoder_identity(self) -> dict[str, Any]:
        """Build identity metadata for validating persisted fingerprints."""
        package_versions = {
            package_name: version
            for package_name in self._cache_package_names
            if (version := _package_version(package_name)) is not None
        }
        checkpoint_path = self.checkpoint_path or _bundled_minimol_checkpoint()
        checkpoint_sha256 = (
            _hash_file(checkpoint_path) if checkpoint_path is not None else None
        )
        return {
            "encoder_class": (f"{type(self).__module__}.{type(self).__qualname__}"),
            "backend": self._cache_backend_name,
            "package_versions": package_versions,
            "fingerprint_spec": {
                "dimension": MINIMOL_FINGERPRINT_DIM,
                "dtype": "float32",
                "pooling": "global_max",
                "representation": "minimol_graph_fingerprint",
            },
            "checkpoint_sha256": checkpoint_sha256,
        }

    def _load_feature_cache(self) -> bool:
        """Load and structurally verify the configured feature cache."""
        feature_path = self.feature_cache_path
        manifest_path = self._feature_cache_manifest_path
        if feature_path is None or manifest_path is None:
            return False
        if self._persistent_features is not None:
            return True

        feature_exists = feature_path.is_file()
        manifest_exists = manifest_path.is_file()
        if not feature_exists and not manifest_exists:
            if feature_path.exists() or manifest_path.exists():
                raise ValueError(
                    "MiniMol feature cache paths must be regular files: "
                    f"{feature_path} and {manifest_path}."
                )
            return False
        if feature_exists != manifest_exists:
            raise ValueError(
                "MiniMol feature cache is incomplete; both the .npy feature "
                f"file and its manifest are required: {feature_path}, "
                f"{manifest_path}."
            )

        try:
            with manifest_path.open("r", encoding="utf-8") as stream:
                manifest = json.load(stream)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"MiniMol feature cache manifest is not valid JSON: {manifest_path}."
            ) from error
        if not isinstance(manifest, dict):
            raise ValueError("MiniMol feature cache manifest must contain an object.")

        required_fields = {
            "format_version",
            "complete",
            "encoder_class",
            "feature_dim",
            "dtype",
            "shape",
            "row_count",
            "input_sha256",
            "encoder_identity",
            "encoder_identity_sha256",
            "feature_file",
        }
        missing_fields = sorted(required_fields.difference(manifest))
        if missing_fields:
            raise ValueError(
                "MiniMol feature cache manifest is incomplete; missing fields: "
                + ", ".join(missing_fields)
            )
        if manifest["format_version"] != _FEATURE_CACHE_FORMAT_VERSION:
            raise ValueError(
                "Unsupported MiniMol feature cache format version: "
                f"{manifest['format_version']}."
            )
        if manifest["complete"] is not True:
            raise ValueError("MiniMol feature cache manifest is not marked complete.")

        expected_class = f"{type(self).__module__}.{type(self).__qualname__}"
        if manifest["encoder_class"] != expected_class:
            raise ValueError(
                "MiniMol feature cache encoder class mismatch: expected "
                f"{expected_class}, got {manifest['encoder_class']}."
            )
        if manifest["feature_dim"] != MINIMOL_FINGERPRINT_DIM:
            raise ValueError(
                "MiniMol feature cache feature dimension mismatch: expected "
                f"{MINIMOL_FINGERPRINT_DIM}, got {manifest['feature_dim']}."
            )
        if manifest["dtype"] != "float32":
            raise ValueError(
                "MiniMol feature cache dtype mismatch: expected float32, got "
                f"{manifest['dtype']}."
            )
        row_count = manifest["row_count"]
        shape = manifest["shape"]
        if (
            not isinstance(row_count, int)
            or isinstance(row_count, bool)
            or row_count < 0
            or shape != [row_count, MINIMOL_FINGERPRINT_DIM]
        ):
            raise ValueError(
                "MiniMol feature cache shape metadata must be "
                f"[{row_count}, {MINIMOL_FINGERPRINT_DIM}]."
            )

        stored_identity = manifest["encoder_identity"]
        stored_identity_hash = manifest["encoder_identity_sha256"]
        if _canonical_hash(stored_identity) != stored_identity_hash:
            raise ValueError(
                "MiniMol feature cache encoder identity hash is inconsistent "
                "with its manifest."
            )
        expected_identity_hash = _canonical_hash(self._cache_encoder_identity())
        if stored_identity_hash != expected_identity_hash:
            raise ValueError(
                "MiniMol feature cache encoder identity mismatch; the MiniMol "
                "package, checkpoint, or fingerprint specification changed."
            )

        feature_metadata = manifest["feature_file"]
        if not isinstance(feature_metadata, dict):
            raise ValueError("MiniMol feature cache file metadata must be an object.")
        if feature_metadata.get("name") != feature_path.name:
            raise ValueError(
                "MiniMol feature cache manifest points to a different feature "
                f"file: {feature_metadata.get('name')}."
            )
        expected_size = feature_metadata.get("byte_size")
        if not isinstance(expected_size, int) or isinstance(expected_size, bool):
            raise ValueError(
                "MiniMol feature cache manifest has an invalid feature byte size."
            )
        actual_size = feature_path.stat().st_size
        if actual_size != expected_size:
            raise ValueError(
                "MiniMol feature cache file size mismatch: expected "
                f"{expected_size} bytes, got {actual_size}."
            )

        try:
            features = np.load(
                feature_path,
                mmap_mode="r",
                allow_pickle=False,
            )
        except (OSError, ValueError) as error:
            raise ValueError(
                f"MiniMol feature cache file is not a valid .npy array: {feature_path}."
            ) from error
        if features.dtype != np.dtype(np.float32) or features.shape != tuple(shape):
            raise ValueError(
                "MiniMol feature cache array metadata mismatch: expected "
                f"dtype float32 and shape {tuple(shape)}, got dtype "
                f"{features.dtype} and shape {features.shape}."
            )

        self._persistent_features = features
        self._persistent_row_count = row_count
        self._persistent_manifest = manifest
        return True

    def _load_matching_feature_prefix(
        self,
        strings: list[str],
    ) -> Tensor | None:
        """Load the persisted prefix when it matches the input sequence."""
        if (
            self._persistent_features is None
            or self._persistent_row_count is None
            or self._persistent_manifest is None
        ):
            raise RuntimeError("MiniMol feature cache did not load its feature array.")

        row_count = self._persistent_row_count
        if len(strings) < row_count:
            return None
        expected_input_hash = self._persistent_manifest["input_sha256"]
        actual_input_hash = _hash_ordered_strings(strings[:row_count])
        if actual_input_hash != expected_input_hash:
            return None

        prefix = np.asarray(self._persistent_features[:row_count])
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The given NumPy array is not writable",
                category=UserWarning,
            )
            cached_features = torch.from_numpy(prefix)
        return cached_features.to(dtype=torch.float32)

    def _write_feature_cache(
        self,
        strings: list[str],
        features: Tensor,
    ) -> None:
        """Atomically publish a new feature cache."""
        feature_path = self.feature_cache_path
        manifest_path = self._feature_cache_manifest_path
        if feature_path is None or manifest_path is None:
            raise RuntimeError("MiniMol feature cache path is unavailable.")
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_feature_path = _temporary_path(feature_path)
        temporary_manifest_path = _temporary_path(manifest_path)
        feature_committed = False
        try:
            cpu_features = (
                features.detach()
                .to(
                    device="cpu",
                    dtype=torch.float32,
                )
                .contiguous()
            )
            feature_array = np.lib.format.open_memmap(
                temporary_feature_path,
                mode="w+",
                dtype=np.float32,
                shape=tuple(cpu_features.shape),
            )
            feature_values = cpu_features.numpy()
            for start in range(0, len(strings), self.batch_size):
                end = min(start + self.batch_size, len(strings))
                feature_array[start:end] = feature_values[start:end]
            feature_array.flush()
            del feature_array

            identity = self._cache_encoder_identity()
            manifest: dict[str, Any] = {
                "format_version": _FEATURE_CACHE_FORMAT_VERSION,
                "complete": True,
                "encoder_class": (f"{type(self).__module__}.{type(self).__qualname__}"),
                "feature_dim": MINIMOL_FINGERPRINT_DIM,
                "dtype": "float32",
                "shape": [len(strings), MINIMOL_FINGERPRINT_DIM],
                "row_count": len(strings),
                "input_sha256": _hash_ordered_strings(strings),
                "encoder_identity": identity,
                "encoder_identity_sha256": _canonical_hash(identity),
                "feature_file": {
                    "name": feature_path.name,
                    "byte_size": temporary_feature_path.stat().st_size,
                },
            }
            with temporary_manifest_path.open("w", encoding="utf-8") as stream:
                json.dump(manifest, stream, sort_keys=True, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())

            os.replace(temporary_feature_path, feature_path)
            feature_committed = True
            os.replace(temporary_manifest_path, manifest_path)
        except Exception:
            if feature_committed and not manifest_path.exists():
                feature_path.unlink(missing_ok=True)
            raise
        finally:
            temporary_feature_path.unlink(missing_ok=True)
            temporary_manifest_path.unlink(missing_ok=True)

        if not self._load_feature_cache():
            raise RuntimeError("MiniMol feature cache was not published.")


class MiniMolSmilesEncoder(LatentEncoder):
    """Encode SMILES with frozen MiniMol fingerprints and a trainable head.

    MiniMol returns fixed-width graph fingerprints rather than token IDs or
    hidden states from a PyTorch module. The fingerprints are kept frozen and
    passed through a trainable linear projection so the DKL surrogate can
    adapt the representation during fitting.

    An optional activation follows that projection. Without one the projection
    is linear, so the kernel it feeds only learns a low-rank metric on the
    fingerprints; with one the feature map is nonlinear and the kernel becomes
    non-stationary in fingerprint space, which is the point of deep kernel
    learning.

    ``latent_dim=None`` removes the projection altogether: the kernel then sees
    the fingerprints themselves, divided by one fixed scalar, and the encoder
    has no trainable parameters. Nothing in the fit can then move dissimilar
    molecules together. The scalar is set by :meth:`calibrate_inputs` to the
    median pairwise distance of the training fingerprints, so the kernel's
    lengthscales start at the scale of the data. It rescales every direction
    alike and leaves the fingerprint geometry as MiniMol produced it.
    """

    def __init__(
        self,
        *,
        batch_size: int = 100,
        latent_dim: int | None = 32,
        cache_size: int = 4096,
        feature_cache_path: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        activation: str = "none",
        cache_only: bool = False,
    ) -> None:
        """Initialize the MiniMol encoder and trainable projection."""
        super().__init__()
        if latent_dim is not None and latent_dim < 1:
            raise ValueError("latent_dim must be positive.")
        if activation not in MINIMOL_ACTIVATIONS:
            raise ValueError(
                f"Unknown activation {activation!r}; expected one of "
                f"{sorted(MINIMOL_ACTIVATIONS)}."
            )
        if latent_dim is None and activation != "none":
            raise ValueError(
                "latent_dim=None feeds the fingerprints to the kernel unprojected, "
                f"so there is no projection for activation {activation!r} to follow."
            )

        self.fixed_encoder = self._build_fixed_encoder(
            batch_size=batch_size,
            cache_size=cache_size,
            feature_cache_path=feature_cache_path,
            checkpoint_path=checkpoint_path,
            cache_only=cache_only,
        )
        self.batch_size = self.fixed_encoder.batch_size
        self.latent_dim = MINIMOL_FINGERPRINT_DIM if latent_dim is None else latent_dim
        self.cache_size = self.fixed_encoder.cache_size
        self.feature_cache_path = self.fixed_encoder.feature_cache_path
        self.checkpoint_path = self.fixed_encoder.checkpoint_path
        self.cache_only = self.fixed_encoder.cache_only
        self.projection = (
            None
            if latent_dim is None
            else nn.Linear(MINIMOL_FINGERPRINT_DIM, latent_dim)
        )
        if latent_dim is None:
            # A buffer, so it follows the runtime device and dtype and is saved
            # with the surrogate state. Registered only here, so the state of an
            # encoder with a projection keeps the keys it always had.
            self.register_buffer("input_scale", torch.ones(()))
        self.activation_name = activation
        self.activation = MINIMOL_ACTIVATIONS[activation]()

    def prepare_inputs(
        self,
        values: Sequence[Any],
        *,
        device: torch.device,
    ) -> Tensor:
        """Convert raw SMILES into cached or live MiniMol fingerprints."""
        return self.fixed_encoder.encode(values, device=device)

    def forward(self, model_inputs: Tensor) -> Tensor:
        """Project MiniMol fingerprints into the DKL latent space.

        Applies the configured activation after the projection. The default
        ``"none"`` is :class:`torch.nn.Identity`, so the output is unchanged.
        Without a projection the fingerprints are returned divided by
        ``input_scale``.
        """
        if model_inputs.ndim != 2:
            raise ValueError(
                "MiniMol fingerprints must be 2-D (B, 512), got "
                f"{tuple(model_inputs.shape)}."
            )
        if model_inputs.shape[-1] != MINIMOL_FINGERPRINT_DIM:
            raise ValueError(
                "MiniMol fingerprints must have width "
                f"{MINIMOL_FINGERPRINT_DIM}, got {model_inputs.shape[-1]}."
            )
        if self.projection is None:
            return model_inputs / self.input_scale.to(
                device=model_inputs.device, dtype=model_inputs.dtype
            )
        projection_inputs = model_inputs.to(
            device=self.projection.weight.device,
            dtype=self.projection.weight.dtype,
        )
        return self.activation(self.projection(projection_inputs))

    def calibrate_inputs(self, model_inputs: Tensor) -> None:
        """Set ``input_scale`` from the training fingerprints.

        A no-op with a projection, whose weights already set the scale of the
        latent space. Without one, the scale is the median pairwise Euclidean
        distance over at most ``_CALIBRATION_ROWS`` training rows, so a typical
        pair of training molecules sits at distance one. Raw fingerprint
        distances are otherwise many lengthscales long at initialization, the
        kernel matrix is numerically the identity, and a few thousand small
        Adam steps cannot move the lengthscales far enough to recover.

        Parameters
        ----------
        model_inputs : Tensor
            Training fingerprints, shaped ``(n, 512)``.
        """
        if self.projection is not None or model_inputs.shape[0] < 2:
            return
        rows = model_inputs.detach()
        if rows.shape[0] > _CALIBRATION_ROWS:
            generator = torch.Generator().manual_seed(0)
            chosen = torch.randperm(rows.shape[0], generator=generator)
            rows = rows[chosen[:_CALIBRATION_ROWS].to(rows.device)]
        median = torch.pdist(rows.to(dtype=torch.float64)).median()
        if bool(torch.isfinite(median)) and float(median) > 0.0:
            self.input_scale.fill_(float(median))

    def _build_fixed_encoder(
        self,
        *,
        batch_size: int,
        cache_size: int,
        feature_cache_path: str | Path | None,
        checkpoint_path: str | Path | None,
        cache_only: bool = False,
    ) -> MiniMolSmilesFixedEncoder:
        """Construct the fixed MiniMol encoder used by DKL."""
        return MiniMolSmilesFixedEncoder(
            batch_size=batch_size,
            cache_size=cache_size,
            feature_cache_path=feature_cache_path,
            checkpoint_path=checkpoint_path,
            cache_only=cache_only,
        )
