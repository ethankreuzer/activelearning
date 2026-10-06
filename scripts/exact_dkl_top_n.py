"""Fit an exact DKL GP on the top-n molecules and evaluate it.

One arm of the exact-DKL study: an exact GP (no inducing points, no
minibatching) over one trainable layer on the frozen MiniMol AmpC features,
fitted to the n highest-scoring molecules of the 10M set, then evaluated and
used to score three candidate sets.

Two things about this script are deliberate and easy to "simplify" wrongly:

**It never calls ``acquisition.score()`` or ``surrogate.predict()``.**
Both re-encode their inputs per chunk, and a chunk of an evaluation set is not
an exact ordered prefix of that set, so with a prefix-keyed MiniMol cache every
chunk would miss and fall back to live inference -- hours per set, and the
timing measurement this study exists to produce would be meaningless. It calls
``score_encoded`` and ``predict_encoded`` on features loaded once per set.

**The per-epoch series are only the training loss and the GP hyperparameters.**
The earlier variational study charted eval-set metrics per epoch; here every
eval-set metric is a one-off summary scalar instead. One-off scalars go through
``log_final_scalars`` -> ``log_summary``, never ``log_metric``: a single-point
scalar in W&B Charts buries the real curves.

Run through ``jobs/exact_dkl_top_n.sh``; the per-set caches must already exist
(``scripts/exact_dkl_prepare.py``).
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.stage_profiler import (  # noqa: E402
    CudaReader,
    StageProfiler,
    write_stage_csv,
)
from scripts.surrogate_eval_io import (  # noqa: E402
    EvalSet,
    build_run_logger,
    evaluate_encoded_set,
    load_labelled_csv,
    log_figures,
    log_final_scalars,
    log_scalars,
    parse_float_column,
    prediction_outputs,
    score_outputs,
    write_json,
    write_per_molecule_csv,
)

_logger = logging.getLogger("exact_dkl_top_n")

STATE_FILE = "surrogate_state.pt"
FIT_SUMMARY_FILE = "fit_summary.json"
EVAL_SUMMARY_FILE = "eval_summary.json"
CONFIG_FILE = "resolved_config.json"
STAGE_PROFILE_FILE = "stage_profile.csv"
LOSS_CURVE_FILE = "loss_curve.csv"

#: GIBBON, and the multi-fidelity MES. Nothing else is supported: the scoring
#: plan below hard-codes what each one's score scale means.
GIBBON_TYPE = "QLowerBoundMaxValueEntropy"
QMFMES_TYPE = "QMultiFidelityMaxValueEntropy"

#: The only charted series. Everything else is a one-off summary scalar.
PER_EPOCH_HYPERPARAMETERS = ("noise", "outputscale", "lengthscale_median")


@dataclass(frozen=True)
class StudySet:
    """One evaluation set and what is computed on it.

    Attributes
    ----------
    name : str
        Set name; also the cache and resolved-CSV basename.
    predict : bool
        Whether to compute posterior predictions and prediction metrics.
    score : bool
        Whether to compute acquisition scores.
    """

    name: str
    predict: bool
    score: bool


#: Order matters: the first predicted set pays the one-off exact-GP
#: ``prediction_strategy`` cache build (the n x n Cholesky), so its stage
#: seconds are not comparable with the later ones. Keeping the order fixed
#: means that cost always lands on ``predict_train_random``.
STUDY_SETS = (
    StudySet("train_random", predict=True, score=False),
    StudySet("train_top", predict=True, score=False),
    StudySet("val_set", predict=True, score=False),
    StudySet("gp_molformer_set", predict=True, score=True),
    StudySet("olivier_invitro", predict=False, score=True),
    StudySet("ampc_331k", predict=False, score=True),
)


@dataclass(frozen=True)
class Scoring:
    """One acquisition scoring pass.

    Attributes
    ----------
    name : str
        Key and column segment, such as ``gibbon_value``.
    label : str
        Human-readable axis label for the figures.
    config_update : dict
        Fields overridden on the acquisition config for this pass.
    log_floor : float or None
        Clamp used by the score figures, or ``None`` when the scores are
        already logarithms and may be negative.
    """

    name: str
    label: str
    config_update: dict[str, Any]
    log_floor: float | None


def scoring_plan(acquisition_type: str) -> tuple[Scoring, ...]:
    """Return the scoring passes for the configured acquisition.

    GIBBON is scored twice, on the value scale and on the log scale. On the
    value scale about 90% of scores underflow to exactly zero, which makes the
    331k distribution unreadable; the log scale ranks every molecule but is not
    comparable with the earlier runs. Both are cheap relative to the fit, so
    both are recorded.

    Parameters
    ----------
    acquisition_type : str
        The configured acquisition's ``type``.

    Returns
    -------
    tuple[Scoring, ...]
        One entry per scoring pass.

    Raises
    ------
    SystemExit
        If the acquisition is not one this study supports.
    """
    if acquisition_type == GIBBON_TYPE:
        return (
            Scoring(
                name="gibbon_value",
                label="GIBBON information gain",
                config_update={"log_space": True, "log_output": False},
                log_floor=1e-12,
            ),
            Scoring(
                name="gibbon_log",
                label="log GIBBON information gain",
                config_update={"log_space": True, "log_output": True},
                log_floor=None,
            ),
        )
    if acquisition_type == QMFMES_TYPE:
        return (
            Scoring(
                name="qmfmes",
                label="qMFMES information gain",
                config_update={},
                log_floor=1e-12,
            ),
        )
    raise SystemExit(
        f"Unsupported acquisition type {acquisition_type!r}. This study runs "
        f"{GIBBON_TYPE} or {QMFMES_TYPE}; add "
        f"'config/ampc/overrides/exact_dkl_qmfmes.yaml' for the latter."
    )


def exact_dkl_hyperparameters(surrogate: Any) -> dict[str, float]:
    """Read the learned hyperparameters of an exact DKL GP.

    The variational study's reader looks for ``_gp_model`` and a variational
    strategy and returns nothing here. The structure in this study is
    ``model.covar_module = EncoderKernel(base_kernel=ScaleKernel(MaternKernel))``.

    Parameters
    ----------
    surrogate : Any
        A fitted ``ExactDKLSurrogate``.

    Returns
    -------
    dict[str, float]
        The learned noise and output scale, the lengthscale spread, the mean
        constant, and the outcome transform's standardization constants. Empty
        if the surrogate has not been fitted.
    """
    if not surrogate.is_fitted():
        return {}
    model = surrogate.get_model()
    scale_kernel = model.covar_module.base_kernel
    matern = scale_kernel.base_kernel
    lengthscale = matern.lengthscale.detach().reshape(-1)
    noise = float(model.likelihood.noise.detach().mean())
    outputscale = float(scale_kernel.outputscale.detach())
    transform = getattr(model, "outcome_transform", None)
    y_mean = 0.0 if transform is None else float(transform.means.reshape(-1)[0])
    y_std = 1.0 if transform is None else float(transform.stdvs.reshape(-1)[0])
    return {
        "noise": noise,
        # The targets are standardized, so the interpretable noise is on the
        # original scale.
        "noise_std_original_scale": float(noise**0.5 * y_std),
        "outputscale": outputscale,
        "prior_std_original_scale": float(outputscale**0.5 * y_std),
        "mean_constant": float(model.mean_module.constant.detach().reshape(-1)[0]),
        "lengthscale_min": float(lengthscale.min()),
        "lengthscale_median": float(lengthscale.median()),
        "lengthscale_max": float(lengthscale.max()),
        "y_mean": y_mean,
        "y_std": y_std,
    }


def make_epoch_callback(
    logger: Any, surrogate: Any, loss_rows: list[dict[str, float]]
) -> Any:
    """Build the per-epoch callback: the training loss and the hyperparameters.

    Parameters
    ----------
    logger : Any
        Receives the charted series.
    surrogate : Any
        The surrogate being fitted, read for its hyperparameters.
    loss_rows : list[dict[str, float]]
        Appended to, one row per epoch, for ``loss_curve.csv``.

    Returns
    -------
    Any
        A callable taking ``(epoch, loss)``.
    """

    def callback(epoch: int, loss: float) -> None:
        metrics = {"train/epoch/loss": float(loss)}
        hyperparameters = exact_dkl_hyperparameters(surrogate)
        for name in PER_EPOCH_HYPERPARAMETERS:
            if name in hyperparameters:
                metrics[f"train/epoch/{name}"] = hyperparameters[name]
        loss_rows.append({"epoch": float(epoch), **metrics})
        log_scalars(logger, metrics)
        logger.log_step(epoch)

    return callback


def load_resolved_set(
    cache_dir: Path,
    name: str,
    *,
    encoder: Any,
    device: Any,
    limit: int | None = None,
) -> EvalSet:
    """Load one set's rows and its cached features.

    Reads the resolved CSV the prep job wrote next to the cache, so the
    requested order is exactly the order the cache was published for.

    Parameters
    ----------
    cache_dir : Path
        Directory holding ``<name>.csv`` and ``<name>.npy``.
    name : str
        Set name.
    encoder : Any
        A cache-only fixed encoder pointed at ``<name>.npy``.
    device : Any
        Device the features are returned on.
    limit : int, optional
        Keep only the first ``limit`` rows. For the smoke run only; it changes
        the cache request, so the encoder must not be cache-only when it is
        used.

    Returns
    -------
    EvalSet
        The set, with features already encoded.
    """
    smiles, targets, extras = load_labelled_csv(
        cache_dir / f"{name}.csv",
        smiles_column="SMILES",
        label_column="y",
        require_columns=(),
    )
    weights = None
    weight_path = cache_dir / f"{name}.csv"
    with weight_path.open(newline="") as handle:
        has_weight = "weight" in (csv.DictReader(handle).fieldnames or [])
    if has_weight:
        _, _, weight_extras = load_labelled_csv(
            weight_path,
            smiles_column="SMILES",
            label_column="y",
            require_columns=("weight",),
        )
        weights = parse_float_column(weight_extras["weight"])
    features = encoder.encode(smiles, device=device)
    if limit is not None and limit > 0:
        smiles = smiles[:limit]
        targets = targets[:limit]
        features = features[:limit]
        weights = None if weights is None else weights[:limit]
    del extras
    return EvalSet(
        name=name,
        smiles=tuple(smiles),
        targets=targets,
        features=features,
        weights=weights,
    )


def build_cache_only_encoder(
    encoder_config: Mapping[str, Any], cache_path: Path
) -> Any:
    """Build a cache-only MiniMol encoder for one set's cache.

    Parameters
    ----------
    encoder_config : Mapping[str, Any]
        The resolved config's ``surrogate.encoder`` block, so the encoder
        identity matches the one the cache was published with.
    cache_path : Path
        The set's ``.npy`` cache.

    Returns
    -------
    Any
        A ``MiniMolAmpcSmilesFixedEncoder`` that raises rather than running live
        inference.
    """
    from activelearning.applications.molecules.minimol_ampc_encoder import (
        MiniMolAmpcSmilesFixedEncoder,
    )

    return MiniMolAmpcSmilesFixedEncoder(
        checkpoint_path=encoder_config["checkpoint_path"],
        package_path=encoder_config.get("package_path"),
        device=encoder_config.get("device", "cpu"),
        batch_size=int(encoder_config.get("batch_size", 64)),
        cache_size=0,
        feature_cache_path=cache_path,
        cache_only=True,
    )


def check_required_paths(
    cache_dir: Path, sets: Sequence[StudySet], training_cache: Path
) -> None:
    """Fail before any work if a cache or resolved CSV is missing.

    Reports every missing path at once: finding them one job at a time wastes a
    queue slot each time.

    Parameters
    ----------
    cache_dir : Path
        Directory the prep job wrote.
    sets : Sequence[StudySet]
        The evaluation sets this arm needs.
    training_cache : Path
        The training set's cache, from the surrogate config.

    Raises
    ------
    SystemExit
        If anything is missing.
    """
    missing: list[Path] = []
    for study_set in sets:
        for suffix in (".csv", ".npy", ".npy.json"):
            path = cache_dir / f"{study_set.name}{suffix}"
            if not path.exists():
                missing.append(path)
    for path in (training_cache, Path(str(training_cache) + ".json")):
        if not path.exists():
            missing.append(path)
    if missing:
        formatted = "\n  ".join(str(path) for path in missing)
        raise SystemExit(
            "These prepared artifacts are missing:\n  "
            + formatted
            + "\nRun 'sbatch jobs/exact_dkl_prepare.sh' first."
        )


def check_training_cache_matches(training_cache: Path, n_rows: int) -> None:
    """Fail if the training cache does not hold exactly ``n_rows`` rows.

    Catches the arm-crossing mistake -- the n=3000 CSV with the n=2000 cache --
    before the fit, where it would otherwise surface as a confusing cache miss.

    Parameters
    ----------
    training_cache : Path
        The ``.npy`` named in the surrogate config.
    n_rows : int
        Rows in the configured training CSV.

    Raises
    ------
    SystemExit
        On a mismatch.
    """
    import json

    manifest = json.loads(Path(str(training_cache) + ".json").read_text())
    row_count = manifest.get("row_count")
    if row_count != n_rows:
        raise SystemExit(
            f"The training cache {training_cache} holds {row_count} rows but the "
            f"configured training set has {n_rows}. Point "
            "surrogate.encoder.feature_cache_path at the cache built for this set."
        )


def check_acquisition_guards(acquisition: Any, scoring: Scoring) -> dict[str, float]:
    """Verify one updated acquisition and return its support sizes.

    Parameters
    ----------
    acquisition : Any
        An acquisition that has just been updated.
    scoring : Scoring
        The pass it belongs to, for the metric keys.

    Returns
    -------
    dict[str, float]
        The candidate-set and BoTorch support sizes.

    Raises
    ------
    RuntimeError
        If the BoTorch acquisition was never built, which would make every
        score the uniform fallback and mimic a real result.
    SystemExit
        If the candidate set fell back to a subset, which changes the support
        and makes the arm incomparable.
    """
    if getattr(acquisition, "_botorch_acqf", None) is None:
        raise RuntimeError(
            f"{scoring.name}: the BoTorch acquisition was not built, so every score "
            "would be the uniform 1.0 fallback."
        )
    if getattr(acquisition, "_fallback_active", False):
        raise SystemExit(
            f"{scoring.name}: the candidate set fell back to a subset. At n <= 3000 "
            "this cannot happen legitimately; the support would differ from the "
            "other arms."
        )
    metrics: dict[str, float] = {f"run/acquisition/{scoring.name}/fallback_active": 0.0}
    spec = getattr(acquisition, "_candidate_set_spec", None)
    observation_count = getattr(spec, "observation_count", None)
    if observation_count is not None:
        metrics[f"run/acquisition/{scoring.name}/candidate_set_size"] = float(
            observation_count
        )
    candidate_set = getattr(acquisition._botorch_acqf, "candidate_set", None)
    if candidate_set is not None:
        # BoTorch concatenates the model's train_inputs onto the candidate set,
        # so this is about twice the candidate-set size. Both are reported, or
        # the number reads wrong.
        metrics[f"run/acquisition/{scoring.name}/acqf_support_size"] = float(
            candidate_set.shape[0]
        )
    return metrics


def _parse_args(argv: Sequence[str] | None) -> tuple[argparse.Namespace, list[str]]:
    """Split script options from config paths and OmegaConf overrides."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("cache/ampc/exact_dkl_top_n")
    )
    parser.add_argument("--eval-chunk-size", type=int, default=5_000)
    parser.add_argument("--score-chunk-size", type=int, default=5_000)
    parser.add_argument("--acquisition-seed", type=int, default=42)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--score-limit",
        type=int,
        default=0,
        help=(
            "Keep only the first N rows of each scored set, for a smoke run that "
            "measures per-candidate cost. Requires --allow-live-encoding."
        ),
    )
    parser.add_argument(
        "--allow-live-encoding",
        action="store_true",
        help=(
            "Permit the encoder to fall back to live MiniMol inference. Only for a "
            "smoke run; it invalidates the timing measurement."
        ),
    )
    parser.add_argument(
        "--skip-scoring",
        action="store_true",
        help=(
            "Fit and predict only, skipping every acquisition update and scoring "
            "pass. For a fit-scalability sweep: per-candidate scoring cost grows "
            "with n, so at large n scoring dominates the runtime being measured. "
            "Sets that are only scored are freed without writing a CSV."
        ),
    )
    args, leftover = parser.parse_known_args(list(argv) if argv is not None else None)
    if args.eval_chunk_size < 1 or args.score_chunk_size < 1:
        raise SystemExit("--eval-chunk-size and --score-chunk-size must be positive.")
    if args.score_limit < 0:
        raise SystemExit("--score-limit must be nonnegative.")
    if args.score_limit and not args.allow_live_encoding:
        raise SystemExit(
            "--score-limit truncates each set, so the cache no longer matches the "
            "request. Pass --allow-live-encoding to acknowledge that the timings "
            "from this run are not usable."
        )
    args.tags = (
        None
        if args.wandb_tags is None
        else [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
    )
    return args, leftover


def run_configuration(
    args: argparse.Namespace,
    resolved: Mapping[str, Any],
    *,
    n_train: int,
    acquisition_type: str,
    scorings: Sequence[Scoring],
    set_sizes: Mapping[str, int],
) -> dict[str, Any]:
    """Build the flat hyperparameter record for the run config.

    Includes the confounders the cost extrapolation needs: without the
    allocation, the precision and the chunk sizes, the measured seconds and
    peaks cannot be compared across runs.
    """
    surrogate = dict(resolved.get("surrogate", {}))
    encoder = dict(surrogate.get("encoder", {}))
    training = dict(surrogate.get("training_params", {}))
    runtime = dict(resolved.get("runtime", {}))
    acquisition = dict(resolved.get("acquisition", {}))
    return {
        "n_train": n_train,
        "acquisition_type": acquisition_type,
        "scorings": ",".join(scoring.name for scoring in scorings),
        "surrogate_type": surrogate.get("type"),
        "encoder_type": encoder.get("type"),
        "latent_dim": encoder.get("latent_dim"),
        "activation": encoder.get("activation"),
        "cache_only": encoder.get("cache_only"),
        "feature_cache_path": encoder.get("feature_cache_path"),
        "standardize_outputs": surrogate.get("standardize_outputs"),
        "epochs": training.get("epochs"),
        "lr": training.get("lr"),
        "num_mv_samples": acquisition.get("num_mv_samples"),
        "num_fantasies": acquisition.get("num_fantasies"),
        "num_y_samples": acquisition.get("num_y_samples"),
        "candidate_set_type": dict(acquisition.get("candidate_set_spec", {})).get(
            "type"
        ),
        "runtime_device": runtime.get("device"),
        "runtime_precision": runtime.get("precision"),
        "runtime_seed": runtime.get("seed"),
        "acquisition_seed": args.acquisition_seed,
        "eval_chunk_size": args.eval_chunk_size,
        "score_chunk_size": args.score_chunk_size,
        "score_limit": args.score_limit,
        "output_dir": str(args.output_dir),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_mem_mb": os.environ.get("SLURM_MEM_PER_NODE"),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        **{f"n_{name}": size for name, size in set_sizes.items()},
        "resolved_config": resolved,
    }


def main(argv: Sequence[str] | None = None) -> None:
    """Fit one arm and write its metrics, scores and stage profile."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    import torch

    from activelearning.main import process_arguments
    from activelearning.runtime import bind_runtime_context
    from activelearning.utils.seeding import set_global_seed
    from activelearning.utils.types import filter_finite_target_observations
    from omegaconf import OmegaConf

    args, config_args = _parse_args(argv)
    if not [token for token in config_args if "=" not in token]:
        raise SystemExit("At least one config file path must be provided.")

    out_dir: Path = args.output_dir
    if (out_dir / STATE_FILE).exists() and not args.overwrite:
        raise SystemExit(
            f"{out_dir / STATE_FILE} already exists; pass --overwrite to replace it."
        )

    raw_cfg, cfg, config_paths, _ = process_arguments(config_args)
    resolved = OmegaConf.to_container(raw_cfg, resolve=True)
    resolved.pop("logger", None)
    resolved.pop("run_writer", None)
    resolved["config_paths"] = [str(path) for path in config_paths]

    surrogate_cfg = dict(resolved.get("surrogate", {}))
    encoder_cfg = dict(surrogate_cfg.get("encoder", {}))
    acquisition_cfg = dict(resolved.get("acquisition", {}))
    acquisition_type = str(acquisition_cfg.get("type"))

    if surrogate_cfg.get("type") != "ExactDKLSurrogate":
        raise SystemExit(
            f"This study fits ExactDKLSurrogate; the config has "
            f"{surrogate_cfg.get('type')!r}."
        )
    if encoder_cfg.get("type") != "MiniMolAmpcSmilesEncoder":
        raise SystemExit(
            "This study uses the MiniMolAmpcSmilesEncoder latent encoder; the config "
            f"has {encoder_cfg.get('type')!r}."
        )
    if encoder_cfg.get("latent_dim") != 256 or encoder_cfg.get("activation") != "gelu":
        _logger.warning(
            "The study's arms use latent_dim=256 and activation='gelu'; this run has "
            "latent_dim=%s activation=%r.",
            encoder_cfg.get("latent_dim"),
            encoder_cfg.get("activation"),
        )
    plan = scoring_plan(acquisition_type)
    training_cache = Path(str(encoder_cfg.get("feature_cache_path")))
    check_required_paths(args.cache_dir, STUDY_SETS, training_cache)

    profiler = StageProfiler(
        cuda_reader=CudaReader() if torch.cuda.is_available() else None
    )
    final_metrics: dict[str, float] = {}
    final_figures: dict[str, Any] = {}
    logger: Any | None = None

    def flush_stage_profile() -> None:
        """Persist the stage profile after every stage, so a kill keeps it."""
        write_stage_csv(out_dir / STAGE_PROFILE_FILE, profiler.rows())

    try:
        with profiler.timing_only("run_total"):
            set_global_seed(cfg.runtime.seed)
            dataset = cfg.dataset.build()
            surrogate = cfg.surrogate.build()
            acquisitions = {
                scoring.name: cfg.acquisition.model_copy(
                    update=scoring.config_update
                ).build()
                for scoring in plan
            }
            runtime_context = cfg.runtime.build(logger=None)
            bind_runtime_context(
                [dataset, surrogate, *acquisitions.values()], runtime_context
            )

            observations = list(
                filter_finite_target_observations(dataset.get_observations_iterable())
            )
            n_train = len(observations)
            if n_train == 0:
                raise SystemExit("The configured training set is empty.")
            check_training_cache_matches(training_cache, n_train)
            for name, acquisition in acquisitions.items():
                if not hasattr(acquisition, "score_encoded"):
                    raise SystemExit(
                        f"{name}: the acquisition has no score_encoded(). This "
                        "checkout predates it; scoring through score() would run "
                        "live MiniMol inference on every chunk."
                    )
            if not hasattr(surrogate, "predict_encoded"):
                raise SystemExit(
                    "The surrogate has no predict_encoded(). This checkout predates "
                    "it; predicting through predict() would re-encode every chunk."
                )

            out_dir.mkdir(parents=True, exist_ok=True)
            write_json(out_dir / CONFIG_FILE, resolved)

            eval_sets: dict[str, EvalSet] = {}
            with profiler.timing_only("features_total"):
                for study_set in STUDY_SETS:
                    with profiler.stage(f"features_{study_set.name}"):
                        encoder = build_cache_only_encoder(
                            encoder_cfg, args.cache_dir / f"{study_set.name}.npy"
                        )
                        if args.allow_live_encoding:
                            encoder.cache_only = False
                        eval_sets[study_set.name] = load_resolved_set(
                            args.cache_dir,
                            study_set.name,
                            encoder=encoder,
                            device=torch.device("cpu"),
                            limit=args.score_limit or None,
                        )
                    flush_stage_profile()
            set_sizes = {
                name: len(eval_set.smiles) for name, eval_set in eval_sets.items()
            }

            logger = build_run_logger(
                project=args.wandb_project,
                run_name=args.run_name or out_dir.name,
                entity=args.wandb_entity,
                tags=args.tags,
                group=args.wandb_group,
                console_project="exact-dkl-top-n",
            )
            logger.log_config(
                run_configuration(
                    args,
                    resolved,
                    n_train=n_train,
                    acquisition_type=acquisition_type,
                    scorings=plan,
                    set_sizes=set_sizes,
                )
            )

            loss_rows: list[dict[str, float]] = []
            surrogate.set_epoch_callback(
                make_epoch_callback(logger, surrogate, loss_rows)
            )
            set_global_seed(cfg.runtime.seed)
            fit_started = time.perf_counter()
            with profiler.stage("gp_fit"):
                surrogate.fit(observations)
            fit_seconds = time.perf_counter() - fit_started
            surrogate.set_epoch_callback(None)
            flush_stage_profile()

            torch.save(surrogate.get_state_dict(), out_dir / STATE_FILE)
            _write_loss_curve(out_dir / LOSS_CURVE_FILE, loss_rows)

            hyperparameters = exact_dkl_hyperparameters(surrogate)
            final_metrics["run/fit/seconds"] = fit_seconds
            final_metrics["run/fit/n_train"] = float(n_train)
            final_metrics["run/fit/epochs"] = float(
                cfg.surrogate.training_params.epochs
            )
            if loss_rows:
                final_metrics["run/fit/final_loss"] = loss_rows[-1]["train/epoch/loss"]
            for key, value in hyperparameters.items():
                if key in ("y_mean", "y_std"):
                    continue
                final_metrics[f"run/hyperparameters/{key}"] = value
            logger.log_config(
                {
                    "y_mean": hyperparameters.get("y_mean"),
                    "y_std": hyperparameters.get("y_std"),
                }
            )
            write_json(
                out_dir / FIT_SUMMARY_FILE,
                {
                    "fit_seconds": fit_seconds,
                    "n_train": n_train,
                    "epochs": cfg.surrogate.training_params.epochs,
                    "lr": cfg.surrogate.training_params.lr,
                    "acquisition_type": acquisition_type,
                    "hyperparameters": hyperparameters,
                    "profiling": surrogate.get_fit_profiling(),
                    "stages": profiler.rows(),
                    "set_sizes": set_sizes,
                },
            )

            for scoring in [] if args.skip_scoring else plan:
                acquisition = acquisitions[scoring.name]
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    with profiler.stage(f"acquisition_update_{scoring.name}"):
                        # The same seed before every update, because the
                        # max-value samples are drawn when the BoTorch
                        # acquisition is constructed. Without this the two
                        # GIBBON columns would not be comparable.
                        set_global_seed(args.acquisition_seed)
                        acquisition.update(surrogate, observations)
                for entry in caught:
                    _logger.warning("%s: %s", scoring.name, entry.message)
                final_metrics.update(check_acquisition_guards(acquisition, scoring))
                flush_stage_profile()

            if not args.skip_scoring:
                final_metrics.update(_check_gibbon_max_values(acquisitions, plan))

            for study_set in STUDY_SETS:
                eval_set = eval_sets[study_set.name]
                if args.skip_scoring and not study_set.predict:
                    # Nothing left to do for a scored-only set: free its features
                    # so the measured peak stays the peak of one set at a time.
                    eval_sets[study_set.name] = EvalSet(
                        name=eval_set.name,
                        smiles=eval_set.smiles,
                        targets=eval_set.targets,
                        features=None,
                        weights=eval_set.weights,
                    )
                    del eval_set
                    continue
                columns: dict[str, np.ndarray] = {}
                if study_set.predict:
                    with profiler.stage(f"predict_{study_set.name}"):
                        predictions = evaluate_encoded_set(
                            surrogate, eval_set, chunk_size=args.eval_chunk_size
                        )
                    flush_stage_profile()
                    columns.update(predictions)
                    metrics, figures = prediction_outputs(eval_set, predictions)
                    final_metrics.update(metrics)
                    final_figures.update(figures)
                if study_set.score and not args.skip_scoring:
                    for scoring in plan:
                        with profiler.stage(f"score_{scoring.name}_{study_set.name}"):
                            scores = np.asarray(
                                acquisitions[scoring.name].score_encoded(
                                    eval_set.features,
                                    chunk_size=args.score_chunk_size,
                                ),
                                dtype=np.float64,
                            )
                        flush_stage_profile()
                        columns[f"score_{scoring.name}"] = scores
                        metrics, figures = score_outputs(
                            eval_set,
                            scores,
                            scoring=scoring.name,
                            label=scoring.label,
                            log_floor=scoring.log_floor,
                        )
                        final_metrics.update(metrics)
                        final_figures.update(figures)
                write_per_molecule_csv(
                    out_dir / "eval" / f"{study_set.name}.csv", eval_set, columns
                )
                # Freeing before the next set keeps the measured peak the peak
                # of one set at a time, not of all six held at once.
                eval_sets[study_set.name] = EvalSet(
                    name=eval_set.name,
                    smiles=eval_set.smiles,
                    targets=eval_set.targets,
                    features=None,
                    weights=eval_set.weights,
                )
                del eval_set, columns

        final_metrics.update(profiler.metrics())
        final_metrics.update(profiler.run_peaks())
        flush_stage_profile()
        write_json(out_dir / EVAL_SUMMARY_FILE, final_metrics)
        if logger is not None:
            log_final_scalars(logger, final_metrics)
            log_figures(logger, final_figures)
            logger.log_step(int(cfg.surrogate.training_params.epochs))
    finally:
        for figure in final_figures.values():
            import matplotlib.pyplot as plt

            plt.close(figure)
        if logger is not None:
            logger.end()


def _write_loss_curve(path: Path, rows: Sequence[Mapping[str, float]]) -> None:
    """Write the per-epoch loss and hyperparameters, atomically."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    os.replace(tmp_path, path)


def _check_gibbon_max_values(
    acquisitions: Mapping[str, Any], plan: Sequence[Scoring]
) -> dict[str, float]:
    """Verify the two GIBBON passes drew identical max-value samples.

    The value-scale and log-scale columns are only comparable if both
    acquisitions sampled the same posterior maxima. Reseeding before each
    update is meant to guarantee that; this checks it rather than trusting it.

    Parameters
    ----------
    acquisitions : Mapping[str, Any]
        The updated acquisitions, keyed by scoring name.
    plan : Sequence[Scoring]
        The scoring passes.

    Returns
    -------
    dict[str, float]
        The comparison result, empty unless both GIBBON passes are present.

    Raises
    ------
    SystemExit
        If the samples differ.
    """
    names = [scoring.name for scoring in plan]
    if not {"gibbon_value", "gibbon_log"} <= set(names):
        return {}
    reference = getattr(
        acquisitions["gibbon_value"]._botorch_acqf, "posterior_max_values", None
    )
    other = getattr(
        acquisitions["gibbon_log"]._botorch_acqf, "posterior_max_values", None
    )
    if reference is None or other is None:
        return {}
    abs_diff = float((reference.double() - other.double()).abs().max())
    if abs_diff != 0.0:
        raise SystemExit(
            "The two GIBBON passes drew different max-value samples (max abs diff "
            f"{abs_diff:g}), so the value-scale and log-scale columns would not be "
            "comparable."
        )
    return {
        "run/acquisition/gibbon_max_values_identical": 1.0,
        "run/acquisition/gibbon_max_value_abs_diff_max": abs_diff,
    }


if __name__ == "__main__":
    main()
