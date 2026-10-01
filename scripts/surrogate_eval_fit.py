"""Fit the AmpC surrogate and evaluate it, logging one W&B run per arm.

Steps 4 and 5 of ``SURROGATE_EVAL_PLAN.md``. The script builds the dataset and
surrogate from a config exactly as ``activelearning.main`` does, fits the surrogate on
the whole initial dataset while tracking per-epoch metrics on three labelled sets, then
scores the generated set once at the end. It writes to ``--output-dir``:

- ``surrogate_state.pt``: the fitted GP state (inducing points, variational
  distribution, kernel, noise, output standardization);
- ``train_random.csv`` and ``train_top.csv``: the two ``train`` evaluation sets
  (seeded random rows, and the rows with the highest target), written before the
  fit so a failed fit still leaves them;
- ``resolved_config.json``: the merged config that was run;
- ``fit_summary.json``: fit time, data size and the learned hyperparameters;
- ``eval_summary.json`` and ``eval/<set>.csv``: the final metrics and the per-molecule
  predictions, acquisition scores and rewards.

Evaluation runs in the same process as the fit because GIBBON's candidate set needs the
surrogate's training data, which reloading the saved state would have to re-encode.

The objective and every other setting are config overrides, so an ELBO and a PLL
arm differ only in one argument::

    python scripts/surrogate_eval_fit.py \\
        config/ampc/s3gfn_minimol_ampc_variational_single_fidelity.yaml \\
        surrogate.training_params.variational_objective=VariationalELBO \\
        acquisition.log_space=true acquisition.log_output=false \\
        --output-dir outputs/ampc/surrogate_eval/elbo

Run it on a whole GPU node through ``jobs/surrogate_eval_fit.sh``, never on the
login node. Pass ``--wandb-project`` to log to W&B (run with ``WANDB_MODE=offline``
on compute nodes and ``wandb sync`` afterwards). Only the per-epoch curves and the
figures are logged as charts; every once-per-run scalar (the final metrics, the
learned hyperparameters and the train time) is stored in the run config.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scripts.surrogate_eval_metrics import (
    prediction_metrics,
    predicted_vs_observed_figure,
    summary_stats,
    two_panel_histogram,
)

STATE_FILE = "surrogate_state.pt"
SUMMARY_FILE = "fit_summary.json"
CONFIG_FILE = "resolved_config.json"
EVAL_SUMMARY_FILE = "eval_summary.json"
TRAIN_RANDOM_FILE = "train_random.csv"
TRAIN_TOP_FILE = "train_top.csv"
EVAL_CSV_COLUMNS = ("SMILE", "y")

#: Evaluation sets scored after every training epoch. The generated set is excluded:
#: it is scored once at the end, since it also needs the acquisition.
PER_EPOCH_SETS = ("train_random", "train_top", "val_set")

#: Name of the generated set, evaluated only after training finishes.
GENERATED_SET = "gp_molformer_set"


@dataclass(frozen=True)
class EvalSet:
    """A labelled evaluation set whose features have already been encoded.

    Attributes
    ----------
    name : str
        Set name, used as the first segment of every metric key.
    smiles : tuple[str, ...]
        Molecule inputs, aligned with ``targets``.
    targets : np.ndarray
        Observed target per molecule.
    features : Any
        Encoded feature matrix (a ``torch.Tensor``), shaped ``(n, feature_dim)``.
        Typed loosely so this module does not import torch at module scope.
    """

    name: str
    smiles: tuple[str, ...]
    targets: np.ndarray
    features: Any


def select_train_eval_rows(
    targets: Sequence[float] | np.ndarray,
    n_random: int,
    n_top: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select the random and the highest-target rows used as ``train`` eval sets.

    The two selections are independent: a row can appear in both, which is
    expected to be rare (about ``n_random * n_top / len(targets)`` rows).

    Parameters
    ----------
    targets : Sequence[float] or np.ndarray
        Finite target value per training row.
    n_random : int
        Number of rows in the seeded random selection, without replacement.
    n_top : int
        Number of rows with the highest target. Ties at the cut-off are broken
        by row order, so the selection is deterministic.
    seed : int
        Seed of the random selection.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Row indices of the random selection (ascending) and of the top
        selection (highest target first).

    Raises
    ------
    ValueError
        If ``targets`` is not one-dimensional or is empty, or either count is
        negative.
    """
    values = np.asarray(targets, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("targets must be a non-empty one-dimensional sequence.")
    if n_random < 0 or n_top < 0:
        raise ValueError("n_random and n_top must be nonnegative.")
    random_rows = np.sort(
        np.random.default_rng(seed).choice(
            values.size,
            size=min(n_random, values.size),
            replace=False,
        )
    )
    top_rows = np.argsort(-values, kind="stable")[:n_top]
    return random_rows, top_rows


def write_eval_csv(
    path: Path,
    observations: Sequence[Any],
    rows: Sequence[int] | np.ndarray,
) -> None:
    """Write selected training observations as an evaluation CSV.

    Parameters
    ----------
    path : Path
        Destination CSV, written atomically.
    observations : Sequence[Any]
        Training observations whose ``x`` is a SMILES string and ``y`` a number.
    rows : Sequence[int] or np.ndarray
        Indices into ``observations`` to write, in order.

    Raises
    ------
    ValueError
        If a selected observation's input is not a string.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(EVAL_CSV_COLUMNS)
        for row in rows:
            observation = observations[int(row)]
            if not isinstance(observation.x, str):
                raise ValueError(
                    f"Observation {int(row)} has a non-string input: "
                    f"{type(observation.x).__name__}."
                )
            writer.writerow([observation.x, repr(float(observation.y))])
    os.replace(tmp_path, path)


def learned_hyperparameters(surrogate: Any) -> dict[str, float]:
    """Read the learned GP hyperparameters from a fitted variational surrogate.

    Noise and output scale are in standardized-target units; the output
    standardization is returned as ``y_mean`` and ``y_std`` so they can be mapped
    back to the original target scale. Reads the surrogate's private GP modules,
    since it exposes no public accessor for them, and returns an empty dict for a
    surrogate without that structure.

    Parameters
    ----------
    surrogate : Any
        Fitted ``VariationalGPSurrogate``.

    Returns
    -------
    dict[str, float]
        Hyperparameter name to value.
    """
    gp_model = getattr(surrogate, "_gp_model", None)
    likelihood = getattr(surrogate, "_likelihood", None)
    state = surrogate.get_state_dict()
    if gp_model is None or likelihood is None or state is None:
        return {}
    lengthscale = gp_model.covar_module.base_kernel.lengthscale.detach().flatten()
    noise = float(likelihood.noise.detach().mean())
    y_std = float(state["outcome_std"])
    return {
        "noise": noise,
        "noise_std_original_scale": noise**0.5 * y_std,
        "outputscale": float(gp_model.covar_module.outputscale.detach()),
        "mean_constant": float(gp_model.mean_module.constant.detach()),
        "lengthscale_min": float(lengthscale.min()),
        "lengthscale_median": float(lengthscale.median()),
        "lengthscale_max": float(lengthscale.max()),
        "y_mean": float(state["outcome_mean"]),
        "y_std": y_std,
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write ``payload`` as sorted, indented JSON, atomically."""
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp_path, path)


def load_labelled_csv(
    path: Path,
    *,
    smiles_column: str = "SMILES",
    label_column: str = "y",
    require_columns: Sequence[str] = (),
) -> tuple[list[str], np.ndarray, dict[str, list[str]]]:
    """Read a CSV of labelled molecules.

    Parameters
    ----------
    path : Path
        CSV to read.
    smiles_column : str, default="SMILES"
        Column holding the molecule input.
    label_column : str, default="y"
        Column holding the target.
    require_columns : Sequence[str], default=()
        Extra columns that must be present and are returned verbatim.

    Returns
    -------
    tuple[list[str], np.ndarray, dict[str, list[str]]]
        The SMILES, the targets (``nan`` where they do not parse) and the extras.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If a required column is missing.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")
    smiles: list[str] = []
    targets: list[float] = []
    extras: dict[str, list[str]] = {name: [] for name in require_columns}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [
            name
            for name in (smiles_column, label_column, *require_columns)
            if name not in (reader.fieldnames or [])
        ]
        if missing:
            raise ValueError(
                f"{path} is missing column(s) {missing}; it has {reader.fieldnames}."
            )
        for row in reader:
            smiles.append(row[smiles_column])
            try:
                targets.append(float(row[label_column]))
            except (TypeError, ValueError):
                targets.append(float("nan"))
            for name in require_columns:
                extras[name].append(row[name])
    return smiles, np.asarray(targets, dtype=np.float64), extras


def filter_generated_set(
    smiles: Sequence[str],
    targets: np.ndarray,
    passes_sa: Sequence[str],
) -> tuple[list[str], np.ndarray, dict[str, int]]:
    """Keep the generated molecules that both docked and pass the SA filter.

    About a tenth of the generated molecules fail to dock and carry a ``nan`` target;
    those failures are biased toward molecules the docking toolchain cannot build, so
    the surviving fraction is reported rather than assumed.

    Parameters
    ----------
    smiles : Sequence[str]
        Molecule inputs.
    targets : np.ndarray
        Docked targets, ``nan`` where docking failed.
    passes_sa : Sequence[str]
        The ``passes_sa`` column, truthy for molecules below the SA threshold.

    Returns
    -------
    tuple[list[str], np.ndarray, dict[str, int]]
        The kept SMILES, their targets, and the counts behind the filter.

    Raises
    ------
    ValueError
        If the three inputs differ in length.
    """
    values = np.asarray(targets, dtype=np.float64)
    if not (len(smiles) == values.size == len(passes_sa)):
        raise ValueError(
            f"Length mismatch: {len(smiles)} smiles, {values.size} targets, "
            f"{len(passes_sa)} sa flags."
        )
    sa_flags = np.asarray(
        [str(flag).strip() not in ("", "0", "False", "false") for flag in passes_sa]
    )
    docked = np.isfinite(values)
    keep = docked & sa_flags
    counts = {
        "n_total": int(values.size),
        "n_docked": int(docked.sum()),
        "n_sa_pass": int(sa_flags.sum()),
        "n_evaluated": int(keep.sum()),
    }
    kept = [smiles[index] for index in np.flatnonzero(keep)]
    return kept, values[keep], counts


def reward_columns(
    scores: np.ndarray, *, transform: str, beta: float
) -> dict[str, np.ndarray]:
    """Turn acquisition scores into the quantities the S3-GFN loss actually uses.

    Under the ``exponential`` transform the prepared score is the acquisition score
    itself and the relative trajectory-balance target is ``log R = beta * s``; ``R``
    only materializes inside the loss. ``R`` is returned too, but it reaches ``e**100``
    at the configured beta, so ``log_reward`` is the readable quantity.

    Parameters
    ----------
    scores : np.ndarray
        Acquisition scores, one per molecule.
    transform : str
        Reward transform name, as the sampler's ``reward_transform``.
    beta : float
        Inverse temperature of the reward.

    Returns
    -------
    dict[str, np.ndarray]
        ``reward_score`` (what the loss multiplies), ``log_reward`` (``beta * s``) and
        ``reward`` (``exp(log_reward)``, clipped so it stays finite).
    """
    from activelearning.sampler.reward_transform import apply_reward_transform

    prepared = np.asarray(
        apply_reward_transform(transform, [float(value) for value in scores]),
        dtype=np.float64,
    )
    log_reward = beta * prepared
    # exp overflows to inf above ~709; clip so the histogram and the summary stay
    # finite, and let the caller see how many were affected.
    reward = np.exp(np.clip(log_reward, None, 700.0))
    return {
        "reward_score": prepared,
        "log_reward": log_reward,
        "reward": reward,
    }


def build_run_logger(
    *,
    project: str | None,
    run_name: str,
    entity: str | None = None,
    tags: Sequence[str] | None = None,
    group: str | None = None,
) -> Any:
    """Build the logger the run uses, falling back to the console without a project.

    Parameters
    ----------
    project : str or None
        W&B project. ``None`` selects a ``ConsoleLogger`` instead, so the script runs
        end to end without W&B.
    run_name : str
        Run name.
    entity : str, optional
        W&B team or user. Without it wandb resolves the account from the environment.
    tags : Sequence[str], optional
        Run tags.
    group : str, optional
        Run group, for showing the arms of one study together.

    Returns
    -------
    Any
        A ``Logger``.
    """
    from activelearning.logger.logger import ConsoleLogger, WandbLogger

    if project is None:
        return ConsoleLogger(project_name="surrogate-eval", run_name=run_name)
    return WandbLogger(
        project_name=project,
        run_name=run_name,
        entity=entity,
        tags=tags,
        group=group,
    )


def log_scalars(logger: Any, metrics: Mapping[str, float]) -> None:
    """Buffer every scalar metric on the logger."""
    for key, value in metrics.items():
        logger.log_metric(key, value)


def final_metrics_config(
    metrics: Mapping[str, float], *, train_time_seconds: float
) -> dict[str, Any]:
    """Nest the once-per-run scalars into a record for the run configuration.

    A value logged once has no curve, so as a step metric it would only produce a
    single-point chart. These go next to the hyperparameters instead.

    Parameters
    ----------
    metrics : Mapping[str, float]
        Final scalars keyed by ``/``-separated paths, e.g. ``val_set/final/nll``.
    train_time_seconds : float
        Wall-clock time of the surrogate fit.

    Returns
    -------
    dict[str, Any]
        ``train_time_seconds`` plus the metrics nested by their path segments, e.g.
        ``{"val_set": {"final": {"nll": ...}}}``.
    """
    record: dict[str, Any] = {"train_time_seconds": float(train_time_seconds)}
    for key, value in metrics.items():
        *parents, leaf = key.split("/")
        node = record
        for parent in parents:
            node = node.setdefault(parent, {})
        node[leaf] = value
    return record


def log_figures(logger: Any, figures: Mapping[str, Any]) -> None:
    """Buffer every figure on the logger."""
    for key, figure in figures.items():
        logger.log_figure(key, figure)


def epoch_metrics(
    surrogate: Any,
    eval_sets: Sequence[EvalSet],
    *,
    num_data: int,
    chunk_size: int,
) -> dict[str, float]:
    """Score every per-epoch evaluation set and return its metrics, keyed for W&B.

    Parameters
    ----------
    surrogate : Any
        The surrogate being fitted, mid-training.
    eval_sets : Sequence[EvalSet]
        Sets to score.
    num_data : int
        Training-set size, so the objective's KL term matches the training loss.
    chunk_size : int
        Rows per posterior call.

    Returns
    -------
    dict[str, float]
        ``<set>/epoch/<metric>`` to value.
    """
    import torch

    metrics: dict[str, float] = {}
    for eval_set in eval_sets:
        prediction = surrogate.predict_encoded(
            eval_set.features, observation_noise=True, chunk_size=chunk_size
        )
        mean = prediction["mean"].numpy()
        std = prediction["std"].numpy()
        for name, value in prediction_metrics(eval_set.targets, mean, std).items():
            metrics[f"{eval_set.name}/epoch/{name}"] = value
        metrics[f"{eval_set.name}/epoch/objective_loss"] = surrogate.evaluate_objective(
            eval_set.features,
            torch.as_tensor(eval_set.targets),
            num_data=num_data,
            chunk_size=chunk_size,
        )
    return metrics


def build_train_eval_set(
    surrogate: Any,
    name: str,
    observations: Sequence[Any],
    rows: np.ndarray,
) -> EvalSet:
    """Build an eval set from rows of the surrogate's already-encoded training data.

    The row indices index ``observations``; they index the encoded matrix correctly
    only because ``fit()`` encodes the same list in the same order. That invariant is
    checked here, because if it ever broke the metrics would silently describe
    different molecules than the labels.

    Parameters
    ----------
    surrogate : Any
        Fitted or mid-fit surrogate holding the encoded training rows.
    name : str
        Set name.
    observations : Sequence[Any]
        The training observations the row indices refer to.
    rows : np.ndarray
        Row indices to take.

    Returns
    -------
    EvalSet
        The set, with features taken from the encoded training matrix.

    Raises
    ------
    RuntimeError
        If the encoded training matrix does not have one row per observation.
    """
    import torch

    encoded_rows = int(surrogate.get_encoded_train_data().shape[0])
    if encoded_rows != len(observations):
        raise RuntimeError(
            f"Encoded training matrix has {encoded_rows} rows but there are "
            f"{len(observations)} observations; row indices would be misaligned."
        )
    indices = torch.as_tensor(np.asarray(rows, dtype=np.int64))
    return EvalSet(
        name=name,
        smiles=tuple(str(observations[int(row)].x) for row in rows),
        targets=np.asarray(
            [float(observations[int(row)].y) for row in rows], dtype=np.float64
        ),
        features=surrogate.get_encoded_train_rows(indices),
    )


def encode_eval_set(
    surrogate: Any, name: str, smiles: Sequence[str], targets: np.ndarray
) -> EvalSet:
    """Encode a set of molecules once, for repeated scoring.

    Parameters
    ----------
    surrogate : Any
        Surrogate whose encoder is used. The encoder works before the GP is fitted.
    name : str
        Set name.
    smiles : Sequence[str]
        Molecule inputs.
    targets : np.ndarray
        Observed targets, aligned with ``smiles``.

    Returns
    -------
    EvalSet
        The encoded set.
    """
    from activelearning.utils.types import Candidate

    candidates = [Candidate(x=value, fidelity=None) for value in smiles]
    return EvalSet(
        name=name,
        smiles=tuple(smiles),
        targets=np.asarray(targets, dtype=np.float64),
        features=surrogate.encode_candidates(candidates),
    )


def make_epoch_callback(
    logger: Any,
    surrogate: Any,
    *,
    static_sets: Sequence[EvalSet],
    train_row_sets: Mapping[str, np.ndarray],
    observations: Sequence[Any],
    num_data: int,
    chunk_size: int,
) -> Callable[[int, float], None]:
    """Build the per-epoch callback that scores the labelled sets and logs them.

    The ``train`` sets cannot be encoded before the fit: the encoded training matrix
    only exists once ``fit()`` has run the encoder. They are therefore built on the
    first call, by which point it does.

    Parameters
    ----------
    logger : Any
        Logger to write to, one step per epoch.
    surrogate : Any
        The surrogate being fitted.
    static_sets : Sequence[EvalSet]
        Sets already encoded before the fit, such as ``val_set``.
    train_row_sets : Mapping[str, np.ndarray]
        Set name to row indices into ``observations``.
    observations : Sequence[Any]
        Training observations.
    num_data : int
        Training-set size, used to scale the objective's KL term.
    chunk_size : int
        Rows per posterior call.

    Returns
    -------
    Callable[[int, float], None]
        Callback matching ``VariationalGPSurrogate``'s epoch-callback signature.
    """
    resolved: list[EvalSet] = []

    def callback(epoch: int, mean_train_loss: float) -> None:
        if not resolved:
            resolved.extend(static_sets)
            for name, rows in train_row_sets.items():
                resolved.append(
                    build_train_eval_set(surrogate, name, observations, rows)
                )
        metrics = epoch_metrics(
            surrogate, resolved, num_data=num_data, chunk_size=chunk_size
        )
        metrics["train/epoch/minibatch_loss"] = mean_train_loss
        log_scalars(logger, metrics)
        logger.log_step(epoch)

    return callback


def update_acquisition(
    acquisition: Any, surrogate: Any, observations: Sequence[Any], *, seed: int
) -> dict[str, float]:
    """Update the acquisition against the fitted surrogate, with a seeded support.

    BoTorch draws GIBBON's max-value samples from the global torch RNG, which nothing
    in the library seeds, so the seed is set here to keep the two arms as close as the
    method allows. The candidate set may fall back to a stratified subset when the full
    training support does not fit in memory; whether it did is recorded, because two
    arms that scored against different supports are not comparable.

    Parameters
    ----------
    acquisition : Any
        The GIBBON acquisition.
    surrogate : Any
        The fitted surrogate.
    observations : Sequence[Any]
        Training observations backing the candidate set.
    seed : int
        Seed applied immediately before the update.

    Returns
    -------
    dict[str, float]
        ``run/acquisition/<metric>`` to value.

    Raises
    ------
    RuntimeError
        If the update left no BoTorch acquisition built, which would make every score
        the same constant and mimic a real finding.
    """
    from activelearning.utils.seeding import set_global_seed

    set_global_seed(seed)
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        acquisition.update(surrogate, observations)
    update_seconds = time.perf_counter() - started
    if getattr(acquisition, "_botorch_acqf", None) is None:
        raise RuntimeError(
            "The acquisition has no BoTorch acquisition after update(); every score "
            "would be 1.0."
        )
    for warning in caught:
        print(f"acquisition update warning: {warning.message}", flush=True)
    fallback_active = bool(getattr(acquisition, "_fallback_active", False))
    return {
        "run/acquisition/update_seconds": update_seconds,
        "run/acquisition/fallback_active": float(fallback_active),
    }


def score_acquisition(
    acquisition: Any, smiles: Sequence[str], *, chunk_size: int | None = None
) -> np.ndarray:
    """Score molecules with the acquisition.

    Parameters
    ----------
    acquisition : Any
        Updated acquisition.
    smiles : Sequence[str]
        Molecule inputs.
    chunk_size : int, optional
        Molecules per call. ``None`` lets the acquisition use its own chunking.

    Returns
    -------
    np.ndarray
        One score per molecule, in order.
    """
    from activelearning.utils.types import Candidate

    candidates = [Candidate(x=value, fidelity=None) for value in smiles]
    if chunk_size is None:
        return np.asarray(acquisition.score(candidates), dtype=np.float64)
    scores: list[float] = []
    for start in range(0, len(candidates), chunk_size):
        scores.extend(acquisition.score(candidates[start : start + chunk_size]))
    return np.asarray(scores, dtype=np.float64)


def evaluate_set(
    surrogate: Any, eval_set: EvalSet, *, chunk_size: int
) -> dict[str, np.ndarray]:
    """Predict mean and both standard deviations for one evaluation set.

    Parameters
    ----------
    surrogate : Any
        Fitted surrogate.
    eval_set : EvalSet
        Set to score.
    chunk_size : int
        Rows per posterior call.

    Returns
    -------
    dict[str, np.ndarray]
        ``mean``, ``std_total`` (with observation noise) and ``std_latent`` (without,
        which is what the acquisition consumes).
    """
    total = surrogate.predict_encoded(
        eval_set.features, observation_noise=True, chunk_size=chunk_size
    )
    latent = surrogate.predict_encoded(
        eval_set.features, observation_noise=False, chunk_size=chunk_size
    )
    return {
        "mean": total["mean"].numpy(),
        "std_total": total["std"].numpy(),
        "std_latent": latent["std"].numpy(),
    }


def write_per_molecule_csv(
    path: Path, eval_set: EvalSet, columns: Mapping[str, np.ndarray]
) -> None:
    """Write one row per molecule with its label and every computed column.

    Parameters
    ----------
    path : Path
        Destination CSV, written atomically.
    eval_set : EvalSet
        The set, supplying SMILES and targets.
    columns : Mapping[str, np.ndarray]
        Extra columns, each aligned with the set.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(columns)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["SMILES", "y", *names])
        for index, smiles in enumerate(eval_set.smiles):
            writer.writerow(
                [
                    smiles,
                    repr(float(eval_set.targets[index])),
                    *(repr(float(columns[name][index])) for name in names),
                ]
            )
    os.replace(tmp_path, path)


def final_set_outputs(
    eval_set: EvalSet,
    predictions: Mapping[str, np.ndarray],
    *,
    extra_distributions: Mapping[str, tuple[np.ndarray, str]] = {},
    max_figure_points: int = 5000,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Metrics and figures for one evaluation set after training.

    Parameters
    ----------
    eval_set : EvalSet
        The scored set.
    predictions : Mapping[str, np.ndarray]
        ``mean``, ``std_total`` and ``std_latent``.
    extra_distributions : Mapping[str, tuple[np.ndarray, str]], optional
        Additional quantities to summarize and histogram, as name to
        ``(values, axis label)``.
    max_figure_points : int, default=5000
        Points drawn in the predicted-vs-observed figure.

    Returns
    -------
    tuple[dict[str, float], dict[str, Any]]
        Scalar metrics and figures, both keyed for W&B.
    """
    name = eval_set.name
    metrics = {
        f"{name}/final/{key}": value
        for key, value in prediction_metrics(
            eval_set.targets, predictions["mean"], predictions["std_total"]
        ).items()
    }
    figures: dict[str, Any] = {}
    figure = predicted_vs_observed_figure(
        title=name,
        targets=eval_set.targets,
        mean=predictions["mean"],
        max_points=max_figure_points,
    )
    if figure is not None:
        figures[f"{name}/figures/predicted_vs_observed"] = figure

    distributions: dict[str, tuple[np.ndarray, str]] = {
        "std_total": (predictions["std_total"], "total predicted std"),
        "std_latent": (predictions["std_latent"], "latent predicted std"),
        **dict(extra_distributions),
    }
    for quantity, (values, xlabel) in distributions.items():
        for key, value in summary_stats(values).items():
            metrics[f"{name}/{quantity}/{key}"] = value
        figures[f"{name}/figures/{quantity}"] = two_panel_histogram(
            values, title=f"{name}: {quantity}", xlabel=xlabel
        )
    return metrics, figures


def _require_acquisition_settings(resolved: Mapping[str, Any]) -> None:
    """Fail fast unless the acquisition is GIBBON on the value scale.

    The base config ships ``log_output: true``, which returns log-scale scores. The
    reward beta is calibrated for value-scale information gain, so a forgotten override
    would silently produce a meaningless reward rather than an error.

    Parameters
    ----------
    resolved : Mapping[str, Any]
        The merged config.

    Raises
    ------
    SystemExit
        If the acquisition is not GIBBON, or is not value-scale without underflow.
    """
    acquisition = dict(resolved.get("acquisition", {}))
    problems = []
    if acquisition.get("type") != "QLowerBoundMaxValueEntropy":
        problems.append(f"type is {acquisition.get('type')!r}, want GIBBON")
    if acquisition.get("log_space") is not True:
        problems.append("acquisition.log_space must be true")
    if acquisition.get("log_output") is not False:
        problems.append("acquisition.log_output must be false")
    if problems:
        raise SystemExit(
            "Acquisition config is wrong for this evaluation: "
            + "; ".join(problems)
            + ". Add 'acquisition.type=QLowerBoundMaxValueEntropy "
            "acquisition.log_space=true acquisition.log_output=false' to the command."
        )


def _parse_args(argv: Sequence[str] | None) -> tuple[argparse.Namespace, list[str]]:
    """Split script options from config paths and OmegaConf overrides."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-train-random", type=int, default=100_000)
    parser.add_argument("--n-train-top", type=int, default=10_000)
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=42,
        help="Seed of the random `train` eval selection.",
    )
    parser.add_argument(
        "--val-csv",
        type=Path,
        default=Path("data/ampc_val_20k.csv"),
        help="Labelled validation subset of the 331k set.",
    )
    parser.add_argument(
        "--gp-molformer-csv",
        type=Path,
        default=Path("data/gpmolformer_prior_100k_docked.csv"),
        help="Docked GP-MoLFormer prior sample.",
    )
    parser.add_argument(
        "--eval-chunk-size",
        type=int,
        default=20_000,
        help="Rows per posterior call during evaluation.",
    )
    parser.add_argument(
        "--acquisition-seed",
        type=int,
        default=42,
        help="Seed applied before the acquisition update, for the max-value samples.",
    )
    parser.add_argument("--reward-transform", default="exponential")
    parser.add_argument("--reward-beta", type=float, default=100.0)
    parser.add_argument("--max-figure-points", type=int, default=5000)
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Fit only, skipping per-epoch tracking and the final evaluation.",
    )
    parser.add_argument(
        "--wandb-project",
        default=None,
        help="Log to this W&B project. Without it, logs to the console instead.",
    )
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument(
        "--wandb-tags", default=None, help="Comma-separated W&B run tags."
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Run name. Defaults to the output directory name.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing fitted state instead of refusing to run.",
    )
    args, config_args = parser.parse_known_args(argv)
    if args.n_train_random < 0 or args.n_train_top < 0:
        parser.error("--n-train-random and --n-train-top must be nonnegative.")
    if args.eval_chunk_size < 1:
        parser.error("--eval-chunk-size must be positive.")
    return args, config_args


def _run_configuration(
    args: argparse.Namespace,
    resolved: Mapping[str, Any],
    cfg: Any,
    sizes: Mapping[str, int],
) -> dict[str, Any]:
    """Assemble the flat hyperparameter record attached to the run."""
    surrogate_cfg = dict(resolved.get("surrogate", {}))
    training = dict(surrogate_cfg.get("training_params", {}))
    acquisition = dict(resolved.get("acquisition", {}))
    objective = training.get("variational_objective")
    return {
        "objective": objective,
        "objective_short": {
            "VariationalELBO": "elbo",
            "PredictiveLogLikelihood": "pll",
        }.get(str(objective), str(objective)),
        "epochs": training.get("epochs"),
        "lr": training.get("lr"),
        "batch_size": training.get("batch_size"),
        "num_inducing": surrogate_cfg.get("num_inducing"),
        "standardize_outputs": surrogate_cfg.get("standardize_outputs"),
        "target_fidelity": surrogate_cfg.get("target_fidelity"),
        "encoder_type": dict(surrogate_cfg.get("encoder", {})).get("type"),
        "feature_cache_path": dict(surrogate_cfg.get("encoder", {})).get(
            "feature_cache_path"
        ),
        "acquisition_type": acquisition.get("type"),
        "acquisition_log_space": acquisition.get("log_space"),
        "acquisition_log_output": acquisition.get("log_output"),
        "acquisition_num_mv_samples": acquisition.get("num_mv_samples"),
        "acquisition_score_chunk_size": acquisition.get("score_chunk_size"),
        "candidate_set_fallback_size": dict(
            acquisition.get("candidate_set_spec", {})
        ).get("fallback_size"),
        "reward_transform": args.reward_transform,
        "reward_beta": args.reward_beta,
        "runtime_seed": cfg.runtime.seed,
        "eval_seed": args.eval_seed,
        "acquisition_seed": args.acquisition_seed,
        "eval_chunk_size": args.eval_chunk_size,
        "val_csv": str(args.val_csv),
        "gp_molformer_csv": str(args.gp_molformer_csv),
        "gp_molformer_filter": "docked and passes_sa",
        "val_label_route": "pprop_hit_rate",
        "gp_molformer_label_route": "dock3_oracle",
        "output_dir": str(args.output_dir),
        **dict(sizes),
        "resolved_config": dict(resolved),
    }


def main(argv: Sequence[str] | None = None) -> None:
    """Fit the surrogate, evaluate it and log one run."""
    args, config_args = _parse_args(argv)
    if not any("=" not in arg for arg in config_args):
        raise SystemExit("At least one config file path must be provided.")
    out_dir: Path = args.output_dir
    if (out_dir / STATE_FILE).exists() and not args.overwrite:
        raise SystemExit(
            f"{out_dir / STATE_FILE} exists; pass --overwrite to replace it."
        )

    from omegaconf import OmegaConf

    from activelearning.main import process_arguments
    from activelearning.runtime import bind_runtime_context
    from activelearning.utils.seeding import set_global_seed
    from activelearning.utils.types import filter_finite_target_observations

    raw_cfg, cfg, config_paths, _ = process_arguments(config_args)
    resolved = OmegaConf.to_container(raw_cfg, resolve=True)
    assert isinstance(resolved, dict)
    resolved.pop("logger", None)
    resolved["config_paths"] = [str(path) for path in config_paths]

    if not args.skip_eval:
        _require_acquisition_settings(resolved)
        cache_path = dict(dict(resolved.get("surrogate", {})).get("encoder", {})).get(
            "feature_cache_path"
        )
        if cache_path is not None and not Path(cache_path).exists():
            raise SystemExit(
                f"The encoder feature cache {cache_path} does not exist. Encoding an "
                "evaluation set before the training data would publish that small set "
                "as the cache. Build the cache first, or unset feature_cache_path."
            )

    set_global_seed(cfg.runtime.seed)
    dataset = cfg.dataset.build()
    surrogate = cfg.surrogate.build()
    acquisition = None if args.skip_eval else cfg.acquisition.build()
    runtime_context = cfg.runtime.build(logger=None)
    components = [dataset, surrogate] + ([] if acquisition is None else [acquisition])
    bind_runtime_context(components, runtime_context)

    observations = list(
        filter_finite_target_observations(dataset.get_observations_iterable())
    )
    print(f"Loaded {len(observations)} finite observations.", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / CONFIG_FILE, resolved)

    targets = np.asarray([float(observation.y) for observation in observations])
    random_rows, top_rows = select_train_eval_rows(
        targets,
        args.n_train_random,
        args.n_train_top,
        args.eval_seed,
    )
    write_eval_csv(out_dir / TRAIN_RANDOM_FILE, observations, random_rows)
    write_eval_csv(out_dir / TRAIN_TOP_FILE, observations, top_rows)
    print(
        f"Wrote {len(random_rows)} random and {len(top_rows)} top rows "
        f"(top target cut-off {targets[top_rows[-1]] if len(top_rows) else 'n/a'}).",
        flush=True,
    )

    # Load both evaluation CSVs before the fit, so a bad path fails in seconds.
    val_smiles: list[str] = []
    val_targets = np.empty(0)
    generated_smiles: list[str] = []
    generated_targets = np.empty(0)
    generated_counts: dict[str, int] = {}
    if not args.skip_eval:
        val_smiles, val_targets, _ = load_labelled_csv(args.val_csv)
        raw_smiles, raw_targets, extras = load_labelled_csv(
            args.gp_molformer_csv, require_columns=("passes_sa",)
        )
        generated_smiles, generated_targets, generated_counts = filter_generated_set(
            raw_smiles, raw_targets, extras["passes_sa"]
        )
        print(
            f"Loaded {len(val_smiles)} val rows and {generated_counts['n_evaluated']} "
            f"of {generated_counts['n_total']} generated rows "
            f"(docked {generated_counts['n_docked']}, "
            f"SA-passing {generated_counts['n_sa_pass']}).",
            flush=True,
        )

    sizes = {
        "n_observations": len(observations),
        "n_train_random": int(len(random_rows)),
        "n_train_top": int(len(top_rows)),
        "n_val_set": len(val_smiles),
        **{f"gp_molformer_{key}": value for key, value in generated_counts.items()},
    }
    logger = build_run_logger(
        project=args.wandb_project,
        run_name=args.run_name or out_dir.name,
        entity=args.wandb_entity,
        tags=(
            [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
            if args.wandb_tags
            else None
        ),
        group=args.wandb_group,
    )
    try:
        logger.log_config(_run_configuration(args, resolved, cfg, sizes))

        static_sets: list[EvalSet] = []
        if not args.skip_eval:
            print(f"Encoding {len(val_smiles)} validation molecules.", flush=True)
            static_sets.append(
                encode_eval_set(surrogate, "val_set", val_smiles, val_targets)
            )
            surrogate.set_epoch_callback(
                make_epoch_callback(
                    logger,
                    surrogate,
                    static_sets=static_sets,
                    train_row_sets={
                        "train_random": random_rows,
                        "train_top": top_rows,
                    },
                    observations=observations,
                    num_data=len(observations),
                    chunk_size=args.eval_chunk_size,
                )
            )

        # Re-seed so the fit is identical whether or not evaluation is enabled: the
        # inducing points are drawn from the global RNG inside fit().
        set_global_seed(cfg.runtime.seed)
        print(f"Fitting surrogate on {len(observations)} observations.", flush=True)
        started = time.perf_counter()
        surrogate.fit(observations)
        fit_seconds = time.perf_counter() - started
        surrogate.set_epoch_callback(None)
        print(f"Surrogate fit in {fit_seconds:.0f} s.", flush=True)

        state = surrogate.get_state_dict()
        if state is None:
            raise SystemExit("The fitted surrogate has no state to save.")
        import torch

        torch.save(state, out_dir / STATE_FILE)
        print(f"Saved surrogate state to {out_dir / STATE_FILE}", flush=True)

        training_params = dict(resolved.get("surrogate", {})).get("training_params", {})
        summary = {
            "n_observations": len(observations),
            "fit_seconds": fit_seconds,
            "objective": training_params.get("variational_objective"),
            "num_inducing": dict(resolved.get("surrogate", {})).get("num_inducing"),
            "epochs": training_params.get("epochs"),
            "lr": training_params.get("lr"),
            "batch_size": training_params.get("batch_size"),
            "seed": cfg.runtime.seed,
            "eval_seed": args.eval_seed,
            "n_train_random": int(len(random_rows)),
            "n_train_top": int(len(top_rows)),
            "target_min": float(targets.min()),
            "target_max": float(targets.max()),
            "target_mean": float(targets.mean()),
            "hyperparameters": learned_hyperparameters(surrogate),
            "profiling": surrogate.get_fit_profiling(),
        }
        write_json(out_dir / SUMMARY_FILE, summary)
        print(f"Wrote {out_dir / SUMMARY_FILE}", flush=True)

        final_metrics: dict[str, float] = {
            "run/fit/seconds": fit_seconds,
            "run/fit/n_observations": float(len(observations)),
            **{
                f"run/hyperparameters/{name}": value
                for name, value in summary["hyperparameters"].items()
            },
        }
        final_figures: dict[str, Any] = {}

        if not args.skip_eval:
            print(f"Encoding {len(generated_smiles)} generated molecules.", flush=True)
            generated_set = encode_eval_set(
                surrogate, GENERATED_SET, generated_smiles, generated_targets
            )
            final_metrics.update(
                update_acquisition(
                    acquisition,
                    surrogate,
                    observations,
                    seed=args.acquisition_seed,
                )
            )
            print("Scoring the generated set with GIBBON.", flush=True)
            scores = score_acquisition(acquisition, generated_set.smiles)
            rewards = reward_columns(
                scores, transform=args.reward_transform, beta=args.reward_beta
            )
            final_metrics[f"{GENERATED_SET}/final/reward_overflow_count"] = float(
                np.sum(rewards["log_reward"] > 700.0)
            )
            for key, value in generated_counts.items():
                final_metrics[f"{GENERATED_SET}/final/{key}"] = float(value)

            all_sets = [
                build_train_eval_set(
                    surrogate, "train_random", observations, random_rows
                ),
                build_train_eval_set(surrogate, "train_top", observations, top_rows),
                static_sets[0],
                generated_set,
            ]
            for eval_set in all_sets:
                print(f"Evaluating {eval_set.name}.", flush=True)
                predictions = evaluate_set(
                    surrogate, eval_set, chunk_size=args.eval_chunk_size
                )
                extras: dict[str, tuple[np.ndarray, str]] = {}
                columns = dict(predictions)
                if eval_set.name == GENERATED_SET:
                    extras = {
                        "acquisition_score": (scores, "GIBBON information gain"),
                        "log_reward": (
                            rewards["log_reward"],
                            f"log R = {args.reward_beta:g} * score",
                        ),
                    }
                    columns["acquisition_score"] = scores
                    columns.update(rewards)
                metrics, figures = final_set_outputs(
                    eval_set,
                    predictions,
                    extra_distributions=extras,
                    max_figure_points=args.max_figure_points,
                )
                final_metrics.update(metrics)
                final_figures.update(figures)
                write_per_molecule_csv(
                    out_dir / "eval" / f"{eval_set.name}.csv", eval_set, columns
                )

        # The final scalars are single values, not curves: they go with the
        # hyperparameters, and only the figures are committed as a step.
        logger.log_config(
            final_metrics_config(final_metrics, train_time_seconds=fit_seconds)
        )
        log_figures(logger, final_figures)
        epochs = int(training_params.get("epochs") or 0)
        logger.log_step(epochs + 1)
        write_json(out_dir / EVAL_SUMMARY_FILE, final_metrics)
        print(f"Wrote {out_dir / EVAL_SUMMARY_FILE}", flush=True)
    finally:
        logger.end()
    print("Done.", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
