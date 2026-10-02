"""Fit the AmpC surrogate and evaluate it, logging one W&B run per arm.

Steps 4 and 5 of ``SURROGATE_EVAL_PLAN.md``. The script builds the dataset and
surrogate from a config exactly as ``activelearning.main`` does, fits the surrogate on
the whole initial dataset while tracking per-epoch metrics on four labelled sets
(two ``train`` sets, the validation set and the generated set), then scores the
generated molecules with the acquisition once at the end. It writes to ``--output-dir``:

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

``--init-state`` continues training from a saved ``surrogate_state.pt`` instead of
starting from scratch (step 7 of the plan: an ELBO fit followed by a PLL phase).
``--trainable variance`` then trains only the parameters that leave the predictive
mean untouched, and ``--epoch-offset`` shifts the logged epochs so the continued run
lines up after the run it starts from.

Run it on a whole GPU node through ``jobs/surrogate_eval_fit.sh``, never on the
login node. Pass ``--wandb-project`` to log to W&B (run with ``WANDB_MODE=offline``
on compute nodes and ``wandb sync`` afterwards). Every scalar is a logged metric: the per-epoch ones at their epoch, and
the once-per-run ones (the final metrics, the learned hyperparameters and the train
time) at the step after the last epoch, so they also land in the run summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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
    error_by_std_figure,
    log_density_figure,
    predicted_vs_observed_figure,
    prediction_metrics,
    score_stats,
    summary_stats,
    top_fraction_split_means,
    two_panel_histogram,
    weighted_prediction_metrics,
)

STATE_FILE = "surrogate_state.pt"
SUMMARY_FILE = "fit_summary.json"
CONFIG_FILE = "resolved_config.json"
EVAL_SUMMARY_FILE = "eval_summary.json"
TRAIN_RANDOM_FILE = "train_random.csv"
TRAIN_TOP_FILE = "train_top.csv"
EVAL_CSV_COLUMNS = ("SMILE", "y")

#: The generated set: the generated molecules that docked and pass the SA filter.
GENERATED_SET = "gp_molformer_set"

#: Every generated molecule that docked, SA-passing or not. The S3-GFN pool is not
#: SA-filtered, so this is the population that actually gets docked. Scored only at the
#: end, without figures.
GENERATED_DOCKED_SET = "gp_molformer_docked_set"

#: Evaluation sets scored after every training epoch.
PER_EPOCH_SETS = ("train_random", "train_top", "val_set", GENERATED_SET)

#: ``(set, metric)`` pairs left out of the per-epoch curves because they say nothing:
#: ``r2`` on a set restricted to the top of the target only restates the bias, and the
#: bias on a random sample of the training data is always about zero.
SKIPPED_EPOCH_METRICS = frozenset({("train_top", "r2"), ("train_random", "bias")})

#: Learned hyperparameters tracked every epoch, under ``train/epoch/<name>``.
PER_EPOCH_HYPERPARAMETERS = (
    "noise_std_original_scale",
    "outputscale",
    "lengthscale_median",
    "variational_covar_eig_max",
)

#: Entries of the learned-hyperparameter record that describe the data, not the fit.
#: They are identical across arms, so they go to the run config rather than the metrics.
DATA_CONSTANT_HYPERPARAMETERS = ("y_mean", "y_std")

#: Values of ``--trainable``; mirrors ``variational_gp.WARM_START_TRAINABLE`` so the
#: argument parser does not import the surrogate module.
WARM_START_TRAINABLE = ("all", "variance")

#: Acquisition scores below this are treated as zero in the log-scale figures, which
#: otherwise span hundreds of orders of magnitude.
SCORE_LOG_FLOOR = 1e-12


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
    weights : np.ndarray or None
        Optional weight per molecule, for sets that over-sample part of the library.
    """

    name: str
    smiles: tuple[str, ...]
    targets: np.ndarray
    features: Any
    weights: np.ndarray | None = None


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
    back to the original target scale. ``prior_std_original_scale`` is the latent
    standard deviation far from every inducing point, and the two
    ``variational_covar_eig_*`` values are the extreme eigenvalues of the (whitened)
    covariance of the inducing values, whose prior is the identity: a latent standard
    deviation above the prior one requires an eigenvalue above 1. Reads the
    surrogate's private GP modules, since it exposes no public accessor for them, and
    returns an empty dict for a surrogate without that structure.

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
    if gp_model is None or likelihood is None:
        return {}
    state = surrogate.get_state_dict()
    if state is None:
        return {}
    import torch

    lengthscale = gp_model.covar_module.base_kernel.lengthscale.detach().flatten()
    noise = float(likelihood.noise.detach().mean())
    y_std = float(state["outcome_std"])
    outputscale = float(gp_model.covar_module.outputscale.detach())
    with torch.no_grad():
        inducing_distribution = (
            gp_model.variational_strategy._variational_distribution()
        )
        eigenvalues = torch.linalg.eigvalsh(inducing_distribution.covariance_matrix)
    return {
        "noise": noise,
        "noise_std_original_scale": noise**0.5 * y_std,
        "outputscale": outputscale,
        "prior_std_original_scale": outputscale**0.5 * y_std,
        "variational_covar_eig_min": float(eigenvalues.min()),
        "variational_covar_eig_max": float(eigenvalues.max()),
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


def parse_float_column(values: Sequence[str]) -> np.ndarray:
    """Parse a CSV column as floats, with ``nan`` where an entry does not parse."""
    parsed: list[float] = []
    for value in values:
        try:
            parsed.append(float(value))
        except (TypeError, ValueError):
            parsed.append(float("nan"))
    return np.asarray(parsed, dtype=np.float64)


def filter_generated_set(
    smiles: Sequence[str],
    targets: np.ndarray,
    passes_sa: Sequence[str],
) -> tuple[list[str], np.ndarray, np.ndarray, dict[str, int]]:
    """Keep the generated molecules that docked, and flag those passing the SA filter.

    About a tenth of the generated molecules fail to dock and carry a ``nan`` target;
    those failures are biased toward molecules the docking toolchain cannot build, so
    the surviving fraction is reported rather than assumed. The S3-GFN pool is not
    SA-filtered, so every docked molecule is kept; the SA-passing ones (the only
    molecules that get reward-driven updates) are marked for the main generated set.

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
    tuple[list[str], np.ndarray, np.ndarray, dict[str, int]]
        The docked SMILES, their targets, a boolean mask over them that is true for
        the SA-passing molecules, and the counts behind the filter (``n_evaluated`` is
        the number that both docked and pass).

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
        [str(flag).strip() not in ("", "0", "False", "false") for flag in passes_sa],
        dtype=bool,
    )
    docked = np.isfinite(values)
    counts = {
        "n_total": int(values.size),
        "n_docked": int(docked.sum()),
        "n_sa_pass": int(sa_flags.sum()),
        "n_evaluated": int((docked & sa_flags).sum()),
    }
    kept = [smiles[index] for index in np.flatnonzero(docked)]
    return kept, values[docked], sa_flags[docked], counts


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


def log_final_scalars(logger: Any, metrics: Mapping[str, float]) -> None:
    """Record every one-off scalar in the run summary, never as a chart metric.

    A scalar logged once as a metric becomes a one-point chart, which clutters the
    Charts tab. In the summary it appears in the runs table and the Overview under its
    own name, where it can be sorted and filtered, and no chart is created. A
    non-finite value is stored as ``None``.

    Parameters
    ----------
    logger : Logger
        Receives one ``log_summary`` call.
    metrics : Mapping[str, float]
        Scalars keyed by their metric name.
    """
    logger.log_summary(
        {key: value if math.isfinite(value) else None for key, value in metrics.items()}
    )


def log_figures(logger: Any, figures: Mapping[str, Any]) -> None:
    """Buffer every figure on the logger."""
    for key, figure in figures.items():
        logger.log_figure(key, figure)


def set_metrics(
    eval_set: EvalSet, predictions: Mapping[str, np.ndarray]
) -> dict[str, float]:
    """The metric set of one evaluation set, from its predictions.

    Parameters
    ----------
    eval_set : EvalSet
        The scored set.
    predictions : Mapping[str, np.ndarray]
        ``mean``, ``std_total`` and ``std_latent``, as :func:`evaluate_set` returns.

    Returns
    -------
    dict[str, float]
        Metric name to value: the prediction metrics, plus their weighted versions for
        a set that carries weights.
    """
    metrics = prediction_metrics(
        eval_set.targets,
        predictions["mean"],
        predictions["std_total"],
        latent_std=predictions["std_latent"],
    )
    if eval_set.weights is not None:
        metrics.update(
            weighted_prediction_metrics(
                eval_set.targets, predictions["mean"], eval_set.weights
            )
        )
    return metrics


def epoch_metrics(
    surrogate: Any,
    eval_sets: Sequence[EvalSet],
    *,
    chunk_size: int,
) -> dict[str, float]:
    """Score every per-epoch evaluation set and return its metrics, keyed for W&B.

    Parameters
    ----------
    surrogate : Any
        The surrogate being fitted, mid-training.
    eval_sets : Sequence[EvalSet]
        Sets to score.
    chunk_size : int
        Rows per posterior call.

    Returns
    -------
    dict[str, float]
        ``<set>/epoch/<metric>`` to value, without the pairs in
        :data:`SKIPPED_EPOCH_METRICS`.
    """
    metrics: dict[str, float] = {}
    for eval_set in eval_sets:
        predictions = evaluate_set(surrogate, eval_set, chunk_size=chunk_size)
        for name, value in set_metrics(eval_set, predictions).items():
            if (eval_set.name, name) not in SKIPPED_EPOCH_METRICS:
                metrics[f"{eval_set.name}/epoch/{name}"] = value
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
    surrogate: Any,
    name: str,
    smiles: Sequence[str],
    targets: np.ndarray,
    weights: np.ndarray | None = None,
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
    weights : np.ndarray, optional
        Weight per molecule, aligned with ``smiles``.

    Returns
    -------
    EvalSet
        The encoded set.

    Raises
    ------
    ValueError
        If ``weights`` is given and does not have one entry per molecule.
    """
    from activelearning.utils.types import Candidate

    if weights is not None and len(weights) != len(smiles):
        raise ValueError(
            f"{len(weights)} weights for {len(smiles)} molecules in {name}."
        )
    candidates = [Candidate(x=value, fidelity=None) for value in smiles]
    return EvalSet(
        name=name,
        smiles=tuple(smiles),
        targets=np.asarray(targets, dtype=np.float64),
        features=surrogate.encode_candidates(candidates),
        weights=None if weights is None else np.asarray(weights, dtype=np.float64),
    )


def subset_eval_set(eval_set: EvalSet, mask: np.ndarray, name: str) -> EvalSet:
    """Return the molecules of an encoded set selected by a boolean mask.

    Parameters
    ----------
    eval_set : EvalSet
        The encoded set to take from.
    mask : np.ndarray
        Boolean mask with one entry per molecule.
    name : str
        Name of the new set.

    Returns
    -------
    EvalSet
        The selected molecules, reusing the features already encoded.

    Raises
    ------
    ValueError
        If the mask does not have one entry per molecule.
    """
    import torch

    keep = np.asarray(mask, dtype=bool)
    if keep.shape != (len(eval_set.smiles),):
        raise ValueError(
            f"Mask of shape {keep.shape} for {len(eval_set.smiles)} molecules in "
            f"{eval_set.name}."
        )
    rows = np.flatnonzero(keep)
    indices = torch.as_tensor(rows, device=eval_set.features.device)
    return EvalSet(
        name=name,
        smiles=tuple(eval_set.smiles[int(row)] for row in rows),
        targets=eval_set.targets[keep],
        features=eval_set.features.index_select(0, indices),
        weights=None if eval_set.weights is None else eval_set.weights[keep],
    )


def make_epoch_callback(
    logger: Any,
    surrogate: Any,
    *,
    static_sets: Sequence[EvalSet],
    train_row_sets: Mapping[str, np.ndarray],
    observations: Sequence[Any],
    chunk_size: int,
    epoch_offset: int = 0,
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
        Sets already encoded before the fit: ``val_set`` and the generated set.
    train_row_sets : Mapping[str, np.ndarray]
        Set name to row indices into ``observations``.
    observations : Sequence[Any]
        Training observations.
    chunk_size : int
        Rows per posterior call.
    epoch_offset : int, default=0
        Added to the epoch index to get the logged step. A warm-started fit reports
        its starting point as epoch ``-1`` with a ``nan`` loss, so an offset equal to
        the earlier run's epoch count puts that point on the earlier run's last step
        and the new epochs after it.

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
        metrics = epoch_metrics(surrogate, resolved, chunk_size=chunk_size)
        if math.isfinite(mean_train_loss):
            metrics["train/epoch/minibatch_loss"] = mean_train_loss
        hyperparameters = learned_hyperparameters(surrogate)
        for name in PER_EPOCH_HYPERPARAMETERS:
            if name in hyperparameters:
                metrics[f"train/epoch/{name}"] = hyperparameters[name]
        log_scalars(logger, metrics)
        logger.log_step(epoch + epoch_offset)

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
    scores: np.ndarray | None = None,
    final_metrics: bool = False,
    figures: bool = True,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Metrics and figures for one evaluation set after training.

    Parameters
    ----------
    eval_set : EvalSet
        The scored set.
    predictions : Mapping[str, np.ndarray]
        ``mean``, ``std_total`` and ``std_latent``.
    scores : np.ndarray, optional
        Acquisition score per molecule, for the generated sets.
    final_metrics : bool, default=False
        Also report the prediction metrics under ``<set>/final/``. For a set tracked
        every epoch they repeat its last epoch, so this is off by default.
    figures : bool, default=True
        Whether to build the figures.

    Returns
    -------
    tuple[dict[str, float], dict[str, Any]]
        Scalar metrics and figures, both keyed for W&B.
    """
    name = eval_set.name
    targets = eval_set.targets
    latent = predictions["std_latent"]
    metrics: dict[str, float] = {}
    if final_metrics:
        for key, value in set_metrics(eval_set, predictions).items():
            metrics[f"{name}/final/{key}"] = value
    for key, value in summary_stats(latent).items():
        metrics[f"{name}/std_latent/{key}"] = value
    top_mean, rest_mean = top_fraction_split_means(latent, targets)
    metrics[f"{name}/std_latent/top1pct_y_mean"] = top_mean
    metrics[f"{name}/std_latent/rest_mean"] = rest_mean
    if scores is not None:
        for key, value in score_stats(scores, targets).items():
            metrics[f"{name}/acquisition_score/{key}"] = value
    if not figures:
        return metrics, {}

    built: dict[str, Any] = {
        "predicted_vs_observed": predicted_vs_observed_figure(
            title=name, targets=targets, mean=predictions["mean"]
        ),
        "std_latent": two_panel_histogram(
            latent, title=f"{name}: std_latent", xlabel="latent predicted std"
        ),
        "std_latent_vs_observed": log_density_figure(
            title=f"{name}: latent std vs observed target",
            targets=targets,
            values=latent,
            ylabel="latent predicted std",
        ),
        "error_by_std_latent": error_by_std_figure(
            title=f"{name}: error by latent std",
            targets=targets,
            mean=predictions["mean"],
            std=latent,
        ),
        "error_by_std_total": error_by_std_figure(
            title=f"{name}: error by total std",
            targets=targets,
            mean=predictions["mean"],
            std=predictions["std_total"],
        ),
    }
    if scores is not None:
        built["acquisition_score"] = two_panel_histogram(
            scores,
            title=f"{name}: acquisition_score",
            xlabel="GIBBON information gain",
            log_floor=SCORE_LOG_FLOOR,
        )
        built["acquisition_score_vs_observed"] = log_density_figure(
            title=f"{name}: GIBBON score vs observed target",
            targets=targets,
            values=scores,
            ylabel="GIBBON information gain",
            floor=SCORE_LOG_FLOOR,
        )
    return metrics, {
        f"{name}/figures/{key}": figure
        for key, figure in built.items()
        if figure is not None
    }


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
    parser.add_argument(
        "--init-state",
        type=Path,
        default=None,
        help="Saved surrogate_state.pt to continue training from.",
    )
    parser.add_argument(
        "--trainable",
        choices=WARM_START_TRAINABLE,
        default="all",
        help="With --init-state: 'variance' trains only the inducing covariance and "
        "the noise, which leaves the predictive mean unchanged.",
    )
    parser.add_argument(
        "--epoch-offset",
        type=int,
        default=0,
        help="Added to the epoch index when logging. With --init-state, set it to the "
        "epoch count of the run the state came from.",
    )
    args, config_args = parser.parse_known_args(argv)
    if args.n_train_random < 0 or args.n_train_top < 0:
        parser.error("--n-train-random and --n-train-top must be nonnegative.")
    if args.eval_chunk_size < 1:
        parser.error("--eval-chunk-size must be positive.")
    if args.epoch_offset < 0:
        parser.error("--epoch-offset must be nonnegative.")
    if args.init_state is None and args.trainable != "all":
        parser.error("--trainable needs --init-state.")
    if args.init_state is not None and args.epoch_offset < 1:
        # The starting point is logged at step `epoch_offset - 1`, which must not be
        # negative.
        parser.error("--init-state needs --epoch-offset of at least 1.")
    return args, config_args


def check_init_state(init_state: Path | None, out_dir: Path) -> None:
    """Fail fast on an initial state that is missing or is this run's own output.

    Parameters
    ----------
    init_state : Path or None
        The ``--init-state`` argument.
    out_dir : Path
        The run's output directory.

    Raises
    ------
    SystemExit
        If the file does not exist, or is the state file this run would write.
    """
    if init_state is None:
        return
    if not init_state.is_file():
        raise SystemExit(f"--init-state {init_state} does not exist.")
    if init_state.resolve() == (out_dir / STATE_FILE).resolve():
        raise SystemExit(
            f"--init-state {init_state} is this run's own output; choose a different "
            "--output-dir so the state it starts from is not overwritten."
        )


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
        "inducing_init": surrogate_cfg.get("inducing_init"),
        "inducing_strata_quantiles": surrogate_cfg.get("inducing_strata_quantiles"),
        "inducing_strata_fractions": surrogate_cfg.get("inducing_strata_fractions"),
        "inducing_init_seed": surrogate_cfg.get("inducing_init_seed"),
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
        "gp_molformer_docked_filter": "docked",
        "val_label_route": "pprop_hit_rate",
        "gp_molformer_label_route": "dock3_oracle",
        "output_dir": str(args.output_dir),
        "init_state": None if args.init_state is None else str(args.init_state),
        "trainable": args.trainable,
        "epoch_offset": args.epoch_offset,
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
    check_init_state(args.init_state, out_dir)

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
    val_weights = np.empty(0)
    docked_smiles: list[str] = []
    docked_targets = np.empty(0)
    docked_passes_sa = np.empty(0, dtype=bool)
    generated_counts: dict[str, int] = {}
    if not args.skip_eval:
        val_smiles, val_targets, val_extras = load_labelled_csv(
            args.val_csv, require_columns=("weight",)
        )
        val_weights = parse_float_column(val_extras["weight"])
        raw_smiles, raw_targets, extras = load_labelled_csv(
            args.gp_molformer_csv, require_columns=("passes_sa",)
        )
        docked_smiles, docked_targets, docked_passes_sa, generated_counts = (
            filter_generated_set(raw_smiles, raw_targets, extras["passes_sa"])
        )
        print(
            f"Loaded {len(val_smiles)} val rows and {generated_counts['n_docked']} "
            f"docked of {generated_counts['n_total']} generated rows "
            f"({generated_counts['n_evaluated']} of them SA-passing; "
            f"SA-passing overall {generated_counts['n_sa_pass']}).",
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
        docked_set: EvalSet | None = None
        if not args.skip_eval:
            print(f"Encoding {len(val_smiles)} validation molecules.", flush=True)
            static_sets.append(
                encode_eval_set(
                    surrogate, "val_set", val_smiles, val_targets, val_weights
                )
            )
            # Encoded before the fit so the generated set gets per-epoch curves too.
            print(f"Encoding {len(docked_smiles)} generated molecules.", flush=True)
            docked_set = encode_eval_set(
                surrogate, GENERATED_DOCKED_SET, docked_smiles, docked_targets
            )
            static_sets.append(
                subset_eval_set(docked_set, docked_passes_sa, GENERATED_SET)
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
                    chunk_size=args.eval_chunk_size,
                    epoch_offset=args.epoch_offset,
                )
            )

        # Re-seed so the fit is identical whether or not evaluation is enabled: the
        # inducing points are drawn from the global RNG inside fit().
        set_global_seed(cfg.runtime.seed)
        import torch

        if args.init_state is not None:
            print(
                f"Continuing from {args.init_state} (trainable: {args.trainable}).",
                flush=True,
            )
            surrogate.warm_start_from(
                torch.load(args.init_state, map_location="cpu", weights_only=True),
                trainable=args.trainable,
            )
        print(f"Fitting surrogate on {len(observations)} observations.", flush=True)
        started = time.perf_counter()
        surrogate.fit(observations)
        fit_seconds = time.perf_counter() - started
        surrogate.set_epoch_callback(None)
        print(f"Surrogate fit in {fit_seconds:.0f} s.", flush=True)

        state = surrogate.get_state_dict()
        if state is None:
            raise SystemExit("The fitted surrogate has no state to save.")

        torch.save(state, out_dir / STATE_FILE)
        print(f"Saved surrogate state to {out_dir / STATE_FILE}", flush=True)

        training_params = dict(resolved.get("surrogate", {})).get("training_params", {})
        summary = {
            "n_observations": len(observations),
            "fit_seconds": fit_seconds,
            "objective": training_params.get("variational_objective"),
            "num_inducing": dict(resolved.get("surrogate", {})).get("num_inducing"),
            "inducing_init": dict(resolved.get("surrogate", {})).get("inducing_init"),
            "epochs": training_params.get("epochs"),
            "lr": training_params.get("lr"),
            "batch_size": training_params.get("batch_size"),
            "seed": cfg.runtime.seed,
            "eval_seed": args.eval_seed,
            "init_state": None if args.init_state is None else str(args.init_state),
            "trainable": args.trainable,
            "epoch_offset": args.epoch_offset,
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

        # The target's mean and std describe the data, not the fit: they go with the
        # other inputs in the run config, like the set sizes logged before the fit.
        logger.log_config(
            {
                name: summary["hyperparameters"][name]
                for name in DATA_CONSTANT_HYPERPARAMETERS
                if name in summary["hyperparameters"]
            }
        )
        final_metrics: dict[str, float] = {
            "run/fit/seconds": fit_seconds,
            **{
                f"run/hyperparameters/{name}": value
                for name, value in summary["hyperparameters"].items()
                if name not in DATA_CONSTANT_HYPERPARAMETERS
            },
        }
        final_figures: dict[str, Any] = {}

        if not args.skip_eval:
            assert docked_set is not None
            final_metrics.update(
                update_acquisition(
                    acquisition,
                    surrogate,
                    observations,
                    seed=args.acquisition_seed,
                )
            )
            print("Scoring the generated molecules with GIBBON.", flush=True)
            docked_scores = score_acquisition(acquisition, docked_set.smiles)
            scores_by_set = {
                GENERATED_SET: docked_scores[docked_passes_sa],
                GENERATED_DOCKED_SET: docked_scores,
            }

            all_sets = [
                build_train_eval_set(
                    surrogate, "train_random", observations, random_rows
                ),
                build_train_eval_set(surrogate, "train_top", observations, top_rows),
                *static_sets,
                docked_set,
            ]
            for eval_set in all_sets:
                print(f"Evaluating {eval_set.name}.", flush=True)
                predictions = evaluate_set(
                    surrogate, eval_set, chunk_size=args.eval_chunk_size
                )
                columns = dict(predictions)
                scores = scores_by_set.get(eval_set.name)
                if scores is not None:
                    rewards = reward_columns(
                        scores, transform=args.reward_transform, beta=args.reward_beta
                    )
                    final_metrics[f"{eval_set.name}/final/reward_overflow_count"] = (
                        float(np.sum(rewards["log_reward"] > 700.0))
                    )
                    columns["acquisition_score"] = scores
                    columns.update(rewards)
                if eval_set.name == GENERATED_DOCKED_SET:
                    columns["passes_sa"] = docked_passes_sa.astype(np.float64)
                # The generated set keeps its `final/` metrics although it now has
                # per-epoch curves: they are the columns the earlier runs are read by.
                metrics, figures = final_set_outputs(
                    eval_set,
                    predictions,
                    scores=scores,
                    final_metrics=scores is not None,
                    figures=eval_set.name != GENERATED_DOCKED_SET,
                )
                final_metrics.update(metrics)
                final_figures.update(figures)
                write_per_molecule_csv(
                    out_dir / "eval" / f"{eval_set.name}.csv", eval_set, columns
                )

        # One-off scalars (including the train time, `run/fit/seconds`) go in the run
        # summary, not the Charts tab; only the figures are logged, at the step after
        # the last epoch. The per-epoch curves are the only scalars charted.
        log_final_scalars(logger, final_metrics)
        log_figures(logger, final_figures)
        epochs = int(training_params.get("epochs") or 0)
        logger.log_step(args.epoch_offset + epochs + 1)
        write_json(out_dir / EVAL_SUMMARY_FILE, final_metrics)
        print(f"Wrote {out_dir / EVAL_SUMMARY_FILE}", flush=True)
    finally:
        logger.end()
    print("Done.", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
