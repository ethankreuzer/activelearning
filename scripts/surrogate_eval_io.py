"""Shared I/O, logging and evaluation-set plumbing for the surrogate studies.

``scripts/surrogate_eval_fit.py`` grew these helpers first;
``scripts/exact_dkl_top_n.py`` needs the same ones. They live here so the two
runners share one implementation rather than repeating the copy-and-paste that
``scripts/surrogate_dataset_size_study.py`` already represents.

Everything here is pure plumbing: reading and writing files, building the run
logger, and slicing encoded evaluation sets. The numerical work lives in
``scripts/surrogate_eval_metrics.py``, and neither module imports torch at
module scope so they stay importable without a GPU.
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

#: Generated molecules that docked and pass the SA filter. The only generated
#: set that gets prediction metrics.
GENERATED_SET = "gp_molformer_set"

#: Every generated molecule that docked, SA-passing or not.
GENERATED_DOCKED_SET = "gp_molformer_docked_set"

#: Acquisition scores span many orders of magnitude and underflow to exactly
#: zero, so the score figures clamp at this floor before taking a logarithm.
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
        Observed target per molecule. All ``nan`` for a set without labels.
    features : Any
        Encoded feature matrix (a ``torch.Tensor``), shaped ``(n, feature_dim)``.
        Typed loosely so this module does not import torch at module scope.
    weights : np.ndarray or None
        Optional weight per molecule, for sets that over-sample part of the
        library.
    """

    name: str
    smiles: tuple[str, ...]
    targets: np.ndarray
    features: Any
    weights: np.ndarray | None = None

    @property
    def has_targets(self) -> bool:
        """Return whether any molecule in the set carries a finite target."""
        return bool(np.isfinite(self.targets).any())


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write ``payload`` as sorted, indented JSON, atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp_path, path)


def parse_float_column(values: Sequence[str]) -> np.ndarray:
    """Parse a CSV column as floats, with ``nan`` where an entry does not parse."""
    parsed: list[float] = []
    for value in values:
        try:
            parsed.append(float(value))
        except (TypeError, ValueError):
            parsed.append(float("nan"))
    return np.asarray(parsed, dtype=np.float64)


def load_labelled_csv(
    path: Path,
    *,
    smiles_column: str = "SMILES",
    label_column: str | None = "y",
    require_columns: Sequence[str] = (),
) -> tuple[list[str], np.ndarray, dict[str, list[str]]]:
    """Read a CSV of molecules, with or without labels.

    Parameters
    ----------
    path : Path
        CSV to read.
    smiles_column : str, default="SMILES"
        Column holding the molecule input.
    label_column : str or None, default="y"
        Column holding the target. ``None`` is for sets that carry no label at
        all, such as the in-vitro set; every target then comes back ``nan``.
    require_columns : Sequence[str], default=()
        Extra columns that must be present and are returned verbatim.

    Returns
    -------
    tuple[list[str], np.ndarray, dict[str, list[str]]]
        The SMILES, the targets (``nan`` where they do not parse or are absent)
        and the extras.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If a required column is missing.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")
    required = [smiles_column, *require_columns]
    if label_column is not None:
        required.append(label_column)
    smiles: list[str] = []
    targets: list[float] = []
    extras: dict[str, list[str]] = {name: [] for name in require_columns}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in required if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(
                f"{path} is missing column(s) {missing}; it has {reader.fieldnames}."
            )
        for row in reader:
            smiles.append(row[smiles_column])
            if label_column is None:
                targets.append(float("nan"))
            else:
                try:
                    targets.append(float(row[label_column]))
                except (TypeError, ValueError):
                    targets.append(float("nan"))
            for name in require_columns:
                extras[name].append(row[name])
    return smiles, np.asarray(targets, dtype=np.float64), extras


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


def filter_generated_set(
    smiles: Sequence[str],
    targets: np.ndarray,
    passes_sa: Sequence[str],
) -> tuple[list[str], np.ndarray, np.ndarray, dict[str, int]]:
    """Keep the generated molecules that docked, and flag those passing the SA filter.

    About a tenth of the generated molecules fail to dock and carry a ``nan``
    target; those failures are biased toward molecules the docking toolchain
    cannot build, so the surviving fraction is reported rather than assumed. The
    S3-GFN pool is not SA-filtered, so every docked molecule is kept; the
    SA-passing ones are marked for the main generated set.

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
        The docked SMILES, their targets, a boolean mask over them that is true
        for the SA-passing molecules, and the counts behind the filter.

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


def build_run_logger(
    *,
    project: str | None,
    run_name: str,
    entity: str | None = None,
    tags: Sequence[str] | None = None,
    group: str | None = None,
    console_project: str = "surrogate-eval",
) -> Any:
    """Build the logger the run uses, falling back to the console without a project.

    Parameters
    ----------
    project : str or None
        W&B project. ``None`` selects a ``ConsoleLogger`` instead, so the script
        runs end to end without W&B.
    run_name : str
        Run name.
    entity : str, optional
        W&B team or user. Without it wandb resolves the account from the
        environment.
    tags : Sequence[str], optional
        Run tags.
    group : str, optional
        Run group, for showing the arms of one study together.
    console_project : str, default="surrogate-eval"
        Project name given to the console fallback.

    Returns
    -------
    Any
        A ``Logger``.
    """
    from activelearning.logger.logger import ConsoleLogger, WandbLogger

    if project is None:
        return ConsoleLogger(project_name=console_project, run_name=run_name)
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

    A scalar logged once as a metric becomes a one-point chart, which clutters
    the Charts tab. In the summary it appears in the runs table and the Overview
    under its own name, where it can be sorted and filtered, and no chart is
    created. A non-finite value is stored as ``None``.

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


def evaluate_encoded_set(
    surrogate: Any, eval_set: EvalSet, *, chunk_size: int
) -> dict[str, np.ndarray]:
    """Predict mean and both standard deviations for one encoded set.

    Goes through ``predict_encoded`` rather than ``predict``: the features are
    already in model space, and re-encoding them per chunk would miss a
    prefix-keyed feature cache and fall back to live inference.

    Parameters
    ----------
    surrogate : Any
        Fitted surrogate exposing ``predict_encoded``.
    eval_set : EvalSet
        Set to score.
    chunk_size : int
        Rows per posterior call.

    Returns
    -------
    dict[str, np.ndarray]
        ``mean``, ``std_total`` (with observation noise) and ``std_latent``
        (without, which is what the acquisition consumes).
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


def prediction_outputs(
    eval_set: EvalSet,
    predictions: Mapping[str, np.ndarray],
    *,
    figures: bool = True,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Final prediction metrics and figures for one set.

    Split out from the combined prediction-and-score reporting the earlier study
    used, because this study has sets that are predicted but not scored, sets
    that are scored but not predicted, and one that is both.

    Parameters
    ----------
    eval_set : EvalSet
        The set, supplying targets and optional weights.
    predictions : Mapping[str, np.ndarray]
        ``mean``, ``std_total`` and ``std_latent``.
    figures : bool, default=True
        Whether to build the figures.

    Returns
    -------
    tuple[dict[str, float], dict[str, Any]]
        Metrics keyed ``<set>/final/<metric>`` and ``<set>/std_latent/<stat>``,
        and figures keyed ``<set>/figures/<name>``.
    """
    from scripts.surrogate_eval_metrics import (
        error_by_std_figure,
        predicted_vs_observed_figure,
        prediction_metrics,
        summary_stats,
        top_fraction_split_means,
        two_panel_histogram,
        weighted_prediction_metrics,
    )

    name = eval_set.name
    metrics: dict[str, float] = {}
    for key, value in prediction_metrics(
        eval_set.targets,
        predictions["mean"],
        predictions["std_total"],
        latent_std=predictions["std_latent"],
    ).items():
        metrics[f"{name}/final/{key}"] = value
    if eval_set.weights is not None:
        for key, value in weighted_prediction_metrics(
            eval_set.targets, predictions["mean"], eval_set.weights
        ).items():
            metrics[f"{name}/final/{key}"] = value

    latent = predictions["std_latent"]
    for key, value in summary_stats(latent).items():
        metrics[f"{name}/std_latent/{key}"] = value
    top_mean, rest_mean = top_fraction_split_means(latent, eval_set.targets)
    metrics[f"{name}/std_latent/top1pct_y_mean"] = top_mean
    metrics[f"{name}/std_latent/rest_mean"] = rest_mean

    built: dict[str, Any] = {}
    if figures:
        predicted = predicted_vs_observed_figure(
            title=name, targets=eval_set.targets, mean=predictions["mean"]
        )
        if predicted is not None:
            built[f"{name}/figures/predicted_vs_observed"] = predicted
        built[f"{name}/figures/std_latent"] = two_panel_histogram(
            latent, title=f"{name} latent std", xlabel="latent std"
        )
        for std_name, values in (
            ("latent", predictions["std_latent"]),
            ("total", predictions["std_total"]),
        ):
            figure = error_by_std_figure(
                title=f"{name} error by {std_name} std",
                targets=eval_set.targets,
                mean=predictions["mean"],
                std=values,
            )
            if figure is not None:
                built[f"{name}/figures/error_by_std_{std_name}"] = figure
    return metrics, built


def score_outputs(
    eval_set: EvalSet,
    scores: np.ndarray,
    *,
    scoring: str,
    label: str,
    log_floor: float | None = SCORE_LOG_FLOOR,
    figures: bool = True,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Acquisition-score statistics and figures for one set.

    Parameters
    ----------
    eval_set : EvalSet
        The set, supplying targets for the rank correlation. A set with no
        finite target yields ``nan`` for ``spearman_y`` and no score-versus-
        observed figure.
    scores : np.ndarray
        One acquisition score per molecule.
    scoring : str
        Key segment naming the scoring, such as ``gibbon_value`` or ``qmfmes``.
    label : str
        Human-readable axis label for the figures.
    log_floor : float, optional
        Clamp applied before taking a logarithm in the figures. ``None`` for
        scores that are already logarithms and so may be negative.
    figures : bool, default=True
        Whether to build the figures.

    Returns
    -------
    tuple[dict[str, float], dict[str, Any]]
        Metrics keyed ``<set>/acquisition/<scoring>/<stat>`` and figures keyed
        ``<set>/figures/<scoring>/<name>``.
    """
    from scripts.surrogate_eval_metrics import (
        log_density_figure,
        score_stats,
        two_panel_histogram,
    )

    name = eval_set.name
    values = np.asarray(scores, dtype=np.float64)
    metrics = {
        f"{name}/acquisition/{scoring}/{key}": value
        for key, value in score_stats(values, eval_set.targets).items()
    }
    metrics[f"{name}/acquisition/{scoring}/count"] = float(values.size)
    metrics[f"{name}/acquisition/{scoring}/n_nonfinite"] = float(
        int((~np.isfinite(values)).sum())
    )

    built: dict[str, Any] = {}
    if figures:
        built[f"{name}/figures/{scoring}/histogram"] = two_panel_histogram(
            values, title=f"{name} {label}", xlabel=label, log_floor=log_floor
        )
        # Skipped explicitly rather than relying on the figure helper returning
        # None: a set with no labels has nothing to correlate against.
        if eval_set.has_targets:
            figure = log_density_figure(
                title=f"{name} {label} vs observed",
                targets=eval_set.targets,
                values=values,
                ylabel=label,
                floor=log_floor,
            )
            if figure is not None:
                built[f"{name}/figures/{scoring}/vs_observed"] = figure
    return metrics, built
