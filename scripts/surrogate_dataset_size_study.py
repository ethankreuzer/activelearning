"""Score fixed molecule sets with a surrogate fit on a given training set.

This reproduces the first step of an active learning round (surrogate fit,
then ``acquisition.update``) and, instead of sampling, scores every SMILES in
one or more evaluation CSVs. Running it once per training-set size shows how
the surrogate predictions and acquisition scores change with the amount of
data the surrogate saw.

For each evaluation CSV ``NAME=PATH`` it writes, under
``<output-dir>/<train-label>/``:

- ``NAME_surrogate.csv``: the original columns plus ``pred_mean`` and ``pred_std``
- ``NAME_acquisition.csv``: the original columns plus ``acquisition_score``
- ``NAME_surrogate.png`` and ``NAME_acquisition.png``: histograms of
  ``pred_mean`` and ``acquisition_score``

Molecules RDKit cannot parse, or that fail to encode, get ``NaN``.

Usage mirrors the ``activelearning`` console script::

    python scripts/surrogate_dataset_size_study.py \\
        config/ampc/s3gfn_minimol_ampc_variational_single_fidelity.yaml \\
        acquisition.type=QLowerBoundMaxValueEntropy \\
        --train-label 10m \\
        --eval-csv ampc_331k=data/ampc_subset_331k.csv \\
        --eval-csv olivier_invitro=data/Olivier_Invitro.csv

Only the dataset, surrogate and acquisition are built from the config; the
oracle, sampler and logger sections are validated but never instantiated.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from matplotlib.figure import Figure

from activelearning.utils.types import Candidate

DEFAULT_OUTPUT_DIR = Path("outputs/ampc/dataset_size_study")
DEFAULT_SMILES_COLUMN = "SMILES"
DEFAULT_CHUNK_SIZE = 5000
HISTOGRAM_BINS = 100

SURROGATE_COLUMNS = ("pred_mean", "pred_std")
ACQUISITION_COLUMNS = ("acquisition_score",)


@dataclass(frozen=True)
class EvalSpec:
    """One evaluation CSV to score.

    Attributes
    ----------
    name : str
        Short name used as the output file prefix.
    path : Path
        Path to the CSV file.
    smiles_column : str
        Column holding the SMILES strings.
    """

    name: str
    path: Path
    smiles_column: str = DEFAULT_SMILES_COLUMN


def parse_eval_spec(value: str) -> EvalSpec:
    """Parse ``NAME=PATH[:SMILES_COLUMN]`` into an :class:`EvalSpec`.

    Parameters
    ----------
    value : str
        Command-line value.

    Returns
    -------
    EvalSpec
        The parsed specification.

    Raises
    ------
    argparse.ArgumentTypeError
        If the value is not of the form ``NAME=PATH[:SMILES_COLUMN]``.
    """
    name, sep, rest = value.partition("=")
    if not sep or not name or not rest:
        raise argparse.ArgumentTypeError(
            f"--eval-csv expects NAME=PATH[:SMILES_COLUMN], got {value!r}."
        )
    path, _, column = rest.partition(":")
    return EvalSpec(
        name=name,
        path=Path(path),
        smiles_column=column or DEFAULT_SMILES_COLUMN,
    )


def read_csv_rows(spec: EvalSpec) -> tuple[list[str], list[dict[str, str]]]:
    """Read an evaluation CSV, keeping every column and the row order.

    Parameters
    ----------
    spec : EvalSpec
        The CSV to read.

    Returns
    -------
    fieldnames : list[str]
        Column names in file order.
    rows : list[dict[str, str]]
        One mapping per data row.

    Raises
    ------
    ValueError
        If the SMILES column is missing.
    """
    with spec.path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        if spec.smiles_column not in fieldnames:
            raise ValueError(
                f"{spec.path} has no {spec.smiles_column!r} column; "
                f"columns are {fieldnames}."
            )
        rows = list(reader)
    return fieldnames, rows


def valid_smiles_mask(smiles: Sequence[str]) -> list[bool]:
    """Return whether RDKit can parse each SMILES.

    Invalid molecules are kept out of the encoder, since a single bad SMILES
    raises inside MiniMol and would fail its whole chunk.

    Parameters
    ----------
    smiles : Sequence[str]
        SMILES strings.

    Returns
    -------
    list[bool]
        ``True`` where RDKit returns a molecule.
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    return [bool(s) and Chem.MolFromSmiles(s) is not None for s in smiles]


def score_in_chunks(
    candidates: Sequence[Candidate],
    score_fn: Callable[[list[Candidate]], dict[str, list[float]]],
    columns: Sequence[str],
    *,
    chunk_size: int,
    label: str,
) -> dict[str, list[float]]:
    """Score candidates chunk by chunk, isolating molecules that fail.

    A chunk that raises is retried one molecule at a time, and molecules that
    still fail get ``NaN`` so one bad input cannot abort a long run.

    Parameters
    ----------
    candidates : Sequence[Candidate]
        Candidates to score.
    score_fn : callable
        Maps a list of candidates to ``{column: values}`` in input order.
    columns : Sequence[str]
        Output columns ``score_fn`` returns.
    chunk_size : int
        Number of candidates per call.
    label : str
        Name used in progress messages.

    Returns
    -------
    dict[str, list[float]]
        One list per column, aligned with ``candidates``.
    """
    results: dict[str, list[float]] = {column: [] for column in columns}
    started = time.perf_counter()
    for start in range(0, len(candidates), chunk_size):
        chunk = list(candidates[start : start + chunk_size])
        try:
            chunk_results = score_fn(chunk)
        except Exception as error:  # noqa: BLE001 - isolate the failing molecule
            print(
                f"[{label}] chunk at {start} failed ({error!r}); "
                "retrying one molecule at a time.",
                flush=True,
            )
            chunk_results = {column: [] for column in columns}
            for candidate in chunk:
                try:
                    single = score_fn([candidate])
                except Exception:  # noqa: BLE001
                    single = {column: [math.nan] for column in columns}
                for column in columns:
                    chunk_results[column].extend(single[column])
        for column in columns:
            results[column].extend(float(v) for v in chunk_results[column])
        done = min(start + chunk_size, len(candidates))
        print(
            f"[{label}] {done}/{len(candidates)} "
            f"({time.perf_counter() - started:.0f} s)",
            flush=True,
        )
    return results


def score_eval_set(
    smiles: Sequence[str],
    score_fn: Callable[[list[Candidate]], dict[str, list[float]]],
    columns: Sequence[str],
    *,
    fidelity: int,
    chunk_size: int,
    label: str,
) -> dict[str, list[float]]:
    """Score every valid SMILES and put ``NaN`` in the rows of invalid ones.

    Parameters
    ----------
    smiles : Sequence[str]
        SMILES strings in file order.
    score_fn : callable
        Maps a list of candidates to ``{column: values}`` in input order.
    columns : Sequence[str]
        Output columns ``score_fn`` returns.
    fidelity : int
        Fidelity level assigned to every candidate.
    chunk_size : int
        Number of candidates per call.
    label : str
        Name used in progress messages.

    Returns
    -------
    dict[str, list[float]]
        One list per column, aligned with ``smiles``.
    """
    mask = valid_smiles_mask(smiles)
    valid_indices = [index for index, ok in enumerate(mask) if ok]
    if len(valid_indices) < len(smiles):
        print(
            f"[{label}] {len(smiles) - len(valid_indices)} SMILES RDKit cannot "
            "parse; they are written as NaN.",
            flush=True,
        )
    candidates = [Candidate(x=smiles[i], fidelity=fidelity) for i in valid_indices]
    scored = score_in_chunks(
        candidates, score_fn, columns, chunk_size=chunk_size, label=label
    )
    results = {column: [math.nan] * len(smiles) for column in columns}
    for column in columns:
        for row_index, value in zip(valid_indices, scored[column]):
            results[column][row_index] = value
    return results


def write_scored_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[dict[str, str]],
    results: dict[str, list[float]],
) -> None:
    """Write the original rows with the scored columns appended.

    Parameters
    ----------
    path : Path
        Output CSV path.
    fieldnames : Sequence[str]
        Original column names.
    rows : Sequence[dict[str, str]]
        Original rows.
    results : dict[str, list[float]]
        New columns, aligned with ``rows``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    new_columns = list(results)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*fieldnames, *new_columns])
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow(
                {**row, **{column: results[column][index] for column in new_columns}}
            )


def read_column(path: Path, column: str) -> list[float]:
    """Read one numeric column from a CSV, parsing blanks as ``NaN``.

    Parameters
    ----------
    path : Path
        CSV path.
    column : str
        Column name.

    Returns
    -------
    list[float]
        Column values in row order.
    """
    with path.open(newline="") as handle:
        return [
            float(row[column]) if row[column] not in ("", None) else math.nan
            for row in csv.DictReader(handle)
        ]


def plot_distribution(
    values: Sequence[float],
    path: Path,
    *,
    title: str,
    xlabel: str,
) -> None:
    """Save a histogram of the finite values.

    Parameters
    ----------
    values : Sequence[float]
        Values to plot; non-finite entries are counted but not drawn.
    path : Path
        Output PNG path.
    title : str
        Figure title.
    xlabel : str
        X-axis label.
    """
    finite = [v for v in values if math.isfinite(v)]
    figure = Figure(figsize=(7.0, 4.5))
    axis = figure.add_subplot(1, 1, 1)
    if finite:
        axis.hist(finite, bins=HISTOGRAM_BINS)
        summary = (
            f"n={len(finite)}  NaN={len(values) - len(finite)}  "
            f"median={statistics.median(finite):.4g}"
        )
    else:
        summary = f"n=0  NaN={len(values)}"
    axis.set_title(f"{title}\n{summary}", fontsize=10)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Count")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)


def plot_outputs(out_dir: Path, specs: Sequence[EvalSpec], train_label: str) -> None:
    """Build the histograms for every evaluation set from its CSVs.

    Parameters
    ----------
    out_dir : Path
        Directory holding the scored CSVs.
    specs : Sequence[EvalSpec]
        Evaluation sets to plot.
    train_label : str
        Training-set label used in titles.
    """
    for spec in specs:
        surrogate_csv = out_dir / f"{spec.name}_surrogate.csv"
        plot_distribution(
            read_column(surrogate_csv, "pred_mean"),
            surrogate_csv.with_suffix(".png"),
            title=f"Surrogate mean, fit on {train_label}, scored on {spec.name}",
            xlabel="Predicted mean",
        )
        acquisition_csv = out_dir / f"{spec.name}_acquisition.csv"
        plot_distribution(
            read_column(acquisition_csv, "acquisition_score"),
            acquisition_csv.with_suffix(".png"),
            title=f"Acquisition score, fit on {train_label}, scored on {spec.name}",
            xlabel="Acquisition score",
        )


def surrogate_score_fn(surrogate: Any) -> Callable[[list[Candidate]], dict]:
    """Wrap ``surrogate.predict`` as a chunk scoring function."""

    def score(candidates: list[Candidate]) -> dict[str, list[float]]:
        prediction = surrogate.predict(candidates)
        return {"pred_mean": prediction["mean"], "pred_std": prediction["std"]}

    return score


def acquisition_score_fn(acquisition: Any) -> Callable[[list[Candidate]], dict]:
    """Wrap ``acquisition.score`` as a chunk scoring function."""

    def score(candidates: list[Candidate]) -> dict[str, list[float]]:
        return {"acquisition_score": acquisition.score(candidates)}

    return score


def run_study(
    surrogate: Any,
    acquisition: Any,
    observations: Sequence[Any],
    specs: Sequence[EvalSpec],
    out_dir: Path,
    *,
    train_label: str,
    fidelity: int,
    chunk_size: int,
) -> None:
    """Fit the surrogate and acquisition, then score and plot every eval set.

    Parameters
    ----------
    surrogate : Surrogate
        Surrogate to fit.
    acquisition : Acquisition
        Acquisition to couple to the fitted surrogate. It must support
        singleton scoring.
    observations : Sequence[Observation]
        Training observations.
    specs : Sequence[EvalSpec]
        Evaluation sets.
    out_dir : Path
        Output directory for this training set.
    train_label : str
        Training-set label used in titles.
    fidelity : int
        Fidelity level assigned to every evaluation candidate.
    chunk_size : int
        Number of candidates per surrogate/acquisition call.
    """
    if not acquisition.supports_singleton_scoring:
        raise ValueError(
            f"{type(acquisition).__name__} does not support singleton scoring."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    print(f"Fitting surrogate on {len(observations)} observations.", flush=True)
    surrogate.fit(observations)
    print(f"Surrogate fit in {time.perf_counter() - started:.0f} s.", flush=True)
    _save_surrogate_state(surrogate, out_dir / "surrogate_state.pt")

    started = time.perf_counter()
    acquisition.update(surrogate, observations)
    print(f"Acquisition updated in {time.perf_counter() - started:.0f} s.", flush=True)

    for spec in specs:
        fieldnames, rows = read_csv_rows(spec)
        smiles = [row[spec.smiles_column] for row in rows]
        for kind, score_fn, columns in (
            ("surrogate", surrogate_score_fn(surrogate), SURROGATE_COLUMNS),
            ("acquisition", acquisition_score_fn(acquisition), ACQUISITION_COLUMNS),
        ):
            results = score_eval_set(
                smiles,
                score_fn,
                columns,
                fidelity=fidelity,
                chunk_size=chunk_size,
                label=f"{spec.name}/{kind}",
            )
            path = out_dir / f"{spec.name}_{kind}.csv"
            write_scored_csv(path, fieldnames, rows, results)
            print(f"Wrote {path}", flush=True)

    plot_outputs(out_dir, specs, train_label)


def _save_surrogate_state(surrogate: Any, path: Path) -> None:
    """Save the fitted surrogate state, so a long fit survives a scoring failure."""
    get_state = getattr(surrogate, "get_state_dict", None)
    state = get_state() if callable(get_state) else None
    if state is None:
        return
    import torch

    torch.save(state, path)
    print(f"Saved surrogate state to {path}", flush=True)


def _parse_args(argv: Sequence[str] | None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the configured surrogate and acquisition on the config's "
            "initial dataset, then score evaluation CSVs. Remaining positional "
            "arguments are config files and key=value overrides, as for "
            "the activelearning CLI."
        ),
    )
    parser.add_argument(
        "--eval-csv",
        type=parse_eval_spec,
        action="append",
        required=True,
        metavar="NAME=PATH[:SMILES_COLUMN]",
        help=(
            "Evaluation CSV to score (repeatable). "
            f"Default column: {DEFAULT_SMILES_COLUMN}."
        ),
    )
    parser.add_argument("--train-label", required=True, help="e.g. 10k or 10m.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument(
        "--fidelity",
        type=int,
        default=None,
        help=(
            "Fidelity of the evaluation candidates "
            "(default: the surrogate's target_fidelity, else 1)."
        ),
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Rebuild the histograms from existing CSVs without fitting.",
    )
    return parser.parse_known_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the study from the command line."""
    args, config_args = _parse_args(argv)
    out_dir = args.output_dir / args.train_label
    if args.plot_only:
        plot_outputs(out_dir, args.eval_csv, args.train_label)
        return
    if not config_args:
        raise SystemExit("At least one config file path must be provided.")

    from activelearning.main import process_arguments
    from activelearning.runtime import bind_runtime_context
    from activelearning.utils.seeding import set_global_seed
    from activelearning.utils.types import filter_finite_target_observations

    _, cfg, _, _ = process_arguments(config_args)
    set_global_seed(cfg.runtime.seed)
    dataset = cfg.dataset.build()
    surrogate = cfg.surrogate.build()
    acquisition = cfg.acquisition.build()
    runtime_context = cfg.runtime.build(logger=None)
    bind_runtime_context([dataset, surrogate, acquisition], runtime_context)

    fidelity = args.fidelity
    if fidelity is None:
        target_fidelity = getattr(cfg.surrogate, "target_fidelity", None)
        fidelity = 1 if target_fidelity is None else target_fidelity

    observations = filter_finite_target_observations(
        dataset.get_observations_iterable()
    )
    run_study(
        surrogate,
        acquisition,
        observations,
        args.eval_csv,
        out_dir,
        train_label=args.train_label,
        fidelity=fidelity,
        chunk_size=args.chunk_size,
    )
    print("Done.", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
