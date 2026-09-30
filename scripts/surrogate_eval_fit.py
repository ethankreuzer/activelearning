"""Fit the AmpC surrogate once and save everything the evaluation needs.

Step 4 of ``SURROGATE_EVAL_PLAN.md``. The script builds the dataset and surrogate
from a config exactly as ``activelearning.main`` does, fits the surrogate on the
whole initial dataset, and writes to ``--output-dir``:

- ``surrogate_state.pt``: the fitted GP state (inducing points, variational
  distribution, kernel, noise, output standardization);
- ``train_random.csv`` and ``train_top.csv``: the two ``train`` evaluation sets
  (seeded random rows, and the rows with the highest target), written before the
  fit so a failed fit still leaves them;
- ``resolved_config.json``: the merged config that was run;
- ``fit_summary.json``: fit time, data size and the learned hyperparameters.

The objective and every other setting are config overrides, so an ELBO and a PLL
arm differ only in one argument::

    python scripts/surrogate_eval_fit.py \\
        config/ampc/s3gfn_minimol_ampc_variational_single_fidelity.yaml \\
        surrogate.training_params.variational_objective=VariationalELBO \\
        --output-dir outputs/ampc/surrogate_eval/elbo

Run it on a whole GPU node through ``jobs/surrogate_eval_fit.sh``, never on the
login node. Pass ``--wandb-project`` to also log the summary to W&B (run with
``WANDB_MODE=offline`` on compute nodes and sync afterwards).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

STATE_FILE = "surrogate_state.pt"
SUMMARY_FILE = "fit_summary.json"
CONFIG_FILE = "resolved_config.json"
TRAIN_RANDOM_FILE = "train_random.csv"
TRAIN_TOP_FILE = "train_top.csv"
EVAL_CSV_COLUMNS = ("SMILE", "y")


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


def log_to_wandb(
    project: str,
    run_name: str,
    config: Mapping[str, Any],
    metrics: Mapping[str, float],
) -> None:
    """Log the run configuration and scalar metrics to one W&B run.

    This is the only place the script talks to W&B, so what is tracked can be
    changed here. The repo's ``WandbLogger`` buffers metrics until ``log_step``.

    Parameters
    ----------
    project : str
        W&B project name.
    run_name : str
        W&B run name.
    config : Mapping[str, Any]
        Run configuration to attach to the run.
    metrics : Mapping[str, float]
        Scalar metrics to log.
    """
    from activelearning.logger.logger import WandbLogger

    logger = WandbLogger(project_name=project, run_name=run_name)
    logger.log_config(dict(config))
    for key, value in metrics.items():
        logger.log_metric(key, value)
    logger.log_step(0)
    logger.end()


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
        "--wandb-project",
        default=None,
        help="Also log the fit summary to this W&B project.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="W&B run name. Defaults to the output directory name.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing fitted state instead of refusing to run.",
    )
    args, config_args = parser.parse_known_args(argv)
    if args.n_train_random < 0 or args.n_train_top < 0:
        parser.error("--n-train-random and --n-train-top must be nonnegative.")
    return args, config_args


def main(argv: Sequence[str] | None = None) -> None:
    """Fit the surrogate and save its state from the command line."""
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
    set_global_seed(cfg.runtime.seed)
    dataset = cfg.dataset.build()
    surrogate = cfg.surrogate.build()
    runtime_context = cfg.runtime.build(logger=None)
    bind_runtime_context([dataset, surrogate], runtime_context)

    observations = list(
        filter_finite_target_observations(dataset.get_observations_iterable())
    )
    print(f"Loaded {len(observations)} finite observations.", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    resolved = OmegaConf.to_container(raw_cfg, resolve=True)
    assert isinstance(resolved, dict)
    resolved.pop("logger", None)
    resolved["config_paths"] = [str(path) for path in config_paths]
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

    print(f"Fitting surrogate on {len(observations)} observations.", flush=True)
    started = time.perf_counter()
    surrogate.fit(observations)
    fit_seconds = time.perf_counter() - started
    print(f"Surrogate fit in {fit_seconds:.0f} s.", flush=True)

    state = surrogate.get_state_dict()
    if state is None:
        raise SystemExit("The fitted surrogate has no state to save.")
    import torch

    torch.save(state, out_dir / STATE_FILE)
    print(f"Saved surrogate state to {out_dir / STATE_FILE}", flush=True)

    training_params = resolved.get("surrogate", {}).get("training_params", {})
    summary = {
        "n_observations": len(observations),
        "fit_seconds": fit_seconds,
        "objective": training_params.get("variational_objective"),
        "num_inducing": resolved.get("surrogate", {}).get("num_inducing"),
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

    if args.wandb_project is not None:
        metrics = {
            "fit/seconds": fit_seconds,
            "fit/n_observations": float(len(observations)),
            **{
                f"hyperparameters/{name}": value
                for name, value in summary["hyperparameters"].items()
            },
        }
        log_to_wandb(
            args.wandb_project,
            args.run_name or out_dir.name,
            {"fit_summary": summary, "resolved_config": resolved},
            metrics,
        )
    print("Done.", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
