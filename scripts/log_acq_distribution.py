"""Score the 300,000 SVGP training molecules and record the log GIBBON distribution.

Every reward-shape arm in ``ampc-reward-shape-svgp`` picks a ``log_score_floor``
against a distribution that was only ever measured on the 83,811 molecules of
``gp_molformer_set`` (REWARD_SHAPE_UNDERFLOW_GAP.md section 3: median -2,688,
p99.9 -752.6). The floor, the underflow gap at -744.4 and any threshold a seeded
replay buffer would use are all quantiles of this distribution, so it is worth
measuring directly on the set the surrogate was actually fitted to.

The surrogate is RESTORED exactly as a run restores it -- same config, same
``surrogate_state.pt``, same stratified max-value support -- so the numbers are
comparable with the arms rather than with a refit.

``acquisition.log_output=true`` IS REQUIRED and the script refuses to run without
it. The arms run ``log_output=false``, which returns ``exp(L)``; in float64 that
is exactly ``0.0`` for every molecule below ``ln(5e-324) = -744.44``, which is
roughly 77% of a pool. Taking a logarithm back gives ``-inf`` and the bulk of the
distribution -- the part the floor is chosen to compress -- cannot be seen at
all. With ``log_output=true`` the acquisition returns ``L`` directly and nothing
underflows.

Molecules are scored in a deterministic shuffled order and the CSV is flushed as
it goes, so a job killed at the wall clock still leaves a uniform random
subsample rather than a biased prefix of whatever order the training file holds.

Needs a GPU node (the config asks for cuda); never run it on a login node.
Usage::

    python scripts/log_acq_distribution.py config/ampc/reward_shape_svgp300k.yaml \\
        acquisition.log_output=true \\
        --out outputs/ampc/log_acq_distribution/svgp300k.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

#: Candidates per acquisition call, matching the config's score_chunk_size.
SCORE_CHUNK = 256
#: Rows per posterior call when reading the GP's latent mean and std.
PREDICT_CHUNK = 4096
#: Natural log of the smallest positive float64 subnormal. Below this an
#: ``exp(L)`` acquisition returns exactly 0.0 and the reward transform floors it,
#: which is the discontinuity REWARD_SHAPE_UNDERFLOW_GAP.md is about.
FLOAT64_UNDERFLOW = -744.4400719213812
#: Floors the study has run, plus the underflow edge, reported as tail fractions.
REPORT_THRESHOLDS = (-150.0, -250.0, -500.0, -744.4400719213812, -875.0, -1000.0)


def parse_args(argv: Sequence[str] | None) -> tuple[argparse.Namespace, list[str]]:
    """Split our own flags from the config paths and Hydra-style overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--n",
        type=int,
        default=0,
        help="Molecules to score. 0, the default, scores every observation.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Chunks between progress lines. Each line re-estimates the ETA.",
    )
    args, config_args = parser.parse_known_args(argv)
    if not any("=" not in arg for arg in config_args):
        raise SystemExit("At least one config file path must be provided.")
    return args, config_args


def latent_predictions(
    surrogate: Any,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the latent mean and std of the fitted training rows.

    The acquisition reads the latent posterior, not the noisy one, so
    ``observation_noise=False`` is what makes these numbers explain the
    acquisition. The encoded rows are taken from the fitted surrogate rather
    than re-encoded, which keeps a 300,000-row read off the encoder.

    Parameters
    ----------
    surrogate : Any
        A fitted surrogate exposing ``get_encoded_train_data`` and
        ``predict_encoded``.
    count : int
        Number of leading rows to predict.

    Returns
    -------
    tuple of numpy.ndarray
        Latent means and standard deviations, both shaped ``(count,)``.
    """
    features = surrogate.get_encoded_train_data()[:count]
    prediction = surrogate.predict_encoded(
        features,
        observation_noise=False,
        chunk_size=PREDICT_CHUNK,
    )
    mean = prediction["mean"].detach().cpu().numpy().astype(np.float64).ravel()
    std = prediction["std"].detach().cpu().numpy().astype(np.float64).ravel()
    return mean, std


def summarize(log_acq: np.ndarray) -> dict[str, Any]:
    """Build the quantile and tail-fraction summary of a log-acquisition sample.

    Parameters
    ----------
    log_acq : numpy.ndarray
        Log-scale acquisition values, one per scored molecule.

    Returns
    -------
    dict
        Quantiles in nats, the fraction above each reported threshold, and the
        fraction an ``exp(L)`` acquisition would underflow to exactly zero.
    """
    finite = log_acq[np.isfinite(log_acq)]
    quantiles = (0.0, 0.1, 1.0, 10.0, 50.0, 90.0, 99.0, 99.9, 100.0)
    return {
        "n_scored": int(log_acq.size),
        "n_nonfinite": int(log_acq.size - finite.size),
        "quantiles_nats": {
            f"p{q:g}": float(np.percentile(finite, q)) for q in quantiles
        },
        "mean_nats": float(finite.mean()),
        # The fraction a floor at this depth would leave graded. A floor that
        # ties everything is the degenerate reward of section 1.
        "fraction_above": {
            f"{threshold:g}": float((finite > threshold).mean())
            for threshold in REPORT_THRESHOLDS
        },
        # What the arms actually see: log_output=false returns exp(L), so this
        # fraction of molecules arrives at the reward transform as exactly 0.0.
        "underflow_fraction": float((finite <= FLOAT64_UNDERFLOW).mean()),
    }


def main(argv: Sequence[str] | None = None) -> None:
    """Restore the surrogate, score the training set and write the artifacts."""
    args, config_args = parse_args(argv)

    from activelearning.main import process_arguments
    from activelearning.runtime import bind_runtime_context
    from activelearning.utils.seeding import set_global_seed
    from activelearning.utils.types import Candidate, filter_finite_target_observations

    raw_cfg, cfg, config_paths, _ = process_arguments(config_args)
    del raw_cfg

    # Encoding anything before the training data would publish that small set as
    # the feature cache, as surrogate_eval_fit.py warns. Refuse rather than
    # corrupt it.
    cache_path = getattr(
        getattr(getattr(cfg, "surrogate", None), "encoder", None),
        "feature_cache_path",
        None,
    )
    if cache_path is not None and not Path(cache_path).exists():
        raise SystemExit(
            f"The encoder feature cache {cache_path} does not exist; build it first."
        )

    set_global_seed(cfg.runtime.seed)
    dataset = cfg.dataset.build()
    surrogate = cfg.surrogate.build()
    acquisition = cfg.acquisition.build()
    runtime_context = cfg.runtime.build(logger=None)
    bind_runtime_context([dataset, surrogate, acquisition], runtime_context)

    # Checked before the restore, which costs minutes: on the value scale the
    # bulk of this distribution is unmeasurable, so there is nothing to record.
    if getattr(acquisition, "score_scale", "value") != "log":
        raise SystemExit(
            "This script needs log-scale scores: pass acquisition.log_output=true.\n"
            "With log_output=false the acquisition returns exp(L), which is exactly "
            "0.0 below -744.44 nats -- about 77% of this set -- so the distribution "
            "below the underflow edge cannot be measured."
        )

    observations = list(
        filter_finite_target_observations(dataset.get_observations_iterable())
    )
    print(f"Loaded {len(observations)} finite observations.", flush=True)

    # fit() is what triggers the state restore; it must run before anything else
    # is encoded so the training rows reach the cache first.
    set_global_seed(cfg.runtime.seed)
    surrogate.fit(observations)
    set_global_seed(cfg.runtime.seed)
    acquisition.update(surrogate, observations)
    fallback_active = bool(getattr(acquisition, "_fallback_active", False))
    acqf_built = getattr(acquisition, "_botorch_acqf", None) is not None
    print(
        f"fallback_active={fallback_active} botorch_acqf_built={acqf_built}",
        flush=True,
    )
    if fallback_active or not acqf_built:
        raise SystemExit(
            "The acquisition fell back or was never built, so its scores would be "
            "degenerate. Fix that before reading a distribution off them."
        )

    mean, std = latent_predictions(surrogate, len(observations))

    # A deterministic shuffle, so a job that dies at the wall clock leaves a
    # uniform random subsample instead of a prefix of the training file's own
    # order. row_index keeps every row traceable back to that file.
    order = np.random.default_rng(cfg.runtime.seed).permutation(len(observations))
    if args.n:
        order = order[: args.n]
    print(f"Scoring {order.size} of {len(observations)} molecules.", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    scored: list[float] = []
    n_chunks = int(np.ceil(order.size / SCORE_CHUNK))
    started = time.perf_counter()
    with args.out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["row_index", "smiles", "y", "gp_latent_mean", "gp_latent_std", "log_acq"]
        )
        for chunk_number, start in enumerate(range(0, order.size, SCORE_CHUNK), 1):
            indices = order[start : start + SCORE_CHUNK]
            candidates = [
                Candidate(x=str(observations[index].x), fidelity=None)
                for index in indices
            ]
            values = acquisition.score(candidates)
            scored.extend(float(value) for value in values)
            writer.writerows(
                [
                    int(index),
                    str(observations[index].x),
                    float(observations[index].y),
                    float(mean[index]),
                    float(std[index]),
                    float(value),
                ]
                for index, value in zip(indices, values)
            )
            # Flushed per chunk: the whole point of the shuffle is that a killed
            # job leaves usable rows on disk.
            handle.flush()
            if chunk_number % args.progress_every == 0 or chunk_number == n_chunks:
                elapsed = time.perf_counter() - started
                rate = len(scored) / elapsed
                remaining = (order.size - len(scored)) / rate if rate else float("nan")
                print(
                    f"chunk {chunk_number}/{n_chunks}: {len(scored)} scored, "
                    f"{rate:.1f} molecules/s, {elapsed / 60:.1f} min elapsed, "
                    f"~{remaining / 60:.1f} min left, "
                    f"running median {np.median(scored):.1f} nats.",
                    flush=True,
                )

    log_acq = np.asarray(scored, dtype=np.float64)
    import torch

    summary = {
        "host": platform.node(),
        "config_paths": [str(p) for p in config_paths],
        "n_observations": len(observations),
        "seed": int(cfg.runtime.seed),
        "torch_version": torch.__version__,
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "python": sys.version.split()[0],
        "fallback_active": fallback_active,
        "botorch_acqf_built": acqf_built,
        "scoring_minutes": (time.perf_counter() - started) / 60.0,
        "gp_latent_std": [float(std.min()), float(std.mean()), float(std.max())],
        "log_acq": summarize(log_acq),
    }
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f"\nWrote {args.out} and {summary_path}", flush=True)


if __name__ == "__main__":
    main()
