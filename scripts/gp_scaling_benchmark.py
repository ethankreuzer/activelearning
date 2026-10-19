"""Measure how the fixed-feature exact GP scales with its training-set size.

Fits the surrogate chosen for the active-learning experiments -- isotropic
Matern-5/2 in a ScaleKernel, mean pinned at the AmpC library mean, float32,
L-BFGS-B on an exact Cholesky -- on ``--n`` real molecules, then times the two
ways the loop uses it:

- **acquisition update**: GIBBON's max-value sampling over the training set,
  which also builds the GP's prediction caches;
- **pool scoring**: ``acquisition.score`` on generated molecules in chunks of
  64, as the selector scores the 100k pool and S3-GFN scores its reward batches.

The training set is ``balanced_25000`` plus rows drawn at random from
``balanced_pool`` (real fingerprints and scores, not in the 25k). Candidates are
the first ``--n-score`` rows of ``gp_molformer_set``.

Features come from the per-set caches through a lookup encoder, so MiniMol
inference is excluded: it does not depend on ``n``, and the 25k reward-shape run
already measured it as part of its selection time.

Each stage records wall time, peak CUDA memory allocated and reserved, and the
memory still allocated after it. A CUDA out-of-memory error ends the run and is
recorded as the stage's outcome rather than raised, so one size running out of
memory still leaves a result.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from activelearning.acquisition.config import QLowerBoundMaxValueEntropyConfig
from activelearning.runtime import RuntimeContext, bind_runtime_context
from activelearning.surrogate.encoder import FixedEncoder
from activelearning.surrogate.fixed_feature_gp import FixedFeatureGPSurrogate
from activelearning.utils.types import Candidate, Observation

CACHE_DIR = Path("cache/ampc/exact_dkl_top_n")
#: Mean y of data/10M_unif_random_subset.csv (n = 10,000,000).
LIBRARY_MEAN = 0.03956042999405787
BASE_SET = "balanced_25000"
EXTRA_SET = "balanced_pool"
CANDIDATE_SET = "gp_molformer_set"
GIB = 2**30


class _LookupEncoder(FixedEncoder):
    """Return cached fingerprints for SMILES seen at construction."""

    feature_dim = 512

    def __init__(self, smiles: Sequence[str], features: np.ndarray) -> None:
        self._rows = {value: index for index, value in enumerate(smiles)}
        self._features = torch.from_numpy(np.ascontiguousarray(features))

    def encode(self, values: Sequence[Any], *, device: torch.device) -> Tensor:
        """Gather the rows for ``values``; a miss raises ``KeyError``."""
        index = torch.tensor([self._rows[value] for value in values])
        return self._features[index].to(device)


def _load_set(name: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Return a cached set's CSV and its row-aligned fingerprints."""
    frame = pd.read_csv(CACHE_DIR / f"{name}.csv")
    features = np.load(CACHE_DIR / f"{name}.npy", mmap_mode="r")
    if features.shape != (len(frame), 512):
        raise SystemExit(f"{name}: {features.shape} features for {len(frame)} rows.")
    return frame, features


def _training_set(n: int, seed: int) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return ``n`` SMILES, scores and fingerprints: the 25k plus pool rows."""
    base, base_features = _load_set(BASE_SET)
    if n < len(base):
        raise SystemExit(f"--n must be at least {len(base)}.")
    pool, pool_features = _load_set(EXTRA_SET)
    eligible = np.flatnonzero(
        np.isfinite(pool["y"].to_numpy()) & ~pool["SMILES"].isin(set(base["SMILES"]))
    )
    # Drop repeated SMILES within the pool so every training row is distinct.
    eligible = eligible[~pool["SMILES"].iloc[eligible].duplicated().to_numpy()]
    extra = np.sort(
        np.random.default_rng(seed).choice(eligible, n - len(base), replace=False)
    )
    smiles = base["SMILES"].tolist() + pool["SMILES"].iloc[extra].tolist()
    y = np.concatenate([base["y"].to_numpy(), pool["y"].iloc[extra].to_numpy()])
    features = np.concatenate([base_features, pool_features[extra]])
    return smiles, y, features


def _stage(
    results: dict[str, Any], name: str, work: Callable[[], Any]
) -> Any:
    """Run ``work`` as one measured stage; return its value, or None on OOM."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    try:
        value = work()
        outcome = "ok"
    except torch.OutOfMemoryError as error:
        value = None
        outcome = f"oom: {str(error).splitlines()[0]}"
    torch.cuda.synchronize()
    results["stages"][name] = {
        "outcome": outcome,
        "seconds": time.perf_counter() - start,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / GIB,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / GIB,
        "allocated_after_gib": torch.cuda.memory_allocated() / GIB,
    }
    print(name, json.dumps(results["stages"][name]), flush=True)
    return value


def _hyperparameters(surrogate: FixedFeatureGPSurrogate) -> dict[str, float]:
    """Return the fitted scalar hyperparameters in standardized units."""
    model = surrogate.get_model()
    return {
        "lengthscale": float(model.covar_module.base_kernel.lengthscale.item()),
        "outputscale": float(model.covar_module.outputscale.item()),
        "noise": float(model.likelihood.noise.item()),
    }


def main(argv: Sequence[str] | None = None) -> None:
    """Fit at one training-set size, time the loop's uses, write a JSON."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, required=True, help="Training-set size.")
    parser.add_argument("--n-score", type=int, default=10_000)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--reward-calls", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    smiles, y, features = _training_set(args.n, args.seed)
    candidates_frame, candidate_features = _load_set(CANDIDATE_SET)
    candidate_smiles = candidates_frame["SMILES"].iloc[: args.n_score].tolist()
    encoder = _LookupEncoder(
        smiles + candidate_smiles,
        np.concatenate([features, candidate_features[: args.n_score]]),
    )
    observations = [Observation(x=s, y=float(v)) for s, v in zip(smiles, y)]
    candidates = [Candidate(x=s) for s in candidate_smiles]

    surrogate = FixedFeatureGPSurrogate(
        encoder,
        ard=False,
        kernel="matern",
        outputscale=True,
        prior_mean=LIBRARY_MEAN,
        # At or above n, so the fit's objective is an exact Cholesky at every size.
        max_cholesky_size=args.n,
    )
    acquisition = QLowerBoundMaxValueEntropyConfig(
        candidate_set_spec={
            "type": "TrainDataCandidateSetSpec",
            "fallback_size": 100_000,
            "seed": args.seed,
        },
        num_mv_samples=10,
        maximize=True,
        log_space=True,
        log_output=False,
        score_chunk_size=args.chunk_size,
    ).build()
    runtime = RuntimeContext(
        device=torch.device("cuda"), dtype=torch.float32, seed=args.seed
    )
    bind_runtime_context([surrogate, acquisition], runtime)

    results: dict[str, Any] = {
        "n": args.n,
        "n_score": args.n_score,
        "chunk_size": args.chunk_size,
        "gpu": torch.cuda.get_device_name(),
        "gpu_total_gib": torch.cuda.get_device_properties(0).total_memory / GIB,
        "train_y_mean": float(y.mean()),
        "stages": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def write() -> None:
        args.output.write_text(json.dumps(results, indent=2) + "\n")

    _stage(results, "fit", lambda: surrogate.fit(observations))
    ok = results["stages"]["fit"]["outcome"] == "ok"
    if ok:
        results["hyperparameters"] = _hyperparameters(surrogate)
        _stage(
            results,
            "acquisition_update",
            lambda: acquisition.update(surrogate, observations),
        )
        ok = results["stages"]["acquisition_update"]["outcome"] == "ok"
    if ok:
        _stage(results, "pool_score", lambda: acquisition.score(candidates))
        ok = results["stages"]["pool_score"]["outcome"] == "ok"
    if ok:
        seconds = results["stages"]["pool_score"]["seconds"]
        results["pool_score_seconds_per_100k"] = seconds * 100_000 / args.n_score
        batch = candidates[: args.chunk_size]
        acquisition.score(batch)  # warm-up

        def reward_calls() -> list[float]:
            latencies = []
            for _ in range(args.reward_calls):
                torch.cuda.synchronize()
                start = time.perf_counter()
                acquisition.score(batch)
                torch.cuda.synchronize()
                latencies.append(time.perf_counter() - start)
            return latencies

        latencies = _stage(results, "reward_batches", reward_calls)
        if results["stages"]["reward_batches"]["outcome"] == "ok":
            results["reward_batch_ms_median"] = 1000 * statistics.median(latencies)
            results["reward_batch_ms_p90"] = 1000 * float(
                np.percentile(latencies, 90)
            )
    write()
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
