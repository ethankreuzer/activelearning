"""Find out why scripts/append_gp_mean.py's restore check fails.

Job 24105437 restored the reward-shape surrogate and predicted the first 256
molecules of the fit's ``eval/val_set.csv``. The result was up to 0.084 away from
the means and stds that fit wrote. There are two candidate causes, and this
separates them:

1. Features. The fit read val_set fingerprints from
   ``cache/ampc/exact_dkl_top_n/val_set.npy``; the append script encodes them
   live with MiniMol on cuda. If the two differ, so do the predictions.
2. The restored GP. If the GP predicts differently even on the cached
   fingerprints the fit used, the restore itself is wrong.

It predicts the same molecules from the cached fingerprints (twice, which also
tests determinism) and from live encoding, reports the mean and std errors
separately for each, compares the fingerprints directly, and prints the restored
hyperparameters next to fit_summary.json. It writes nothing.

    python -m scripts.diagnose_gp_restore [--rows 256]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np

from scripts.append_gp_mean import CONFIGS, FITTED_DIR

_logger = logging.getLogger(__name__)

CACHE_DIR = Path("cache/ampc/exact_dkl_top_n")


def _report(name, got, expected):
    error = np.abs(got - expected)
    _logger.info(
        "%-28s max %.3g  mean %.3g  median %.3g  (rows > 1e-4: %d/%d)",
        name,
        float(error.max()),
        float(error.mean()),
        float(np.median(error)),
        int((error > 1e-4).sum()),
        len(error),
    )


def _predict(surrogate, features, *, observation_noise, chunk):
    prediction = surrogate.predict_encoded(
        features, observation_noise=observation_noise, chunk_size=chunk
    )
    return prediction["mean"].numpy(), prediction["std"].numpy()


def _hyperparameters(surrogate):
    model = surrogate.get_model()
    values = {}
    for name, getter in {
        "noise": lambda: model.likelihood.noise,
        "mean_constant": lambda: model.mean_module.constant,
        "outputscale": lambda: model.covar_module.base_kernel.outputscale,
        "input_scale": lambda: model.covar_module.encoder.input_scale,
        "y_mean": lambda: model.outcome_transform.means,
        "y_std": lambda: model.outcome_transform.stdvs,
    }.items():
        try:
            values[name] = float(getter().detach().reshape(-1)[0])
        except (AttributeError, IndexError) as error:
            values[name] = f"unavailable ({error})"
    return values


def main(argv=None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--predict-chunk", type=int, default=64)
    args = parser.parse_args(argv)

    with (FITTED_DIR / "eval" / "val_set.csv").open(newline="") as handle:
        check = [row for _, row in zip(range(args.rows), csv.DictReader(handle))]
    with (CACHE_DIR / "val_set.csv").open(newline="") as handle:
        cached_smiles = [
            row["SMILES"] for _, row in zip(range(args.rows), csv.DictReader(handle))
        ]
    smiles = [row["SMILES"] for row in check]
    if smiles != cached_smiles:
        raise SystemExit("eval/val_set.csv and the cache's val_set.csv rows differ.")
    expected_mean = np.array([float(row["mean"]) for row in check])
    expected_total = np.array([float(row["std_total"]) for row in check])
    expected_latent = np.array([float(row["std_latent"]) for row in check])
    cached = np.load(CACHE_DIR / "val_set.npy", mmap_mode="r")[: args.rows]
    cached = np.array(cached, dtype=np.float32)

    import torch

    from activelearning.main import process_arguments
    from activelearning.runtime import bind_runtime_context
    from activelearning.utils.seeding import set_global_seed
    from activelearning.utils.types import Candidate

    _, cfg, _, _ = process_arguments(list(CONFIGS))
    set_global_seed(cfg.runtime.seed)
    dataset = cfg.dataset.build()
    surrogate = cfg.surrogate.build()
    bind_runtime_context([dataset, surrogate], cfg.runtime.build(logger=None))
    observations = list(dataset.get_observations_iterable())
    _logger.info("Restoring on %d observations.", len(observations))
    surrogate.fit(observations)

    fidelity = int(observations[0].fidelity)
    with torch.no_grad():
        live = surrogate.encode_candidates(
            [Candidate(x=s, fidelity=fidelity) for s in smiles]
        )
    live_np = live.detach().cpu().numpy()
    _logger.info(
        "Feature widths: cached %s, live %s.", cached.shape, tuple(live_np.shape)
    )

    _logger.info("--- 1. Fingerprints: live MiniMol vs val_set.npy")
    if live_np.shape == cached.shape:
        per_row = np.abs(live_np - cached).max(axis=1)
        _logger.info(
            "max |live - cached| %.3g; rows differing by > 1e-4: %d/%d; "
            "typical |cached| %.3g; all-zero live rows (unfeaturizable): %d.",
            float(per_row.max()),
            int((per_row > 1e-4).sum()),
            len(per_row),
            float(np.abs(cached).mean()),
            int((np.abs(live_np).max(axis=1) == 0).sum()),
        )
    else:
        _logger.info("Shapes differ, so the fingerprints cannot be compared.")

    cached_t = torch.from_numpy(cached)
    for label, features in (("cached", cached_t), ("live", live)):
        _logger.info("--- 2. Predictions from %s features vs the fit's", label)
        mean, total = _predict(
            surrogate, features, observation_noise=True, chunk=args.predict_chunk
        )
        _, latent = _predict(
            surrogate, features, observation_noise=False, chunk=args.predict_chunk
        )
        _report(f"{label}: mean", mean, expected_mean)
        _report(f"{label}: std_total", total, expected_total)
        _report(f"{label}: std_latent", latent, expected_latent)
        if label == "cached":
            again, _ = _predict(
                surrogate, features, observation_noise=True, chunk=args.predict_chunk
            )
            _report("cached: mean, rerun vs run", again, mean)
            whole, _ = _predict(
                surrogate, features, observation_noise=True, chunk=None
            )
            _report("cached: mean, one chunk", whole, mean)

    _logger.info("--- 3. Restored hyperparameters vs fit_summary.json")
    fitted = json.loads((FITTED_DIR / "fit_summary.json").read_text())[
        "hyperparameters"
    ]
    for name, value in _hyperparameters(surrogate).items():
        _logger.info("%-14s restored %s  fit %s", name, value, fitted.get(name, "-"))


if __name__ == "__main__":
    main()
