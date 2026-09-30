"""Generate a fixed sample of molecules from the untrained GP-MoLFormer prior.

The sample is the starting distribution of S3-GFN: the molecules its policy
proposes before any acquisition-guided training. It is used to evaluate the
surrogate, acquisition and reward on generated molecules (see
``SURROGATE_EVAL_PLAN.md``, step 1).

Generation goes through the configured ``S3GFNSampler`` itself with
``n_train_steps=0``, so the sample passes exactly the same pipeline as a real
round's candidate pool: RDKit canonicalization (non-isomeric), rejection of
invalid, unterminated and disconnected molecules, and deduplication. That pool
is not SA-filtered (the SA threshold only splits training molecules into replay
buffers), so neither is this sample; the SA score is recorded as a column.

Usage (on a GPU compute node, never the login node)::

    python scripts/generate_prior_sample.py \\
        config/ampc/s3gfn_minimol_ampc_variational_single_fidelity.yaml \\
        --n-samples 100000 --seed 42 \\
        --output data/gpmolformer_prior_100k.csv

Extra ``key=value`` arguments are OmegaConf overrides applied to the config.
Writes the CSV and a ``<output>.json`` record of the settings and generation
counts next to it.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CSV_COLUMNS = ("SMILES", "sa_score", "passes_sa")


def write_sample(
    output: Path,
    smiles: Sequence[str],
    sa_scores: Sequence[float],
    passes_sa: Sequence[bool],
    record: dict[str, Any],
) -> Path:
    """Write the sample CSV and its JSON record atomically.

    Parameters
    ----------
    output : Path
        Destination CSV path. The record is written to ``output`` with a
        ``.json`` suffix.
    smiles : Sequence[str]
        Canonical SMILES, in generation order.
    sa_scores : Sequence[float]
        SA score per molecule, aligned with ``smiles``.
    passes_sa : Sequence[bool]
        Whether each molecule passes the sampler's SA threshold.
    record : dict[str, Any]
        Settings and generation counts to store alongside the sample.

    Returns
    -------
    Path
        Path of the written JSON record.

    Raises
    ------
    ValueError
        If ``smiles``, ``sa_scores`` and ``passes_sa`` differ in length.
    """
    if not len(smiles) == len(sa_scores) == len(passes_sa):
        raise ValueError("smiles, sa_scores and passes_sa must align.")
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output.with_name(output.name + ".tmp")
    with tmp_output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for molecule, sa_score, passes in zip(smiles, sa_scores, passes_sa):
            writer.writerow([molecule, f"{sa_score:.6g}", int(passes)])
    os.replace(tmp_output, output)

    record_path = output.with_suffix(".json")
    tmp_record = record_path.with_name(record_path.name + ".tmp")
    tmp_record.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    os.replace(tmp_record, record_path)
    return record_path


def _parse_args(argv: Sequence[str] | None) -> tuple[argparse.Namespace, list[str]]:
    """Split script options from config paths and OmegaConf overrides."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n-samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/gpmolformer_prior_100k.csv"),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file instead of refusing to run.",
    )
    args, config_args = parser.parse_known_args(argv)
    if args.n_samples < 1:
        parser.error("--n-samples must be positive.")
    return args, config_args


def main(argv: Sequence[str] | None = None) -> None:
    """Generate the prior sample from the command line."""
    args, config_args = _parse_args(argv)
    if not any("=" not in arg for arg in config_args):
        raise SystemExit("At least one config file path must be provided.")
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"{args.output} exists; pass --overwrite to replace it.")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from activelearning.main import process_arguments
    from activelearning.runtime import bind_runtime_context
    from activelearning.sampler.s3gfn import (
        SAScoreSynthesizability,
        passes_sa_threshold,
    )
    from activelearning.utils.seeding import set_global_seed

    overrides = [
        "sampler.n_train_steps=0",
        f"sampler.n_samples={args.n_samples}",
        f"sampler.seed={args.seed}",
    ]
    _, cfg, config_paths, _ = process_arguments([*config_args, *overrides])
    if cfg.sampler.type != "S3GFNSampler":
        raise SystemExit(f"Expected an S3GFNSampler config, got {cfg.sampler.type}.")

    set_global_seed(args.seed)
    sampler = cfg.sampler.build()
    runtime_context = cfg.runtime.build(logger=None)
    bind_runtime_context([sampler], runtime_context)

    started = time.perf_counter()
    candidates = sampler.sample()
    generation_s = time.perf_counter() - started
    smiles = [candidate.x for candidate in candidates]

    started = time.perf_counter()
    synthesizability = SAScoreSynthesizability(threshold=sampler.sa_threshold)
    sa_scores = synthesizability.score_batch(smiles)
    passes_sa = [
        passes_sa_threshold(score, sampler.sa_threshold) for score in sa_scores
    ]
    sa_s = time.perf_counter() - started

    metrics = sampler.round_metrics
    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "argv": sys.argv,
        "config_paths": [str(path) for path in config_paths],
        "seed": args.seed,
        "n_samples": len(smiles),
        "model_name_or_path": sampler.model_name_or_path,
        "tokenizer_name_or_path": sampler.tokenizer_name_or_path,
        "model_dtype": str(sampler.effective_model_dtype),
        "max_length": sampler.max_length,
        "sampling_temperature": sampler.sampling_temperature,
        "sa_threshold": sampler.sa_threshold,
        "generation_attempts": metrics.generation_attempts,
        "generation_invalid": metrics.generation_invalid,
        "generation_duplicates": metrics.generation_duplicates,
        "n_passes_sa": sum(passes_sa),
        "generation_s": round(generation_s, 1),
        "sa_scoring_s": round(sa_s, 1),
    }
    record_path = write_sample(
        args.output,
        smiles,
        sa_scores,
        passes_sa,
        record,
    )
    logging.info(
        "Wrote %d molecules to %s (record: %s). attempts=%d invalid=%d "
        "duplicates=%d passes_sa=%d.",
        len(smiles),
        args.output,
        record_path,
        record["generation_attempts"],
        record["generation_invalid"],
        record["generation_duplicates"],
        record["n_passes_sa"],
    )


if __name__ == "__main__":
    main()
