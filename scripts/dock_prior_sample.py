"""Dock one shard of the GP-MoLFormer prior sample, or merge the shards.

Step 2 of ``SURROGATE_EVAL_PLAN.md``. Docking the 100k prior molecules is split
into ``--num-shards`` independent shards so a Slurm job array can dock them in
parallel on CPU-only nodes. Shard ``i`` takes rows ``i, i + N, i + 2N, ...`` of the
input CSV (interleaved, so molecules of different cost spread evenly) and docks
them in chunks with the configured ``Dock3Oracle``. Every finished chunk is
appended to ``shard_XXXX.csv`` and flushed, and a restarted shard skips the
molecules already in its file, so a task that hits its wall-clock limit loses at
most one chunk.

Dock a shard (through ``jobs/dock_prior_sample.sh``, never on the login node)::

    python scripts/dock_prior_sample.py \\
        config/ampc/s3gfn_minimol_ampc_variational_single_fidelity.yaml \\
        runtime.device=cpu \\
        --input data/gpmolformer_prior_100k.csv \\
        --output-dir data/gpmolformer_prior_docking \\
        --shard 0 --num-shards 40

Merge the shards into one CSV in the input's row order (a small job,
``jobs/merge_dock_prior_shards.sh``)::

    python scripts/dock_prior_sample.py --merge \\
        --input data/gpmolformer_prior_100k.csv \\
        --output-dir data/gpmolformer_prior_docking \\
        --num-shards 40 \\
        --merged-output data/gpmolformer_prior_100k_docked.csv

The oracle settings come from the config. Extra ``key=value`` arguments are
OmegaConf overrides. Failed dockings keep a ``NaN`` ``y`` and their failure
reason, so the failure rate can be reported.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import sys
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SHARD_COLUMNS = ("SMILES", "y", "raw_score", "pprop", "failure_reason")
MERGED_COLUMNS = ("SMILES", "sa_score", "passes_sa", *SHARD_COLUMNS[1:])


def read_input_rows(path: Path) -> list[dict[str, str]]:
    """Read the prior-sample CSV, requiring a ``SMILES`` column.

    Parameters
    ----------
    path : Path
        CSV written by ``scripts/generate_prior_sample.py``.

    Returns
    -------
    list[dict[str, str]]
        One dict per row, in file order.

    Raises
    ------
    ValueError
        If the file has no ``SMILES`` column.
    """
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "SMILES" not in reader.fieldnames:
            raise ValueError(f"{path} has no SMILES column.")
        return list(reader)


def shard_indices(n_rows: int, shard: int, num_shards: int) -> list[int]:
    """Return the input row indices assigned to one shard.

    Parameters
    ----------
    n_rows : int
        Number of input rows.
    shard : int
        Shard number, ``0 <= shard < num_shards``.
    num_shards : int
        Total number of shards.

    Returns
    -------
    list[int]
        Rows ``shard, shard + num_shards, ...`` in ascending order.

    Raises
    ------
    ValueError
        If ``num_shards`` is not positive or ``shard`` is out of range.
    """
    if num_shards < 1:
        raise ValueError("num_shards must be positive.")
    if not 0 <= shard < num_shards:
        raise ValueError(f"shard must be in [0, {num_shards}), got {shard}.")
    return list(range(shard, n_rows, num_shards))


def shard_path(output_dir: Path, shard: int) -> Path:
    """Return the result file of one shard."""
    return output_dir / f"shard_{shard:04d}.csv"


def read_shard(path: Path) -> dict[str, dict[str, str]]:
    """Read a shard result file into a SMILES-keyed dict, empty if absent.

    A line cut short by a killed job is ignored, so the molecule is docked again.

    Parameters
    ----------
    path : Path
        Shard result file.

    Returns
    -------
    dict[str, dict[str, str]]
        Result row per SMILES.
    """
    if not path.exists():
        return {}
    results: dict[str, dict[str, str]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if None in row.values():
                continue
            results[row["SMILES"]] = row
    return results


def _format(value: float) -> str:
    """Format a float, keeping NaN as ``nan``."""
    return "nan" if math.isnan(value) else repr(float(value))


def dock_shard(
    oracle: Any,
    smiles: Sequence[str],
    fidelity: int,
    output_path: Path,
    chunk_size: int,
) -> None:
    """Dock a shard's molecules in chunks, appending each chunk's results.

    Parameters
    ----------
    oracle : Any
        Oracle with ``query(candidates) -> observations``, as ``Dock3Oracle``.
    smiles : Sequence[str]
        Molecules of this shard. Those already in ``output_path`` are skipped.
    fidelity : int
        Fidelity level the oracle declares.
    output_path : Path
        Shard result file, created with a header if absent.
    chunk_size : int
        Molecules per ``oracle.query`` call. Each finished chunk is flushed.
    """
    from activelearning.utils.types import Candidate

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    done = read_shard(output_path)
    todo = [molecule for molecule in smiles if molecule not in done]
    logger.info(
        "%s: %d molecules, %d already docked, %d to do.",
        output_path.name,
        len(smiles),
        len(smiles) - len(todo),
        len(todo),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists() or output_path.stat().st_size == 0
    started = time.perf_counter()
    with output_path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(SHARD_COLUMNS)
        for start in range(0, len(todo), chunk_size):
            chunk = todo[start : start + chunk_size]
            observations = oracle.query(
                [Candidate(x=molecule, fidelity=fidelity) for molecule in chunk]
            )
            for molecule, observation in zip(chunk, observations, strict=True):
                metadata = observation.metadata or {}
                writer.writerow(
                    [
                        molecule,
                        _format(float(observation.y)),
                        _format(float(metadata["dock3_raw_score"])),
                        _format(float(metadata["dock3_pprop"])),
                        metadata.get("dock3_failure_reason") or "",
                    ]
                )
            handle.flush()
            os.fsync(handle.fileno())
            logger.info(
                "%s: %d/%d docked (%.0f s).",
                output_path.name,
                start + len(chunk),
                len(todo),
                time.perf_counter() - started,
            )


def merge_shards(
    rows: Sequence[dict[str, str]],
    output_dir: Path,
    num_shards: int,
    merged_output: Path,
) -> Counter[str]:
    """Merge shard results into one CSV in the input's row order.

    Parameters
    ----------
    rows : Sequence[dict[str, str]]
        Input rows; ``sa_score`` and ``passes_sa`` are carried over if present.
    output_dir : Path
        Directory of the shard files.
    num_shards : int
        Number of shards the docking was split into.
    merged_output : Path
        Merged CSV, written atomically.

    Returns
    -------
    Counter[str]
        Failure reason counts (``"ok"`` for molecules that docked), plus
        ``"missing"`` for molecules absent from their shard.
    """
    results: dict[str, dict[str, str]] = {}
    for shard in range(num_shards):
        results.update(read_shard(shard_path(output_dir, shard)))

    counts: Counter[str] = Counter()
    merged_output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = merged_output.with_name(merged_output.name + ".tmp")
    with tmp_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(MERGED_COLUMNS)
        for row in rows:
            smiles = row["SMILES"]
            result = results.get(smiles)
            if result is None:
                counts["missing"] += 1
                writer.writerow(
                    [smiles, row.get("sa_score", ""), row.get("passes_sa", "")]
                    + ["nan", "nan", "nan", "missing"]
                )
                continue
            counts[result["failure_reason"] or "ok"] += 1
            writer.writerow(
                [smiles, row.get("sa_score", ""), row.get("passes_sa", "")]
                + [result[column] for column in SHARD_COLUMNS[1:]]
            )
    os.replace(tmp_path, merged_output)
    return counts


def _parse_args(argv: Sequence[str] | None) -> tuple[argparse.Namespace, list[str]]:
    """Split script options from config paths and OmegaConf overrides."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--shard", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--merged-output", type=Path, default=None)
    args, config_args = parser.parse_known_args(argv)
    if args.merge:
        if args.merged_output is None:
            parser.error("--merge requires --merged-output.")
    elif args.shard is None:
        parser.error("--shard is required unless --merge is given.")
    return args, config_args


def main(argv: Sequence[str] | None = None) -> None:
    """Dock one shard, or merge the shards, from the command line."""
    args, config_args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    rows = read_input_rows(args.input)

    if args.merge:
        counts = merge_shards(
            rows, args.output_dir, args.num_shards, args.merged_output
        )
        total = sum(counts.values())
        for reason, count in counts.most_common():
            logger.info("%-28s %7d  (%.2f%%)", reason, count, 100 * count / total)
        logger.info("Wrote %d rows to %s.", total, args.merged_output)
        if counts["missing"]:
            raise SystemExit(
                f"{counts['missing']} molecules are missing; rerun the failed shards."
            )
        return

    if not any("=" not in arg for arg in config_args):
        raise SystemExit("At least one config file path must be provided.")
    indices = shard_indices(len(rows), args.shard, args.num_shards)
    smiles = [rows[index]["SMILES"] for index in indices]

    from activelearning.main import process_arguments
    from activelearning.runtime import bind_runtime_context

    _, cfg, _, _ = process_arguments(config_args)
    if cfg.oracle.type != "Dock3Oracle":
        raise SystemExit(f"Expected a Dock3Oracle config, got {cfg.oracle.type}.")
    (fidelity,) = cfg.oracle.fidelity_costs
    oracle = cfg.oracle.build()
    bind_runtime_context([oracle], cfg.runtime.build(logger=None))
    dock_shard(
        oracle,
        smiles,
        fidelity,
        shard_path(args.output_dir, args.shard),
        args.chunk_size,
    )
    logger.info("Shard %d done.", args.shard)


if __name__ == "__main__":
    main(sys.argv[1:])
