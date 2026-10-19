"""Time Dock3Oracle on 1000 selected molecules at the job's CPU count.

One point of a docking-throughput curve on Nibi, measured so it carries over to
an active-learning round:

- the oracle is built from the same configs a reward-shape round uses
  (config/ampc/reward_shape.yaml + config/ampc/overrides/nibi_single_fidelity.yaml),
  through the same config loader, so every Dock3Oracle setting matches;
- num_workers stays null, so the pool is sized from the job's CPUs exactly as
  in a round;
- the molecules are what a round selects: the top 1000 by acquisition from the
  100k pool of job 24057277 (data/dock_scaling_top1000.csv), which contains the
  500 that job actually selected;
- all 1000 go through one oracle.query call, as the round's selection does.

Run inside a job that holds an H100, so it lands on a GPU node (Intel 8570,
112 cores) rather than the 192-core CPU nodes. Appends one summary row to
--summary and writes per-molecule results next to it.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from activelearning.main import process_arguments
from activelearning.utils.types import Candidate

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS = [
    REPO_ROOT / "config/ampc/reward_shape.yaml",
    REPO_ROOT / "config/ampc/overrides/nibi_single_fidelity.yaml",
]


def _cpu_model() -> str:
    try:
        output = subprocess.run(
            ["lscpu"], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return platform.processor()
    for line in output.splitlines():
        if line.startswith("Model name:"):
            return line.split(":", 1)[1].strip()
    return platform.processor()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molecules", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--details-dir", type=Path, required=True)
    args = parser.parse_args()

    with args.molecules.open(newline="", encoding="utf-8") as handle:
        smiles_list = [row["SMILE"] for row in csv.DictReader(handle)]
    candidates = [Candidate(x=smiles, fidelity=1) for smiles in smiles_list]

    _, cfg, _, _ = process_arguments([str(path) for path in CONFIGS])
    build_started = time.monotonic()
    oracle = cfg.oracle.build()
    build_s = time.monotonic() - build_started

    query_started = time.monotonic()
    observations = list(oracle.query(candidates))
    query_s = time.monotonic() - query_started
    if len(observations) != len(candidates):
        raise RuntimeError(
            f"DOCK3 returned {len(observations)} observations for "
            f"{len(candidates)} molecules"
        )

    job_id = os.environ.get("SLURM_JOB_ID", "local")
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 0))
    workers = oracle._num_workers
    args.details_dir.mkdir(parents=True, exist_ok=True)
    details_path = args.details_dir / f"dock_scaling_{workers}w_{job_id}.csv"
    with details_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["rank", "smiles", "y", "dock3_raw_score", "dock3_failure_reason"],
        )
        writer.writeheader()
        for rank, (candidate, observation) in enumerate(zip(candidates, observations)):
            metadata = observation.metadata or {}
            writer.writerow(
                {
                    "rank": rank,
                    "smiles": candidate.x,
                    "y": observation.y,
                    "dock3_raw_score": metadata.get("dock3_raw_score"),
                    "dock3_failure_reason": metadata.get("dock3_failure_reason"),
                }
            )

    succeeded = sum(
        observation.y is not None and math.isfinite(float(observation.y))
        for observation in observations
    )
    reasons = Counter(
        (observation.metadata or {}).get("dock3_failure_reason")
        for observation in observations
    )
    reasons.pop(None, None)
    summary = {
        "job_id": job_id,
        "node": socket.gethostname(),
        "cpu_model": _cpu_model(),
        "cpus_per_task": cpus,
        "workers": workers,
        "molecules": len(candidates),
        "succeeded": succeeded,
        "build_s": round(build_s, 1),
        "query_s": round(query_s, 1),
        "core_s_per_molecule": round(query_s * workers / len(candidates), 2),
        "molecules_per_hour": round(len(candidates) / query_s * 3600, 1),
        "failures": json.dumps(dict(sorted(reasons.items()))),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.summary.exists()
    with args.summary.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        if write_header:
            writer.writeheader()
        writer.writerow(summary)

    print(json.dumps(summary, indent=2))
    return 0 if succeeded else 1


if __name__ == "__main__":
    sys.exit(main())
