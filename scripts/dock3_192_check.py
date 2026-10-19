"""Dock exactly 192 molecules on one 192-core node: 1 array task, 1 wave.

No sharding, no S3-GFN sampling -- reuses 192 already-generated SMILES to
check that the JAVA_TOOL_OPTIONS fix (SINGLE_THREAD_ENV in _parallel.py)
holds at full target concurrency (192 ligbuild calls at once on 192 cores).
"""

from __future__ import annotations

import csv
import math
import sys
from collections import Counter
from pathlib import Path

from activelearning.applications.molecules.slurm_dock3_oracle import SlurmDock3Oracle
from activelearning.utils.types import Candidate

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_CSV = REPO_ROOT / "nibi_slurm_dock3_fix_check_results.csv"
OUTPUT_CSV = REPO_ROOT / "nibi_slurm_dock3_192_check_results.csv"
N = 192


def main() -> int:
    with SOURCE_CSV.open(newline="", encoding="utf-8") as handle:
        smiles_list = [row["smiles"] for row in csv.DictReader(handle)][:N]
    if len(smiles_list) != N:
        raise RuntimeError(f"expected {N} SMILES, found {len(smiles_list)}")

    candidates = [Candidate(x=smiles, fidelity=1) for smiles in smiles_list]

    oracle = SlurmDock3Oracle(
        indock_template="/project/def-mailhoto/share/ampc_chemstep_dockfiles/dockfiles/INDOCK",
        dockfiles_dir="/project/def-mailhoto/share/ampc_chemstep_dockfiles/dockfiles/",
        dockenv_sh="/home/ethankrz/dock/dockenv_nibi.sh",
        dock64_exe="/project/def-mailhoto/soft/software/dock-3.8.0-2022/bin/dock64",
        fidelity_costs={1: 32.0},
        fidelity_confidences={1: 1.0},
        hitrate_params=str(REPO_ROOT / "ampc_hitrate_fits/fitted_params.json"),
        score_pprop_table=str(REPO_ROOT / "ampc_hitrate_fits/data/full_scores.df"),
        hitrate_target="ampc",
        pki_threshold=6.5,
        shared_work_dir="/scratch/ethankrz/activelearning/dock3_queries",
        num_array_tasks=1,
        max_parallel_tasks=1,
        num_workers=192,
        poll_interval=60.0,
        ligbuild_timeout=150,
        sbatch_args=["--account=def-mailhoto_cpu", "--mem=64G", "--time=00:20:00"],
    )

    observations = list(oracle.query(candidates))
    if len(observations) != len(candidates):
        raise RuntimeError(
            f"DOCK3 returned {len(observations)} observations for "
            f"{len(candidates)} molecules"
        )

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["index", "smiles", "y", "dock3_raw_score", "dock3_failure_reason"],
        )
        writer.writeheader()
        for index, (candidate, observation) in enumerate(zip(candidates, observations)):
            metadata = observation.metadata or {}
            writer.writerow(
                {
                    "index": index,
                    "smiles": candidate.x,
                    "y": observation.y,
                    "dock3_raw_score": metadata.get("dock3_raw_score"),
                    "dock3_failure_reason": metadata.get("dock3_failure_reason"),
                }
            )

    successful = sum(math.isfinite(float(o.y)) for o in observations)
    reasons = Counter(
        (o.metadata or {}).get("dock3_failure_reason") for o in observations
    )
    reasons.pop(None, None)
    print(f"{successful}/{len(observations)} successful docks; results={OUTPUT_CSV}")
    for reason, count in sorted(reasons.items()):
        print(f"Failure {reason}: {count}")
    return 0 if successful else 1


if __name__ == "__main__":
    sys.exit(main())
