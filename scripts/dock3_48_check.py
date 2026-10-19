"""Dock 192 molecules in-process with 48 workers on a 48-core allocation.

Checks the full ligbuild + dock64 path of Dock3Oracle at the largest
concurrency NIBI_LIGBUILD_SCALING.md measured clean for ligbuild alone (48).
No Slurm array, no sampling: 192 SMILES from the 10k AmpC subset, four waves
of 48. num_workers is left unset so the pool is sized from the job's CPUs, as
in the single-fidelity configs.
"""

from __future__ import annotations

import csv
import math
import sys
import time
from collections import Counter
from pathlib import Path

from activelearning.applications.molecules.dock3_oracle import Dock3Oracle
from activelearning.utils.types import Candidate

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_CSV = REPO_ROOT / "data/10k_unif_random_subset.csv"
OUTPUT_CSV = REPO_ROOT / "nibi_dock3_48_check_results.csv"
N = 192
EXPECTED_WORKERS = 48


def main() -> int:
    with SOURCE_CSV.open(newline="", encoding="utf-8") as handle:
        smiles_list = [row["SMILE"] for row in csv.DictReader(handle)][:N]
    if len(smiles_list) != N:
        raise RuntimeError(f"expected {N} SMILES, found {len(smiles_list)}")

    candidates = [Candidate(x=smiles, fidelity=1) for smiles in smiles_list]

    oracle = Dock3Oracle(
        indock_template="/project/def-mailhoto/share/ampc_chemstep_dockfiles/dockfiles/INDOCK",
        dockfiles_dir="/project/def-mailhoto/share/ampc_chemstep_dockfiles/dockfiles/",
        dockenv_sh="/home/ethankrz/dock/dockenv_nibi.sh",
        dock64_exe="/project/def-mailhoto/soft/software/dock-3.8.0-2022/bin/dock64",
        fidelity_costs={1: 1.0},
        fidelity_confidences={1: 1.0},
        hitrate_params=str(REPO_ROOT / "ampc_hitrate_fits/fitted_params.json"),
        score_pprop_table=str(REPO_ROOT / "ampc_hitrate_fits/data/full_scores.df"),
        hitrate_target="ampc",
        pki_threshold=6.5,
        num_workers=None,
        ligbuild_timeout=150,
    )
    if oracle._num_workers != EXPECTED_WORKERS:
        raise RuntimeError(
            f"expected {EXPECTED_WORKERS} workers, resolved {oracle._num_workers}"
        )

    started = time.monotonic()
    observations = list(oracle.query(candidates))
    elapsed = time.monotonic() - started
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
    print(
        f"{successful}/{len(observations)} successful docks in {elapsed:.0f} s "
        f"with {oracle._num_workers} workers; results={OUTPUT_CSV}"
    )
    for reason, count in sorted(reasons.items()):
        print(f"Failure {reason}: {count}")
    return 0 if successful else 1


if __name__ == "__main__":
    sys.exit(main())
