"""Check the 232k validation screen's labels against the 1.7B score -> pProp table.

For every row, ``y`` is recomputed from the raw docking ``score`` (score ->
pProp through the table, then the fitted hit-rate model) and compared with the
labelled file. The stored ``pprop``, the SMILES row order and the weights are
checked too. Read-only. Run on a compute node, from the repository root:

    python scripts/verify_val_screen_y.py
"""

from __future__ import annotations

import csv
import logging
import sys

from activelearning.applications.molecules.hit_rate import HitRateModel

SCORES = "data/ampc_val_screen_232k.csv"
LABELLED = "data/ampc_val_screen_232k_labelled.csv"
TABLE = "ampc_hitrate_fits/data/ampc_full_scores_1p7b.df"
PARAMS = "ampc_hitrate_fits/fitted_params.json"
PKI_THRESHOLD = 6.5
# The table stores float32 scores (-144.3000030517578 for -144.3), so the
# recomputed values differ from the stored ones by a few 1e-6.
TOLERANCE = 1e-5


def main() -> int:
    """Compare every row and report mismatch counts and largest differences."""
    # pprop() warns for each out-of-range score; they are counted below instead.
    logging.getLogger("activelearning.applications.molecules.hit_rate").setLevel(
        logging.ERROR
    )
    model = HitRateModel.from_files(PARAMS, TABLE, target="ampc")
    cache: dict[float, tuple[float, float]] = {}
    rows = out_of_range = smiles_bad = 0
    bad = {"pprop": 0, "y": 0, "weight": 0}
    worst = dict.fromkeys(bad, 0.0)

    with open(SCORES, newline="") as left, open(LABELLED, newline="") as right:
        for source, labelled in zip(
            csv.DictReader(left), csv.DictReader(right), strict=True
        ):
            rows += 1
            smiles_bad += source["SMILES"] != labelled["SMILES"]
            score = float(source["score"])
            out_of_range += not model.score_min <= score <= model.score_max
            if score not in cache:
                pprop = model.pprop(score)
                cache[score] = (
                    pprop,
                    model.hit_rate_from_pprop(pprop, PKI_THRESHOLD),
                )
            pprop, y = cache[score]
            differences = {
                "pprop": abs(float(source["pprop"]) - pprop),
                "y": abs(float(labelled["y"]) - y),
                "weight": abs(float(labelled["weight"]) - float(source["ipw"])),
            }
            for key, difference in differences.items():
                # Written so that a NaN counts as a mismatch.
                bad[key] += not difference <= TOLERANCE
                worst[key] = max(worst[key], difference)

    print(f"rows checked              : {rows}")
    print(f"scores outside the table  : {out_of_range}")
    print(f"SMILES mismatches         : {smiles_bad}")
    for key in bad:
        print(f"{key:<6} mismatches (>{TOLERANCE:g}) : {bad[key]}  max diff {worst[key]:.3g}")
    passed = rows > 0 and smiles_bad == 0 and not any(bad.values())
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
