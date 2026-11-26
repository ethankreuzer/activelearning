#!/usr/bin/env python3
"""Relabel gp_molformer_set from its DOCK scores, without touching any tracked file or sweep output.

The cached labels are stale (273 of 83,811 sit on the old floor 0.0318212). The docked source CSV
(data/gpmolformer_prior_100k_docked.csv, from Narval) carries the raw DOCK score, so we use the library's own
label convention, as scripts/label_olivier_invitro.py does:

    raw score --[ampc_full_scores_1p7b.df]--> pProp --[hit_rate_from_pprop(., pKi 6.5)]--> y

Writes restructured/gp_molformer_relabelled.csv with columns SMILES, y_old, y, raw_score, pprop, block, in the row order
of cache/ampc/exact_dkl_top_n/gp_molformer_set.csv, and prints the checks.

Run from the repo root:
  .venv/bin/python thesis/surrogate_models/restructured/relabel_gpmolformer.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))
from activelearning.applications.molecules.hit_rate import (  # noqa: E402
    DEFAULT_ENRICHED_THRESHOLD, DEFAULT_HIT_THRESHOLD, HitRateModel)

OLD_FLOOR = 0.031821230353210841
TABLE = REPO / "ampc_hitrate_fits" / "data" / "ampc_full_scores_1p7b.df"
PARAMS = REPO / "ampc_hitrate_fits" / "fitted_params.json"
CACHE = REPO / "cache" / "ampc" / "exact_dkl_top_n" / "gp_molformer_set.csv"
SOURCE = REPO / "data" / "gpmolformer_prior_100k_docked.csv"
LIBRARY = REPO / "data" / "10M_unif_random_subset.csv"
OUT = HERE / "gp_molformer_relabelled.csv"


def main():
    model = HitRateModel.from_files(PARAMS, TABLE, target="ampc")
    cache = pd.read_csv(CACHE)  # SMILES, y (stale), in the order the sweep evaluated
    src = pd.read_csv(SOURCE, usecols=["SMILES", "raw_score", "y"]).rename(columns={"y": "y_src"})
    assert cache.SMILES.is_unique and src.SMILES.is_unique, "SMILES must be unique for a safe join"
    m = cache.merge(src, on="SMILES", how="left", validate="one_to_one")
    assert m.raw_score.notna().all(), f"{int(m.raw_score.isna().sum())} cached molecules lack a raw score"
    assert np.allclose(m.y, m.y_src), "source y should equal the stale cached y"

    pprop = np.array([model.pprop(float(s)) for s in m.raw_score])
    y_new = np.array([model.hit_rate_from_pprop(float(p), 6.5) for p in pprop])
    out = pd.DataFrame({"SMILES": m.SMILES, "y_old": m.y, "y": y_new, "raw_score": m.raw_score, "pprop": pprop})
    out["block"] = np.where(y_new >= DEFAULT_HIT_THRESHOLD, 2, np.where(y_new >= DEFAULT_ENRICHED_THRESHOLD, 1, 0))
    out.to_csv(OUT, index=False)

    print("rows", len(out), "| rows at the old floor:", int((np.abs(out.y - OLD_FLOOR) < 1e-9).sum()))
    print(out.y.describe().to_string())
    print("blocks (bulk/enriched/hit):", out.block.value_counts().sort_index().to_dict())

    # check 1: the 32 molecules that are also in the 10M library must reproduce the library's own y
    lib = pd.read_csv(LIBRARY, usecols=["SMILE", "y"]).rename(columns={"SMILE": "SMILES", "y": "y_lib"})
    j = out.merge(lib, on="SMILES", how="inner")
    print("in library:", len(j), "| worst |y_new - y_library|:", float(np.abs(j.y - j.y_lib).max()) if len(j) else "n/a",
          "| worst |y_old - y_library|:", float(np.abs(j.y_old - j.y_lib).max()) if len(j) else "n/a")
    # check 2: monotone in the score (better score = higher y)
    order = np.argsort(out.raw_score.to_numpy())
    print("y non-increasing as score rises:", bool(np.all(np.diff(out.y.to_numpy()[order]) <= 1e-12)))


if __name__ == "__main__":
    main()
