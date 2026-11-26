"""Corrected y labels for gp_molformer_set, for the restructured chapter scripts.

The sweep's eval CSVs carry the stale y (273 molecules on the old floor). relabel_gpmolformer.py writes the corrected labels
(raw DOCK score -> ampc_full_scores_1p7b.df -> pProp -> hit rate); this module joins them onto any eval frame by SMILES.
Only the label-based quantities change; the model's mean, std and GIBBON columns are untouched.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
_NEW = None


def new_y_table() -> pd.Series:
    global _NEW
    if _NEW is None:
        t = pd.read_csv(HERE / "gp_molformer_relabelled.csv", usecols=["SMILES", "y"])
        assert t.SMILES.is_unique
        _NEW = t.set_index("SMILES").y
    return _NEW


def with_new_y(d: pd.DataFrame) -> pd.DataFrame:
    """Replace d['y'] by the corrected label, matched on SMILES (d must have a SMILES column)."""
    d = d.copy()
    y = d["SMILES"].map(new_y_table())
    assert y.notna().all(), f"{int(y.isna().sum())} molecules have no corrected label"
    d["y"] = y.to_numpy()
    return d
