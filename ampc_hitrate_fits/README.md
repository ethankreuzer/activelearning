# ampc_hitrate_fits

## Which score -> pProp table is correct

**`data/ampc_full_scores_1p7b.df` is the correct DOCK score -> pProp mapping.**
Use it for any score -> pProp lookup.

The score/pProp tables used before it are **wrong** and are kept only for
history. They must not be used for new work:

- `data/full_scores.df` (13B-library pProp space)
- `data/full_scores_1p47b.df` (ranks against the 1,468,863,695 molecules that
  docked, not the whole library, so every pProp is understated)

The error was never fixed in place, by decision: the corrected mapping lives in
the new file, and the old ones stay as they were. All `score_pprop_table`
settings in `config/` and the `scripts/dock3_*_check.py` scripts now point at the
new file. Runs launched before that switch carry the old, wrong pProp.

### What stays the same

`fitted_params.json` is unchanged. It was fitted in the old pProp space and is
deliberately kept as is. The pProp -> y mapping (`HitRateModel.hit_rate_from_pprop`)
therefore has the same shape as before. Only the pProp a given dock score gets
has changed.

### Corrected 10M subset labels

`data/10M_unif_random_subset.csv` carries `y` computed from the old pProp. A
corrected copy, with the same SMILES and `y` recomputed from the 1.7B table and
`fitted_params.json`, is `data/10M_unif_random_subset_fixed_y.csv` (gitignored).
It was built by `scripts/fix_unif_random_10m_y.py`
(`jobs/fix_unif_random_10m_y.sh`) from the dock scores in
`data/ampc_unif_random_10M.csv.gz`. Its `pprop` column is the old, wrong one and
is not used. Mean `y` over the 10M fell from 0.0396 to 0.00429.

`data/10M_unif_random_subset.csv` itself has not been replaced, so anything
still reading it has the old `y`.
