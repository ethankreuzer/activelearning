#!/bin/bash
# Build the balanced set's stratified farthest-point selection with every row at
# y >= 0.5 kept outright, plus its feature caches. Run once, before the GP fit
# on it (TRAIN_SET=fps_keep05 jobs/fixed_feature_gp_balanced_nibi.sh).
#
#   sbatch jobs/exact_dkl_fps_keep05_prepare.sh
#   SIZE=45000 sbatch --mem=80G --time=1:30:00 jobs/exact_dkl_fps_keep05_prepare.sh
#
# SIZE (default 25000) is the training-set size; the selection method is the
# same at every size. A size other than 25000 gets its own label,
# fps_keep05_n<SIZE>, and so its own pool: the pool is sized for the budgets,
# and one shared with the 25,000 set would change that set's selection. At
# 45,000 the bulk bands' pools grow 1.8x, to ~1.6M rows (~30 min to encode),
# hence the larger request.
#
# The selection is step 4a of analyze_archive_diversity/ARCHIVE_DIVERSITY_ANALYSIS.md
# (job 24755513): the settings of jobs/exact_dkl_balanced_prepare.sh, plus
# --strat-keep-above 0.5. The 513 rows of the 10M at y >= 0.5 are kept, the top
# band [0.232, 0.5) keeps 5% of the 25,000, and the six lower bands share the
# rest in proportion to the library, farthest-point sampled within each band.
# Step 4a read the pool's fingerprints from the 10M store; this job encodes the
# pool live, as the balanced prep did, so its features match the evaluation
# caches exactly.
#
# --strat-label fps_keep05 namespaces every output (data/ampc_fps_keep05_25000.csv,
# the fps_keep05_25000 and fps_keep05_pool caches, and the
# 'stratified_fps_keep05' block of prep_manifest.json), so the balanced set and
# its caches are untouched.
#
# Cost. The pool (891,380 rows) took 17.5 minutes to encode on one H100 (job
# 24793946, peak RSS 48 GB). A rerun reuses its cache, fps_keep05_pool.npy, and
# only encodes the 25,000 selected rows.
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --job-name=ampc_fps_keep05_prep
# Re-check the fair share before every submit:
#   sshare -l -A def-bengioy_gpu,def-mailhoto_gpu,def-yvesbrun_gpu
#SBATCH --account=def-yvesbrun_gpu
#SBATCH --output=slurm_logs/ampc_fps_keep05_prep_%j.out
#SBATCH --error=slurm_logs/ampc_fps_keep05_prep_%j.err

set -euo pipefail

SIZE="${SIZE:-25000}"
if [[ ! "${SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SIZE must be a positive integer." >&2
  exit 2
fi
if [[ "${SIZE}" == 25000 ]]; then
  LABEL="fps_keep05"
else
  LABEL="fps_keep05_n${SIZE}"
fi

cd /home/ethankrz/activelearning
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1   # compute nodes have no internet

mkdir -p slurm_logs cache/ampc/exact_dkl_top_n

# --sets restricts the encoding loop to the new set, so the job does not
# re-check the caches the earlier prep runs already published. The candidate
# pool is encoded regardless: the selection cannot run without it.
# The evaluation sets are only resolved here to count their overlap with the
# new selection; their caches already exist and are not rewritten (--sets). The
# original source CSVs stayed on Narval (job 24793946 failed on
# data/ampc_val_20k.csv), so the resolved copies in the cache directory stand
# in, except gp_molformer_set: its copy has no passes_sa column, which the SA
# filter reads, so the raw docked prior sample is read instead.
uv run --no-sync python -m scripts.exact_dkl_prepare \
  --cache-dir cache/ampc/exact_dkl_top_n \
  --training-csv data/10M_unif_random_subset.csv \
  --val-csv cache/ampc/exact_dkl_top_n/val_set.csv \
  --gp-molformer-csv analyze_scored_samples_csv/gpmolformer_prior_100k_docked.csv \
  --ampc-331k-csv cache/ampc/exact_dkl_top_n/ampc_331k.csv \
  --olivier-csv cache/ampc/exact_dkl_top_n/olivier_invitro.csv \
  --train-sizes 2000 \
  --n-train-random 100000 \
  --n-train-top 10000 \
  --strat-sizes "${SIZE}" \
  --strat-quantiles 50,75,90,95,99,99.75 \
  --strat-top-fraction 0.05 \
  --strat-band-fill proportional \
  --strat-pool-multiple 40,40,40,15,15,15,15 \
  --strat-keep-above 0.5 \
  --strat-label "${LABEL}" \
  --strat-seed 42 \
  --eval-seed 42 \
  --sets "${LABEL}_${SIZE}" \
  --checkpoint-path minimol_ampc_encoder/model/final.pt \
  --package-path minimol_ampc_encoder \
  --encoder-device cuda \
  --encoder-batch-size 64 \
  "$@"

echo
echo "Band diagnostics, the farthest-point spread against a random draw, and the"
echo "evaluation-set overlap counts are under 'stratified_${LABEL}' in"
echo "cache/ampc/exact_dkl_top_n/prep_manifest.json."
echo
echo "Check before the fit:"
echo "  - n_kept is 513 (every 10M row at y >= 0.5)"
echo "  - the log says 'resolved gp_molformer_set: 83811 rows'"
echo "  - nn_ratio > 1 per band, i.e. the spreading beat a random draw"
echo "  - data/ampc_balanced_25000.csv is unchanged"
