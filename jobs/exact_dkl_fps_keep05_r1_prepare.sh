#!/bin/bash
# Simulate the training set of active-learning round 2: the same selection as
# jobs/exact_dkl_fps_keep05_prepare.sh, but over the 10M library PLUS the 10,000
# molecules the first reward-shape round docked.
#
#   sbatch jobs/exact_dkl_fps_keep05_r1_prepare.sh
#
# Two steps. First scripts/build_round_archive.py writes the round-1 archive,
# data/ampc_archive_round1.csv: the library row for row, then every molecule
# that round docked and the library does not already hold (9,887 of 10,000 have
# a target; 113 failed to dock). Then the same stratified farthest-point
# selection runs over that archive with --strat-keep-above 0.5.
#
# What changes, and is the point of the simulation:
# Everything this job writes lands in data/round1 and under the fps_keep05_r1
# label, so round 0 can be re-fitted from its own files at any time.
#
#   - 699 of the docked molecules score y >= 0.5, so keep_above now keeps ~1,212
#     rows outright, not 513, and the generated molecules are the majority of
#     that tier.
#   - the band edges are quantiles of the archive, which 10,000 rows in 10M move
#     negligibly, so the bands are the round-0 bands and the comparison is clean.
#   - the top band [0.232, 0.5) and every lower band now draw from library and
#     generated molecules together, FPS deciding between them on fingerprint
#     distance alone.
#
# --strat-label fps_keep05_r1 namespaces every output
# (data/round1/ampc_fps_keep05_r1_25000.csv, the fps_keep05_r1_25000 and
# fps_keep05_r1_pool caches, the 'stratified_fps_keep05_r1' manifest block), so
# the round-0 set and its caches are untouched and the two can be compared.
#
# Cost. The archive stream is a few minutes; the pool (~891k rows, of which the
# generated molecules are new to MiniMol) took 17.5 minutes to encode on one
# H100 at round 0. A rerun reuses fps_keep05_r1_pool.npy.
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1:30:00
#SBATCH --job-name=ampc_fps_keep05_r1_prep
# Re-check the fair share before every submit:
#   sshare -l -A def-bengioy_gpu,def-mailhoto_gpu,def-yvesbrun_gpu
#SBATCH --account=def-yvesbrun_gpu
#SBATCH --output=slurm_logs/ampc_fps_keep05_r1_prep_%j.out
#SBATCH --error=slurm_logs/ampc_fps_keep05_r1_prep_%j.err

set -euo pipefail

cd /home/ethankrz/activelearning
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1   # compute nodes have no internet

mkdir -p slurm_logs cache/ampc/exact_dkl_top_n data/round1

ARCHIVE="data/ampc_archive_round1.csv"
SCORED="outputs/ampc/reward_shape_nibi/reward_power_beta0.1_vargrad_floor-150_steps5000_fixedgp_fps_keep05_fp32/artifacts/acquisition/general/round_0001/scored_samples.csv"

# Idempotent: the archive is a pure function of the library and the scored
# samples, so a rerun of this job does not rebuild it.
if [[ -f "${ARCHIVE}" ]]; then
  echo "${ARCHIVE} exists; keeping it."
else
  uv run --no-sync python -m scripts.build_round_archive \
    --library-csv data/10M_unif_random_subset.csv \
    --scored-csv "${SCORED}" \
    --out-csv "${ARCHIVE}"
fi

# Identical to jobs/exact_dkl_fps_keep05_prepare.sh but for --training-csv,
# --strat-label and --data-dir. The last one matters: --train-sizes cannot be
# empty, so the script always rewrites ampc_top_<size>.csv from whatever
# --training-csv holds, and over this archive that file would no longer be the
# library's top 2,000. --data-dir data/round1 sends every CSV this job writes
# there, so no round-0 file is touched. The evaluation sets are resolved only to count their overlap
# with the new selection; their caches already exist and are not rewritten
# (--sets).
uv run --no-sync python -m scripts.exact_dkl_prepare \
  --cache-dir cache/ampc/exact_dkl_top_n \
  --data-dir data/round1 \
  --training-csv "${ARCHIVE}" \
  --val-csv cache/ampc/exact_dkl_top_n/val_set.csv \
  --gp-molformer-csv analyze_scored_samples_csv/gpmolformer_prior_100k_docked.csv \
  --ampc-331k-csv cache/ampc/exact_dkl_top_n/ampc_331k.csv \
  --olivier-csv cache/ampc/exact_dkl_top_n/olivier_invitro.csv \
  --train-sizes 2000 \
  --n-train-random 100000 \
  --n-train-top 10000 \
  --strat-sizes 25000 \
  --strat-quantiles 50,75,90,95,99,99.75 \
  --strat-top-fraction 0.05 \
  --strat-band-fill proportional \
  --strat-pool-multiple 40,40,40,15,15,15,15 \
  --strat-keep-above 0.5 \
  --strat-label fps_keep05_r1 \
  --strat-seed 42 \
  --eval-seed 42 \
  --sets fps_keep05_r1_25000 \
  --checkpoint-path minimol_ampc_encoder/model/final.pt \
  --package-path minimol_ampc_encoder \
  --encoder-device cuda \
  --encoder-batch-size 64 \
  "$@"

echo
echo "Band diagnostics, the farthest-point spread against a random draw, and the"
echo "evaluation-set overlap counts are under 'stratified_fps_keep05_r1' in"
echo "cache/ampc/exact_dkl_top_n/prep_manifest.json."
echo
echo "Check before the fit:"
echo "  - the archive has 10,009,887 rows (10,000,000 + 9,887 docked, 0 already held)"
echo "  - n_kept is ~1212 (513 from the library + 699 docked at y >= 0.5)"
echo "  - nn_ratio > 1 per band, i.e. the spreading beat a random draw"
echo "  - data/ampc_fps_keep05_25000.csv and data/ampc_top_2000.csv are unchanged"
echo
echo "Then fit:"
echo "  TRAIN_SET=fps_keep05_r1 PRECISION=32 KERNEL=iso FAMILY=matern OUTPUTSCALE=1 \\"
echo "    PRIOR_MEAN=0.03956042999405787 sbatch --time=2:00:00 jobs/fixed_feature_gp_balanced_nibi.sh"
