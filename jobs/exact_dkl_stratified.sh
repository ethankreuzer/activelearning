#!/bin/bash
# Fit the exact-DKL surrogate on a stratified farthest-point training set.
#
#   sbatch jobs/exact_dkl_stratified.sh <n>
#
# The A/B partner of jobs/exact_dkl_scaling.sh: every setting below matches that
# sweep -- linear projection (activation=none), float32, 1000 epochs, scoring
# skipped, --eval-chunk-size 2000 -- so scaling_<n>_fp32 and stratified_<n>
# differ in the training set and nothing else. Fit time depends only on n, not
# on which molecules were chosen, so the measured cost carries over: about
# 5 minutes at n=10000 and 70 minutes at n=25000.
#
# What it is meant to show: the top-n arms predicted roughly the training mean
# for everything off-train (bias +0.24 on the generated set, val_set R2 -2.7,
# 1-sigma coverage below 0.1%) because every training molecule scored at least
# 0.232. The stratified sets keep half the budget in that tail and spread the
# rest over the target distribution, so the comparison to watch is the
# off-train block: val_set / train_random / gp_molformer_set bias, R2 and
# coverage. train_top stays a partly in-sample set and is not the target here.
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --job-name=ampc_dkl_strat
#SBATCH --account=def-yvesbrun_gpu
#SBATCH --output=slurm_logs/ampc_dkl_strat_%j.out
#SBATCH --error=slurm_logs/ampc_dkl_strat_%j.err

set -euo pipefail

N="${1:-}"

if [[ -z "${N}" ]]; then
  echo "usage: sbatch jobs/exact_dkl_stratified.sh <n>" >&2
  exit 2
fi

cd /home/ethankrz/activelearning
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1   # compute nodes have no internet
export WANDB_MODE=offline

OUTPUT_DIR="outputs/ampc/exact_dkl_top_n/stratified_${N}"
# Per-arm, so concurrent runs never race on a shared ./wandb/latest-run symlink.
export WANDB_DIR="${OUTPUT_DIR}"
mkdir -p "${WANDB_DIR}" slurm_logs

uv run --no-sync python -m scripts.exact_dkl_top_n \
  config/ampc/exact_dkl_top_n.yaml \
  "dataset.initial_data.path=data/ampc_strat_${N}.csv" \
  "surrogate.encoder.feature_cache_path=cache/ampc/exact_dkl_top_n/strat_${N}.npy" \
  "surrogate.encoder.activation=none" \
  "runtime.precision=32" \
  --output-dir "${OUTPUT_DIR}" \
  --cache-dir cache/ampc/exact_dkl_top_n \
  --eval-chunk-size 2000 \
  --skip-scoring \
  --overwrite \
  --wandb-project ampc-exact-dkl-top-n \
  --wandb-entity models-mila5723 \
  --wandb-group exact-dkl-stratified \
  --wandb-tags "exact-dkl-stratified,n${N},fp32,linear" \
  --run-name "stratified-${N}-${SLURM_JOB_ID}" \
  "${@:2}"

echo
echo "Done. Sync from a login node with:"
echo "  wandb sync ${OUTPUT_DIR}/wandb/offline-run-*"
