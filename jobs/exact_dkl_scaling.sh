#!/bin/bash
# Fit-scalability sweep for the exact-DKL surrogate: how long does an exact GP
# take to fit at n = 10k/15k/20k/25k, and how much VRAM does it need?
#
#   sbatch jobs/exact_dkl_scaling.sh <n> <32|64>
#
# Differences from jobs/exact_dkl_top_n.sh, all deliberate:
#
#   --skip-scoring   Per-candidate acquisition cost grows with n (the kernel
#                    re-encodes all n training rows for every candidate), so at
#                    n=25000 scoring would add ~30-60 min per arm and swamp the
#                    fit measurement this sweep exists to produce. Predict
#                    stages are kept: they are cheap and they answer whether a
#                    larger n generalizes any better than top-2000 did.
#
#   activation=none  The 256-d projection is linear here, not GELU. A linear
#                    projection makes the kernel learn only a low-rank metric on
#                    the MiniMol fingerprints, which is the capacity ablation
#                    for the feature collapse the 2000/3000 arms showed
#                    (in-sample R2 +1.00, out-of-sample Pearson -0.01).
#
#   precision        Narval has only 40 GB A100s, and fit memory is ~12.5
#                    simultaneous n x n matrices: 37 GiB at n=20000 and 58 GiB
#                    at n=25000 in float64, so float64 caps the sweep near
#                    n=19600. The sweep therefore runs in float32, with one
#                    float64 run at n=15000 as an anchor for what the precision
#                    costs in time and memory.
#
#   --eval-chunk-size 2000   Smaller than the arms' 5000: predicting builds a
#                    chunk x n test-train covariance, which grows with n.
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --job-name=ampc_dkl_scaling
#SBATCH --account=def-yvesbrun_gpu
#SBATCH --output=slurm_logs/ampc_dkl_scaling_%j.out
#SBATCH --error=slurm_logs/ampc_dkl_scaling_%j.err

set -euo pipefail

N="${1:-}"
PRECISION="${2:-}"

if [[ -z "${N}" || -z "${PRECISION}" ]]; then
  echo "usage: sbatch jobs/exact_dkl_scaling.sh <n> <32|64>" >&2
  exit 2
fi
if [[ "${PRECISION}" != "32" && "${PRECISION}" != "64" ]]; then
  echo "precision must be 32 or 64, got '${PRECISION}'" >&2
  exit 2
fi

cd /home/ethankrz/activelearning
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1   # compute nodes have no internet
export WANDB_MODE=offline

OUTPUT_DIR="outputs/ampc/exact_dkl_top_n/scaling_${N}_fp${PRECISION}"
# Per-arm, so concurrent runs never race on a shared ./wandb/latest-run symlink.
export WANDB_DIR="${OUTPUT_DIR}"
mkdir -p "${WANDB_DIR}" slurm_logs

uv run --no-sync python -m scripts.exact_dkl_top_n \
  config/ampc/exact_dkl_top_n.yaml \
  "dataset.initial_data.path=data/ampc_top_${N}.csv" \
  "surrogate.encoder.feature_cache_path=cache/ampc/exact_dkl_top_n/train_${N}.npy" \
  "surrogate.encoder.activation=none" \
  "runtime.precision=${PRECISION}" \
  --output-dir "${OUTPUT_DIR}" \
  --cache-dir cache/ampc/exact_dkl_top_n \
  --eval-chunk-size 2000 \
  --skip-scoring \
  --overwrite \
  --wandb-project ampc-exact-dkl-top-n \
  --wandb-entity models-mila5723 \
  --wandb-group exact-dkl-scaling \
  --wandb-tags "exact-dkl-scaling,n${N},fp${PRECISION},linear" \
  --run-name "scaling-${N}-fp${PRECISION}-${SLURM_JOB_ID}" \
  "${@:3}"

echo
echo "Done. Sync from a login node with:"
echo "  wandb sync ${OUTPUT_DIR}/wandb/offline-run-*"
