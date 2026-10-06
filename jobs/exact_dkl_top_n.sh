#!/bin/bash
# One arm of the exact-DKL top-n study.
#
#   sbatch jobs/exact_dkl_top_n.sh 2000 gibbon
#   sbatch jobs/exact_dkl_top_n.sh 2000 qmfmes
#   sbatch jobs/exact_dkl_top_n.sh 3000 gibbon
#   sbatch jobs/exact_dkl_top_n.sh 3000 qmfmes
#
# Requires jobs/exact_dkl_prepare.sh to have run: the arm reads the per-set
# caches and runs no MiniMol inference.
#
# Right-sized deliberately. The variational jobs took a whole 48-core 510 GB
# node because of the 10M rows; here the fit is n <= 3000 and the largest
# resident object is the 331k feature matrix at 679 MB, so 32 GB is about ten
# times the headroom needed -- enough that an OOM can never corrupt the memory
# measurement, small enough to schedule quickly. 4 CPUs: no featurization, no
# docking, nothing parallel.
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --job-name=ampc_exact_dkl
#SBATCH --account=def-yvesbrun_gpu
#SBATCH --output=slurm_logs/ampc_exact_dkl_%j.out
#SBATCH --error=slurm_logs/ampc_exact_dkl_%j.err

set -euo pipefail

N="${1:?usage: sbatch jobs/exact_dkl_top_n.sh <2000|3000> <gibbon|qmfmes> [extra args]}"
ARM="${2:?usage: sbatch jobs/exact_dkl_top_n.sh <2000|3000> <gibbon|qmfmes> [extra args]}"

case "${ARM}" in
  gibbon) OVERLAY="" ;;
  qmfmes) OVERLAY="config/ampc/overrides/exact_dkl_qmfmes.yaml" ;;
  *) echo "arm must be 'gibbon' or 'qmfmes', got '${ARM}'" >&2; exit 2 ;;
esac

if [[ ! -f "data/ampc_top_${N}.csv" ]]; then
  echo "data/ampc_top_${N}.csv is missing; run 'sbatch jobs/exact_dkl_prepare.sh' first" >&2
  exit 2
fi

cd /home/ethankrz/activelearning
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1   # compute nodes have no internet
export WANDB_MODE=offline

OUTPUT_DIR="outputs/ampc/exact_dkl_top_n/${N}_${ARM}"
# Per-arm, so the four arms never race on a shared ./wandb/latest-run symlink.
export WANDB_DIR="${OUTPUT_DIR}"
mkdir -p "${WANDB_DIR}" slurm_logs

# --score-chunk-size is small on purpose, and is NOT the variational config's 5000.
# Scoring adds a q-dimension (score_encoded unsqueezes each row to (chunk, 1, d)), so
# GPyTorch's exact-GP posterior carries a copy of all n training inputs per batch
# element: chunk x (n+1) x 512 x 8 bytes in float64. At 5000 that is 38 GiB for
# n=2000 and 57 GiB for n=3000, which OOMs a 40 GB A100; at 128 it is ~1.6 GiB.
# --eval-chunk-size can stay large because predict_encoded passes (chunk, d) with no
# q-dimension, so its concatenation is (n + chunk, d) and costs megabytes.
# shellcheck disable=SC2086  # OVERLAY is intentionally unquoted: empty = no overlay
uv run --no-sync python -m scripts.exact_dkl_top_n \
  config/ampc/exact_dkl_top_n.yaml ${OVERLAY} \
  "dataset.initial_data.path=data/ampc_top_${N}.csv" \
  "surrogate.encoder.feature_cache_path=cache/ampc/exact_dkl_top_n/train_${N}.npy" \
  --output-dir "${OUTPUT_DIR}" \
  --cache-dir cache/ampc/exact_dkl_top_n \
  --eval-chunk-size 5000 \
  --score-chunk-size 128 \
  --acquisition-seed 42 \
  --wandb-project ampc-exact-dkl-top-n \
  --wandb-entity models-mila5723 \
  --wandb-group exact-dkl-top-n \
  --wandb-tags "exact-dkl-top-n,n${N},${ARM}" \
  --run-name "${N}-${ARM}-${SLURM_JOB_ID}" \
  "${@:3}"

echo
echo "Done. Sync from a login node with:"
echo "  wandb sync ${OUTPUT_DIR}/wandb/offline-run-*"
