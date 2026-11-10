#!/bin/bash
# One arm of the exact-GP sweep on Mila. Six arms: two axes, fully crossed.
#
#   sbatch jobs/exact_gp_25k_mila.sh on  pinned     # arm 1, the config as written
#   sbatch jobs/exact_gp_25k_mila.sh on  learned    # arm 2
#   sbatch jobs/exact_gp_25k_mila.sh on  constant   # arm 3
#   sbatch jobs/exact_gp_25k_mila.sh off pinned     # arm 4
#   sbatch jobs/exact_gp_25k_mila.sh off learned    # arm 5
#   sbatch jobs/exact_gp_25k_mila.sh off constant   # arm 6
#
# AXIS 1 is the lengthscale prior, AXIS 2 is how the function's level is
# represented. config/ampc/exact_gp_25k.yaml holds the reasoning for every
# fixed setting and for both axes; THESIS_ACTIVELEARNING.md section 5a is the
# decision record. `learned constant` is rejected by the surrogate: a learned
# mean and the constant kernel's implicit constant are the same quantity.
#
# W&B. Runs go to the project given by WANDB_PROJECT, default ExactGP_thesis --
# the thesis-grade project, kept apart from the exploratory ones. Compute nodes
# have no internet, so the run is written OFFLINE and must be synced from a
# login node afterwards; the command is printed at the end.
#
# EVALUATION, after the fit: predictions on all three held-out sets, and GIBBON
# scores (value and log scale) on gp_molformer_set and olivier_invitro.
#   - ampc_val_screen_232k   230,887  ground-truth labels + ipw. Predict-only in
#                                     STUDY_SETS itself, so NO --predict-only
#                                     flag is passed: that flag only accepts a
#                                     set that would otherwise be scored, and
#                                     naming this one is a hard error. Reason:
#                                     scoring 230,887 candidates against an
#                                     n=25,000 exact GP is not affordable.
#   - gp_molformer_set        83,811  scored. Its labels are still wrong, so its
#                                     label-based metrics must be ignored; the
#                                     GIBBON DISTRIBUTION is label-free and is
#                                     the point of scoring it.
#   - olivier_invitro          1,521  labelled 2026-11-09 from its DOCK scores.
#
# --score-chunk-size 128, MEASURED. The cost is chunk x (n+1) x 512 x 4 bytes at
# fp32: 48.83 MiB per candidate row at n=25,001.
#
# 256 was tried first and OOM'd (job 11620005), which settles how many buffers
# of that size are live at once: the failed request was exactly 12.21 GiB = 256
# rows, with 34.59 GiB of the 39.49 GiB card ALREADY in use, so scoring holds
# two to three such buffers rather than the one a naive reading of the
# allocation suggests. The fit also leaves ~7.5 GiB resident afterwards -- see
# cuda_allocated_at_start_mib on the predict stages of stage_profile.csv -- not
# the ~2.5 GiB the cached Cholesky alone would account for.
#
# So the budget is ~31 GiB for two or three chunk-sized buffers: 128 is 6.1 GiB
# each, ~18 GiB for three, which fits with room. Do not raise it without
# measuring, and note the failure comes AFTER the fit, so a wrong value costs
# the whole run.
#
# expandable_segments reduces the fragmentation the OOM message complained
# about (2.00 GiB reserved but unallocated).
#
# RESOURCES. a100 (40 GB). The fp64 fit of this model at n=25,000 peaked at
# 38.8 GB; fp32 should land near 20 GB, and scoring at chunk 256 adds 12.2 GiB
# to a resident ~2.5 GB of cached Cholesky -- so 40 GB fits both phases with
# room, which a 24 GB card would not.
#
# TIME is the one number with no measurement behind it, and the risk is real:
# the fp64 fit (job 24352820) was CANCELLED at 1 h 55 min after ~4,400
# L-BFGS-B evaluations and was still far from converging. fp32 halves the cost
# per evaluation but does not bound their number. So:
#   - the default is 12 h, uncapped, which is the arm as designed;
#   - MAXFUN=<n> caps the optimizer's objective evaluations instead, which
#     GUARANTEES the run reaches evaluation and W&B sync. A capped fit ends
#     ABNORMAL, which accept_abnormal_lbfgsb already accepts. Use it for a test
#     run; do not use it for the six arms without recording that the cap was
#     applied to all six equally, since an unequal cap is a confound.
#SBATCH --partition=main
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=exact_gp_25k
#SBATCH --output=slurm_logs/exact_gp_25k_%j.out
#SBATCH --error=slurm_logs/exact_gp_25k_%j.err

set -euo pipefail

PRIOR="${1:?usage: sbatch jobs/exact_gp_25k_mila.sh <on|off> <pinned|learned|constant>}"
MEAN="${2:?usage: sbatch jobs/exact_gp_25k_mila.sh <on|off> <pinned|learned|constant>}"

case "${PRIOR}" in
  on)  PRIOR_OVERRIDE="surrogate.lengthscale_prior=true" ;;
  off) PRIOR_OVERRIDE="surrogate.lengthscale_prior=false" ;;
  *) echo "axis 1 must be 'on' or 'off', got '${PRIOR}'" >&2; exit 2 ;;
esac

case "${MEAN}" in
  pinned)   MEAN_OVERRIDE="surrogate.constant_kernel=false" ;;
  learned)  MEAN_OVERRIDE="surrogate.prior_mean=null" ;;
  constant) MEAN_OVERRIDE="surrogate.constant_kernel=true" ;;
  *) echo "axis 2 must be 'pinned', 'learned' or 'constant', got '${MEAN}'" >&2; exit 2 ;;
esac

cd "${HOME}/activelearning"

ARM="${PRIOR}_${MEAN}"
OUTPUT_DIR="outputs/ampc/exact_gp_25k/${ARM}"
WANDB_PROJECT="${WANDB_PROJECT:-ExactGP_thesis}"
WANDB_ENTITY="${WANDB_ENTITY:-models-mila5723}"
SCORE_CHUNK_SIZE="${SCORE_CHUNK_SIZE:-128}"

EXTRA_ARGS=()
if [[ -n "${MAXFUN:-}" ]]; then
  if [[ ! "${MAXFUN}" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAXFUN must be a positive integer, got '${MAXFUN}'" >&2
    exit 2
  fi
  EXTRA_ARGS+=("surrogate.fit_kwargs.optimizer_kwargs.options.maxfun=${MAXFUN}")
  echo "CAPPED FIT: maxfun=${MAXFUN}. This is NOT the arm as designed; it ends"
  echo "ABNORMAL by construction. Record the cap with the results."
fi

source .venv/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1   # compute nodes have no internet
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Per-arm, so arms never race on a shared ./wandb/latest-run symlink.
export WANDB_DIR="${OUTPUT_DIR}"

mkdir -p "${WANDB_DIR}" slurm_logs

nvidia-smi --query-gpu=name,memory.total --format=csv

uv run --no-sync python -m scripts.exact_dkl_top_n \
  config/ampc/exact_gp_25k.yaml \
  "${PRIOR_OVERRIDE}" \
  "${MEAN_OVERRIDE}" \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
  --output-dir "${OUTPUT_DIR}" \
  --cache-dir cache/ampc/exact_dkl_top_n \
  --eval-chunk-size 5000 \
  --score-chunk-size "${SCORE_CHUNK_SIZE}" \
  --acquisition-seed 42 \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-entity "${WANDB_ENTITY}" \
  --wandb-group exact-gp-25k \
  --wandb-tags "exact-gp-25k,prior-${PRIOR},mean-${MEAN}" \
  --run-name "${ARM}-${SLURM_JOB_ID}" \
  "${@:3}"

echo
echo "Done. Sync from a LOGIN node (compute nodes have no internet):"
echo "  cd ${HOME}/activelearning && source .venv/bin/activate"
echo "  wandb sync ${OUTPUT_DIR}/wandb/offline-run-*"
