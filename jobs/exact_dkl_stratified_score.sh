#!/bin/bash
# Score the candidate sets with an already-fitted stratified exact-DKL arm.
#
#   sbatch jobs/exact_dkl_stratified_score.sh <n> [latent_dim|none] [prior_mean]
#
# The arguments are the ones jobs/exact_dkl_stratified.sh was given for the arm.
# That job runs with --skip-scoring, so its arms have predictions but no
# acquisition scores. This one does not fit again: it rebuilds the exact GP on
# the same training set, loads the arm's saved surrogate_state.pt, and then runs
# what the fit job skipped -- the GIBBON acquisition update and the scores of
# gp_molformer_set, olivier_invitro and ampc_331k, on the value and log scales.
# Those three sets are predicted as well; ampc_331k and gp_molformer_set are
# labelled, so each gets a predicted-vs-observed figure with its Pearson
# correlation and R2. olivier_invitro has no docking target: its predictions
# are written to its CSV, with no accuracy metric and no such figure.
#
# It writes to <arm>_scored, next to the fitted arm and never into it. The
# predictions are recomputed there too, so <arm>_scored/eval/<set>.csv can be
# compared with <arm>/eval/<set>.csv to confirm the restored model is the
# fitted one: the mean and std columns should agree to float32 precision.
#
# Cost: no fit, but the n x n Cholesky is paid once and every scored chunk is a
# solve against all n training molecules. Scoring at n=25000 has not been
# measured; the n=3000 arm scored ampc_331k in about 100 s per pass, and the
# cost per candidate grows with n.
#
# --score-chunk-size is small for the reason given in jobs/exact_dkl_top_n.sh:
# score_encoded adds a q-dimension, so GPyTorch carries a copy of all n training
# inputs per candidate in the chunk -- chunk x (n+1) x 512 x 4 bytes in float32,
# 3.3 GiB at n=25000 and a chunk of 64, with several such temporaries alive.
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --job-name=ampc_dkl_strat_score
#SBATCH --account=def-yvesbrun_gpu
#SBATCH --output=slurm_logs/ampc_dkl_strat_score_%j.out
#SBATCH --error=slurm_logs/ampc_dkl_strat_score_%j.err

set -euo pipefail

N="${1:-}"
LATENT_DIM="${2:-256}"
PRIOR_MEAN="${3:-}"

if [[ -z "${N}" || ! "${LATENT_DIM}" =~ ^([0-9]+|none)$ \
      || ! "${PRIOR_MEAN}" =~ ^([0-9]*\.?[0-9]+)?$ ]]; then
  echo "usage: sbatch jobs/exact_dkl_stratified_score.sh <n> [latent_dim|none] [prior_mean]" >&2
  exit 2
fi

# The arm name and overrides must stay in step with jobs/exact_dkl_stratified.sh:
# the script refuses a state whose run was configured differently.
ARM="stratified_${N}"
LATENT_DIM_OVERRIDE="${LATENT_DIM}"
LAYER_TAGS="d${LATENT_DIM},linear"
if [[ "${LATENT_DIM}" == none ]]; then
  ARM="${ARM}_nolayer"
  LATENT_DIM_OVERRIDE=null
  LAYER_TAGS="nolayer"
elif [[ "${LATENT_DIM}" != 256 ]]; then
  ARM="${ARM}_d${LATENT_DIM}"
fi

PRIOR_MEAN_OVERRIDE=()
if [[ -n "${PRIOR_MEAN}" ]]; then
  ARM="${ARM}_pm${PRIOR_MEAN}"
  PRIOR_MEAN_OVERRIDE=("surrogate.prior_mean=${PRIOR_MEAN}")
fi

cd /home/ethankrz/activelearning
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1   # compute nodes have no internet
export WANDB_MODE=offline

FITTED_DIR="outputs/ampc/exact_dkl_top_n/${ARM}"
OUTPUT_DIR="${FITTED_DIR}_scored"
if [[ ! -f "${FITTED_DIR}/surrogate_state.pt" ]]; then
  echo "${FITTED_DIR}/surrogate_state.pt not found: fit the arm first with" >&2
  echo "  sbatch jobs/exact_dkl_stratified.sh $*" >&2
  exit 2
fi

# Per-arm, so concurrent runs never race on a shared ./wandb/latest-run symlink.
export WANDB_DIR="${OUTPUT_DIR}"
mkdir -p "${WANDB_DIR}" slurm_logs

# A plain-language description, so the run can be understood on its own months
# later; kept next to the run's outputs as well.
export WANDB_NOTES="ACQUISITION SCORES for the already-fitted arm ${ARM}. Nothing was trained in this run: the exact GP was rebuilt on the ${N}-molecule stratified training set data/ampc_strat_${N}.csv and its fitted hyperparameters were loaded from ${FITTED_DIR}/surrogate_state.pt.
What is new here: GIBBON acquisition scores (value scale and log scale) for gp_molformer_set, olivier_invitro and ampc_331k, which the fit run skipped; and predictions on ampc_331k (labelled: Pearson, R2 and a predicted-vs-observed figure under ampc_331k/) and on olivier_invitro (no docking target, so predictions only, in its CSV).
What repeats the fit run: the prediction metrics on train_random, train_top, val_set and gp_molformer_set. They are recomputed from the restored model and should match the run named ${ARM//_/-}-<job id>.
There are no per-epoch training curves in this run; they are in the fit run.
Arm name: ${ARM}, see ${FITTED_DIR}/arm_description.txt for what was fitted.
Submitted with: sbatch jobs/exact_dkl_stratified_score.sh $*   (Slurm job ${SLURM_JOB_ID:-unknown})"
printf '%s\n' "${WANDB_NOTES}" > "${OUTPUT_DIR}/arm_description.txt"

uv run --no-sync python -m scripts.exact_dkl_top_n \
  config/ampc/exact_dkl_top_n.yaml \
  "dataset.initial_data.path=data/ampc_strat_${N}.csv" \
  "surrogate.encoder.feature_cache_path=cache/ampc/exact_dkl_top_n/strat_${N}.npy" \
  "surrogate.encoder.activation=none" \
  "surrogate.encoder.latent_dim=${LATENT_DIM_OVERRIDE}" \
  "${PRIOR_MEAN_OVERRIDE[@]}" \
  "runtime.precision=32" \
  --load-state "${FITTED_DIR}/surrogate_state.pt" \
  --output-dir "${OUTPUT_DIR}" \
  --cache-dir cache/ampc/exact_dkl_top_n \
  --eval-chunk-size 2000 \
  --score-chunk-size 64 \
  --overwrite \
  --wandb-project ampc-exact-dkl-top-n \
  --wandb-entity models-mila5723 \
  --wandb-group exact-dkl-stratified \
  --wandb-tags "exact-dkl-stratified,n${N},${LAYER_TAGS},fp32${PRIOR_MEAN:+,fixed-prior-mean},scored,restored-state" \
  --run-name "${ARM//_/-}-scored-${SLURM_JOB_ID}" \
  "${@:4}"

echo
echo "Done. Sync from a login node with:"
echo "  wandb sync ${OUTPUT_DIR}/wandb/offline-run-*"
