#!/bin/bash
# Fit the exact-DKL surrogate on the library-proportional ("balanced") training
# set. Scoring is a separate job: jobs/exact_dkl_balanced_score.sh.
#
#   sbatch jobs/exact_dkl_balanced.sh <n> [latent_dim|none] [prior_mean] [seed]
#
# The A/B partner of jobs/exact_dkl_stratified.sh. Every setting below matches
# that sweep -- linear projection (activation=none), float32, 1000 epochs,
# scoring skipped -- so stratified_<n>_* and balanced_<n>_* differ in the
# training set and nothing else, and the stratified arms are the before picture.
#
# What changed and why. The stratified sets put half the budget in the top band,
# which is 0.25% of the 10M library. The arm scored from one of them predicted
# 0.2-0.3 for a dense floor of molecules whose observed target is ~0 (bias
# +0.118, R2 -4.44 on ampc_331k), its uncertainty was anti-correlated with its
# error off the training region, and 99.8% of its GIBBON scores sat at the 1e-12
# floor. The balanced set mirrors the library instead -- top band floored at 5%
# of the budget, lower bands proportional -- so ~95% of training molecules sit
# below the top band's cut. See jobs/exact_dkl_balanced_prepare.sh.
#
# What to watch, in order: whether a round's worth of top-scoring molecules is
# any good (ampc_331k/acquisition/gibbon_*/top1000_enrichment and
# top1000_achievable_fraction), whether the predicted uncertainty tracks the
# error OFF the training region (spearman_std_error on ampc_331k, train_random
# and gp_molformer_set -- it must be positive, and on the stratified arms it was
# positive only on train_top and negative everywhere else), and whether the
# acquisition distinguishes more than a handful of molecules
# (effective_support, n_mass90, fraction_at_floor).
#
# latent_dim sweeps the width of the linear projection inside the kernel and
# defaults to 256. "none" removes the layer so the GP sees the 512-d MiniMol
# fingerprints rescaled by one fixed scalar. Read the width against
# run/hyperparameters/lengthscale_effective_dims, which counts the dimensions
# the ARD kernel actually relies on: a width is only worth its cost if that
# number grew with it.
#
# prior_mean pins the GP's constant mean on the original target scale. The
# stratified arms needed it (0.0395) to counteract their enrichment. This set
# should not: its learned constant landing near 0.0395 on its own is the
# diagnostic that the mix did the work, so the default is to leave it learned.
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --job-name=ampc_dkl_bal
#SBATCH --account=def-yvesbrun_gpu
#SBATCH --output=slurm_logs/ampc_dkl_bal_%j.out
#SBATCH --error=slurm_logs/ampc_dkl_bal_%j.err

set -euo pipefail

N="${1:-}"
LATENT_DIM="${2:-256}"
PRIOR_MEAN="${3:-}"
SEED="${4:-}"

if [[ -z "${N}" || ! "${LATENT_DIM}" =~ ^([0-9]+|none)$ \
      || ! "${PRIOR_MEAN}" =~ ^([0-9]*\.?[0-9]+)?$ \
      || ! "${SEED}" =~ ^[0-9]*$ ]]; then
  echo "usage: sbatch jobs/exact_dkl_balanced.sh <n> [latent_dim|none] [prior_mean] [seed]" >&2
  exit 2
fi

ARM="balanced_${N}"
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

SEED_OVERRIDE=()
if [[ -n "${SEED}" ]]; then
  ARM="${ARM}_s${SEED}"
  SEED_OVERRIDE=("runtime.seed=${SEED}")
fi

cd /home/ethankrz/activelearning
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1   # compute nodes have no internet
export WANDB_MODE=offline

OUTPUT_DIR="outputs/ampc/exact_dkl_top_n/${ARM}"
# Per-arm, so concurrent runs never race on a shared ./wandb/latest-run symlink.
export WANDB_DIR="${OUTPUT_DIR}"
mkdir -p "${WANDB_DIR}" slurm_logs

# A plain-language description of the arm, so a run can be understood on its own
# months later. wandb reads WANDB_NOTES at init and shows it on the run's
# Overview page and in the Notes column of the runs table; the same text is kept
# next to the run's outputs.
if [[ "${LATENT_DIM}" == none ]]; then
  LAYER_TEXT="NO trainable layer: the GP kernel sees the 512-d fingerprints directly, divided by one fixed scalar (the median pairwise distance of the training molecules)"
else
  LAYER_TEXT="one trainable LINEAR layer of width ${LATENT_DIM} (512 -> ${LATENT_DIM}, no activation) inside the kernel"
fi
if [[ -n "${PRIOR_MEAN}" ]]; then
  MEAN_TEXT="FIXED at ${PRIOR_MEAN} on the original score scale and not trained"
else
  MEAN_TEXT="LEARNED with the other hyperparameters. On this training set it should settle near 0.0395, the mean score of the 10M library; that it does so without being pinned there is the check that the training mix, not the fixed mean, is carrying the correction"
fi
export WANDB_NOTES="Exact GP (no inducing points, no minibatching) fitted to the ${N}-molecule BALANCED farthest-point training set data/ampc_balanced_${N}.csv. Unlike the earlier stratified sets, this one mirrors the 10M library: the top score band is floored at 5% of the budget and the lower bands take their share of the library, so about 95% of the training molecules sit below the top band's cut. It exists because the stratified arms, trained with half the budget in the top 0.25% of the library, predicted 0.2-0.3 for molecules actually scoring ~0 and produced a GIBBON score that was at its floor for 99.8% of the 331k pool.
Features: frozen MiniMol AmpC fingerprints, then ${LAYER_TEXT}.
GP prior mean: ${MEAN_TEXT}.
Other settings: float32, full-batch Adam, acquisition scoring skipped (run jobs/exact_dkl_balanced_score.sh afterwards); epochs and learning rate from config/ampc/exact_dkl_top_n.yaml.
Arm name: ${ARM}. Arms differ from balanced_${N} (width 256, learned mean, seed from the config) only in what the name adds: _d<w> = layer width, _nolayer = no layer, _pm<v> = fixed prior mean, _s<n> = seed.
Submitted with: sbatch jobs/exact_dkl_balanced.sh $*   (Slurm job ${SLURM_JOB_ID:-unknown})"
printf '%s\n' "${WANDB_NOTES}" > "${OUTPUT_DIR}/arm_description.txt"

uv run --no-sync python -m scripts.exact_dkl_top_n \
  config/ampc/exact_dkl_top_n.yaml \
  "dataset.initial_data.path=data/ampc_balanced_${N}.csv" \
  "surrogate.encoder.feature_cache_path=cache/ampc/exact_dkl_top_n/balanced_${N}.npy" \
  "surrogate.encoder.activation=none" \
  "surrogate.encoder.latent_dim=${LATENT_DIM_OVERRIDE}" \
  "${PRIOR_MEAN_OVERRIDE[@]}" \
  "${SEED_OVERRIDE[@]}" \
  "runtime.precision=32" \
  --output-dir "${OUTPUT_DIR}" \
  --cache-dir cache/ampc/exact_dkl_top_n \
  --eval-chunk-size 2000 \
  --skip-scoring \
  --overwrite \
  --wandb-project ampc-exact-dkl-balanced \
  --wandb-entity models-mila5723 \
  --wandb-group exact-dkl-balanced \
  --wandb-tags "exact-dkl-balanced,n${N},${LAYER_TAGS},fp32,top5pct,proportional${PRIOR_MEAN:+,fixed-prior-mean}${SEED:+,seed${SEED}}" \
  --run-name "${ARM//_/-}-${SLURM_JOB_ID}" \
  "${@:5}"

echo
echo "Fitted. Score it with:"
echo "  sbatch jobs/exact_dkl_balanced_score.sh ${N} ${LATENT_DIM} ${PRIOR_MEAN} ${SEED}"
echo "Sync from a login node with:"
echo "  wandb sync ${OUTPUT_DIR}/wandb/offline-run-*"
