#!/bin/bash
# Fit the exact-DKL surrogate on a stratified farthest-point training set.
#
#   sbatch jobs/exact_dkl_stratified.sh <n> [latent_dim|none] [prior_mean]
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
#
# latent_dim defaults to the sweep's 256 and writes to stratified_<n>. Any other
# width is the capacity ablation and writes to stratified_<n>_d<latent_dim>, so
# it can run alongside the 256 arm: on these sets the 256-d fit drove the GP
# noise to its lower bound, and a narrower projection has less room to explain
# the training targets without noise. Fit cost is the n x n kernel matrix, so
# the width does not change the time or memory above.
#
# latent_dim "none" removes the layer: the GP sees the 512-d MiniMol
# fingerprints, rescaled by one fixed scalar, and writes to
# stratified_<n>_nolayer. Nothing trainable can then pull dissimilar molecules
# together, which is the far end of the same ablation.
#
# prior_mean pins the GP's constant mean, on the original target scale, and
# appends _pm<value> to the arm. Half of a stratified set is top molecules, so
# its learned constant sits near 0.17 while a typical library molecule scores
# 0.04 -- and that constant is what the GP predicts for anything unfamiliar,
# which is the +0.06 to +0.12 bias every off-train set showed. The population
# value is 0.0395: the mean target of train_random, a uniform 100k draw from
# the 10M set (the generated set's own mean is 0.0398).
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
LATENT_DIM="${2:-256}"
PRIOR_MEAN="${3:-}"

if [[ -z "${N}" || ! "${LATENT_DIM}" =~ ^([0-9]+|none)$ \
      || ! "${PRIOR_MEAN}" =~ ^([0-9]*\.?[0-9]+)?$ ]]; then
  echo "usage: sbatch jobs/exact_dkl_stratified.sh <n> [latent_dim|none] [prior_mean]" >&2
  exit 2
fi

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
  MEAN_TEXT="FIXED at ${PRIOR_MEAN} on the original score scale and not trained (0.0395 is the mean score of the 10M library)"
else
  MEAN_TEXT="LEARNED with the other hyperparameters (the default)"
fi
export WANDB_NOTES="Exact GP (no inducing points, no minibatching) fitted to the ${N}-molecule STRATIFIED farthest-point training set data/ampc_strat_${N}.csv: half the molecules from the top score band, the rest spread evenly over lower score bands.
Features: frozen MiniMol AmpC fingerprints, then ${LAYER_TEXT}.
GP prior mean: ${MEAN_TEXT}.
Other settings: float32, full-batch Adam, acquisition scoring skipped; epochs and learning rate from config/ampc/exact_dkl_top_n.yaml.
Arm name: ${ARM}. Arms differ from stratified_${N} (width 256, learned mean) only in what the name adds: _d<w> = layer width, _nolayer = no layer, _pm<v> = fixed prior mean.
Submitted with: sbatch jobs/exact_dkl_stratified.sh $*   (Slurm job ${SLURM_JOB_ID:-unknown})"
printf '%s\n' "${WANDB_NOTES}" > "${OUTPUT_DIR}/arm_description.txt"

uv run --no-sync python -m scripts.exact_dkl_top_n \
  config/ampc/exact_dkl_top_n.yaml \
  "dataset.initial_data.path=data/ampc_strat_${N}.csv" \
  "surrogate.encoder.feature_cache_path=cache/ampc/exact_dkl_top_n/strat_${N}.npy" \
  "surrogate.encoder.activation=none" \
  "surrogate.encoder.latent_dim=${LATENT_DIM_OVERRIDE}" \
  "${PRIOR_MEAN_OVERRIDE[@]}" \
  "runtime.precision=32" \
  --output-dir "${OUTPUT_DIR}" \
  --cache-dir cache/ampc/exact_dkl_top_n \
  --eval-chunk-size 2000 \
  --skip-scoring \
  --overwrite \
  --wandb-project ampc-exact-dkl-top-n \
  --wandb-entity models-mila5723 \
  --wandb-group exact-dkl-stratified \
  --wandb-tags "exact-dkl-stratified,n${N},${LAYER_TAGS},fp32${PRIOR_MEAN:+,fixed-prior-mean}" \
  --run-name "${ARM//_/-}-${SLURM_JOB_ID}" \
  "${@:4}"

echo
echo "Done. Sync from a login node with:"
echo "  wandb sync ${OUTPUT_DIR}/wandb/offline-run-*"
