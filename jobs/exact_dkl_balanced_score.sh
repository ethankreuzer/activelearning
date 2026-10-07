#!/bin/bash
# Score an already-fitted balanced arm: GIBBON acquisition values over the large
# pools, plus predictions on every evaluation set. Nothing is trained here.
#
#   sbatch jobs/exact_dkl_balanced_score.sh <n> [latent_dim|none] [prior_mean] [seed]
#
# Pass exactly the arguments the fit job was given. The restored state carries
# parameters only, so check_state_matches_config refuses a load whose dataset,
# surrogate or precision differs from the run that produced it -- which is what
# stops a width or a training set being crossed by accident.
#
# What this run produces that the fit run did not: the GIBBON scores (value and
# log scale) for gp_molformer_set, olivier_invitro and ampc_331k, and with them
# the numbers that say whether one active-learning round would be worth running.
# Per scored set, under <set>/acquisition/<scoring>/:
#
#   top100_* / top1000_*   what a round of that size would actually select --
#                          enrichment is 1.0 for a random pick,
#                          achievable_fraction is 1.0 for the best possible pick
#   effective_support      how many molecules the score distribution behaves as
#   n_mass90               though it were spread over; a score that is flat at
#   fraction_at_floor      its floor everywhere has nothing for a sampler to climb
#
# and, pooling ampc_331k with gp_molformer_set, pooled/acquisition/<scoring>/
# top{k}/<set>_share: which set the acquisition prefers when it has to choose. A
# top-k drawn overwhelmingly from the generated set is the acquisition rewarding
# distance from the training data rather than information about the target.
#
# Cost: dominated by GIBBON over ampc_331k's 331,480 molecules, about 8.5
# minutes per scoring pass, so roughly 25 minutes in total.
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --job-name=ampc_dkl_bal_score
#SBATCH --account=def-yvesbrun_gpu
#SBATCH --output=slurm_logs/ampc_dkl_bal_score_%j.out
#SBATCH --error=slurm_logs/ampc_dkl_bal_score_%j.err

set -euo pipefail

N="${1:-}"
LATENT_DIM="${2:-256}"
PRIOR_MEAN="${3:-}"
SEED="${4:-}"

if [[ -z "${N}" || ! "${LATENT_DIM}" =~ ^([0-9]+|none)$ \
      || ! "${PRIOR_MEAN}" =~ ^([0-9]*\.?[0-9]+)?$ \
      || ! "${SEED}" =~ ^[0-9]*$ ]]; then
  echo "usage: sbatch jobs/exact_dkl_balanced_score.sh <n> [latent_dim|none] [prior_mean] [seed]" >&2
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

FITTED_DIR="outputs/ampc/exact_dkl_top_n/${ARM}"
OUTPUT_DIR="${FITTED_DIR}_scored"
if [[ ! -f "${FITTED_DIR}/surrogate_state.pt" ]]; then
  echo "${FITTED_DIR}/surrogate_state.pt not found: fit the arm first with" >&2
  echo "  sbatch jobs/exact_dkl_balanced.sh $*" >&2
  exit 2
fi

# Per-arm, so concurrent runs never race on a shared ./wandb/latest-run symlink.
export WANDB_DIR="${OUTPUT_DIR}"
mkdir -p "${WANDB_DIR}" slurm_logs

# A plain-language description, so the run can be understood on its own months
# later; kept next to the run's outputs as well.
export WANDB_NOTES="ACQUISITION SCORES for the already-fitted arm ${ARM}. Nothing was trained in this run: the exact GP was rebuilt on the ${N}-molecule BALANCED training set data/ampc_balanced_${N}.csv -- the library-proportional mix, top band floored at 5% of the budget -- and its fitted hyperparameters were loaded from ${FITTED_DIR}/surrogate_state.pt.
What is new here: GIBBON acquisition scores (value scale and log scale) for gp_molformer_set, olivier_invitro and ampc_331k, which the fit run skipped; and predictions on ampc_331k and olivier_invitro.
The numbers this study turns on are top1000_enrichment and top1000_achievable_fraction on ampc_331k (what a round would actually select; 1.0 enrichment is a random pick), spearman_std_error on the OFF-training sets (must be positive -- on the stratified arms it was positive only on train_top and negative on everything else), effective_support and fraction_at_floor (whether the score distinguishes more than a handful of molecules), and pooled/acquisition/*/top1000/gp_molformer_set_share (whether the acquisition just prefers whatever is furthest from the training data).
What repeats the fit run: the prediction metrics on train_random, train_top, val_set and gp_molformer_set, recomputed from the restored model; they should match the run named ${ARM//_/-}-<job id>.
There are no per-epoch training curves in this run; they are in the fit run.
Arm name: ${ARM}, see ${FITTED_DIR}/arm_description.txt for what was fitted.
Submitted with: sbatch jobs/exact_dkl_balanced_score.sh $*   (Slurm job ${SLURM_JOB_ID:-unknown})"
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
  --load-state "${FITTED_DIR}/surrogate_state.pt" \
  --output-dir "${OUTPUT_DIR}" \
  --cache-dir cache/ampc/exact_dkl_top_n \
  --eval-chunk-size 2000 \
  --score-chunk-size 64 \
  --overwrite \
  --wandb-project ampc-exact-dkl-balanced \
  --wandb-entity models-mila5723 \
  --wandb-group exact-dkl-balanced \
  --wandb-tags "exact-dkl-balanced,n${N},${LAYER_TAGS},fp32,top5pct,proportional${PRIOR_MEAN:+,fixed-prior-mean}${SEED:+,seed${SEED}},scored,restored-state" \
  --run-name "${ARM//_/-}-scored-${SLURM_JOB_ID}" \
  "${@:5}"

echo
echo "Done. Sync from a login node with:"
echo "  wandb sync ${OUTPUT_DIR}/wandb/offline-run-*"
