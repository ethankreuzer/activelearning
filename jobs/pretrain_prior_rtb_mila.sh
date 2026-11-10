#!/bin/bash
# Pretrain the S3-GFN prior with off-policy RTB on the TRUE binding probability.
#
#   sbatch jobs/pretrain_prior_rtb_mila.sh <beta> <n_steps> <on_policy_fraction> [seed]
#
# The beta scan, arm 1 (no surrogate at all):
#   sbatch jobs/pretrain_prior_rtb_mila.sh 5  1000 0.0 42
#   sbatch jobs/pretrain_prior_rtb_mila.sh 10 1000 0.0 42
#   sbatch jobs/pretrain_prior_rtb_mila.sh 15 1000 0.0 42
# then arm 2 at the best beta, adding the on-policy stratum:
#   sbatch jobs/pretrain_prior_rtb_mila.sh 10 1000 0.25 42
#
# WHY. REWARD_SHAPE_UNDERFLOW_GAP.md section 7 measures a first phase of ~750
# steps in which every acquisition arm learns nothing: log GIBBON over the 300k
# training set has median -1864 and 98.4% of it underflows to exactly 0.0 on the
# value scale, so a batch of 64 is entirely tied at the floor, Var(r) = 0, and
# the reward contributes no gradient until something breaches. GP-MoLFormer is
# too far from where the acquisition carries information. And the acquisition is
# only loosely about binding in the first place -- over that same set its
# correlation with y is 0.483 against 0.578 with the GP's own latent std
# (analyze_log_acq_distribution/svgp300k_mila/stats.json), so it tracks the
# model's uncertainty more than the thing we want.
#
# This job moves the policy to the right region first, toward
#
#     pi_1(x)  proportional to  pi_0(x) * exp(beta * y(x))
#
# NO SURROGATE in the default arm. y here is the real docking-derived binding
# probability already stored in data/ampc_svgpscale_300000.csv, so the molecules
# and the labels are both real and there is no GP extrapolation for the policy to
# exploit -- the failure that section calls "the objective's actual optimum", and
# the source of the [Al]-in-an-aromatic-ring artefact. A surrogate is built only
# when on_policy_fraction > 0, which needs predicted y for generated molecules.
#
# NO FLOOR. y lives in [0.03, 0.67], so log R = beta * y is bounded: no
# underflow, no log_score_floor, no 255.6-nat accidental gap, no per-batch floor
# drift. None of that apparatus applies here, which is most of why this arm is
# cheap.
#
# BETA IS NOT COMPARABLE TO THE ACQUISITION ARMS' 0.015. log R = beta * r, and r
# is whatever the transform makes: there r = max(log acq, floor) spanned hundreds
# of nats, here r = y spans 0.63 units. Only beta * spread(r) compares, and by
# that measure beta=25 here (15.9 nats) is about equal to tvdet6ii's
# 0.015 / floor -1000 (14.9 nats). Same pressure, different units.
#
# USE 5 TO 15, CENTRE 10. effective_support is the diagnostic that actually
# predicted collapse in that study, and it is harsher here because y is
# continuous and graded while log-acquisition is a floored heavy tail with almost
# everything tied. Measured over 400 real mixed batches of 32 tail + 32 bulk from
# this pool:
#
#   beta      5     8    10    15    25    50
#   supp/64  52.0  39.1  30.3  16.8   5.0   1.8
#
# The study's recommended gate was ~28.78/64 (power:0.175) and its 6.36/64 arm
# (power:0.25) collapsed onto a handful of modes. So beta=10 is the equivalent
# operating point, and beta=25 is past the most aggressive arm they ran -- the
# ampc-prior-rtb test run at 25 logged effective_support falling to 1.3-6/64.
# Passing 0.015 would give support 64.00 and express no preference at all.
#
# WHAT DECIDES WHETHER THE CHECKPOINT IS WORTH ANYTHING. The cheapest way for the
# policy to raise exp(beta*y) is to memorize the training tail and emit it, which
# would look like success on every distribution metric while being worthless --
# those molecules are already labelled, already in the surrogate's training set,
# and _generate_final_candidates does not exclude training molecules. So 2,000
# tail molecules are withheld from training and the script reports, at the end:
#
#   gap/holdout_minus_bulk   large and positive  -> it generalized
#   gap/train_minus_holdout  large               -> it memorized
#
# Read those two lines before anything else in the summary.
#
# Resources. The heavy parts are a forward+backward pass over GP-MoLFormer (47M
# parameters, batch 64, sequences under 80 tokens) per step, and one generated
# sample at the end. That fits an a100-40GB with room; the 80 GB a100l would only
# buy queue time. The one-off pool filter -- canonicalize, tokenize and SA-score
# 300,000 molecules -- was measured at 961 molecules/s single-threaded on a Mila
# CPU node, so ~5 min, and it is the reason for 4 CPUs rather than 2.
#SBATCH --partition=main
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
# Measured on an A100-40GB, batch 64: 0.2 s per training step, so 2,500 steps is
# ~8.5 min. Add the 4.8 min pool filter, model load, two gap passes, the final
# 2,048-molecule generation and the Tanimoto novelty pass and a whole run is
# ~20-25 min. 1 h is comfortable margin and schedules far faster than 3 h did.
# A kill at the wall clock loses the run: unlike the scoring jobs, nothing is
# written until training finishes, so do not cut this close.
#SBATCH --time=1:00:00
#SBATCH --job-name=ampc_prior_rtb
#SBATCH --output=slurm_logs/ampc_prior_rtb_%j.out
#SBATCH --error=slurm_logs/ampc_prior_rtb_%j.err

set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: sbatch jobs/pretrain_prior_rtb_mila.sh <beta> <n_steps> <on_policy_fraction> [seed] [overrides...]" >&2
  echo "  beta                 reward inverse temperature; log R = beta * y." >&2
  echo "                       Use 5-15, centre 10. NOT the acquisition arms'" >&2
  echo "                       0.015: r is y here, not log(acq). See the header." >&2
  echo "  n_steps              training steps, >= 100, or 0 for the untrained" >&2
  echo "                       control arm (samples the prior, trains nothing)." >&2
  echo "  on_policy_fraction   share of each batch generated and surrogate-scored." >&2
  echo "                       0.0 never builds a surrogate." >&2
  echo "  seed                 optional; defaults to the config's runtime.seed." >&2
  echo "Extra OmegaConf key=value overrides go after the seed, which must then" >&2
  echo "be given explicitly -- an override in its place is read as the seed." >&2
  exit 2
fi

BETA="$1"
N_TRAIN_STEPS="$2"
ON_POLICY_FRACTION="$3"
SEED="${4:-}"

if [[ ! "${BETA}" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
  echo "beta must be a positive number, got '${BETA}'." >&2
  exit 2
fi
if [[ ! "${N_TRAIN_STEPS}" =~ ^[0-9]+$ ]] \
  || { (( N_TRAIN_STEPS < 100 )) && (( N_TRAIN_STEPS != 0 )); }; then
  echo "n_steps must be 0 or an integer >= 100, got '${N_TRAIN_STEPS}'." >&2
  echo "0 is the untrained control arm; see the header." >&2
  exit 2
fi
if [[ ! "${ON_POLICY_FRACTION}" =~ ^0(\.[0-9]+)?$ ]]; then
  echo "on_policy_fraction must be in [0, 1), got '${ON_POLICY_FRACTION}'." >&2
  exit 2
fi
if [[ -n "${SEED}" && ! "${SEED}" =~ ^[0-9]+$ ]]; then
  echo "seed must be a non-negative integer, got '${SEED}'." >&2
  exit 2
fi

BASE_CONFIG="config/ampc/prior_rtb_pretrain.yaml"
FITTED_DIR="outputs/ampc/variational_gp/svgpscale_300000_svgp_M1024_iso_matern_scaled_pm0.0042864_elbo_cosine_lr3e-3_steps10064_fp32"
STATE_PATH="${FITTED_DIR}/surrogate_state.pt"

# Optional suffix for a rerun of an identical cell, so it gets its own directory
# instead of tripping the never-overwrite-a-checkpoint guard below.
ARM_SUFFIX="${ARM_SUFFIX:-}"
if [[ -n "${ARM_SUFFIX}" && ! "${ARM_SUFFIX}" =~ ^_[A-Za-z0-9.-]+$ ]]; then
  echo "ARM_SUFFIX must look like _name (letters, digits, '.', '-')." >&2
  exit 2
fi

SEED_TAG=""
SEED_ARG=()
if [[ -n "${SEED}" ]]; then
  SEED_TAG="_s${SEED}"
  SEED_ARG=(--seed "${SEED}")
fi

ARM="prior_rtb_beta${BETA}_steps${N_TRAIN_STEPS}_onpolicy${ON_POLICY_FRACTION}${SEED_TAG}_svgp300k${ARM_SUFFIX}"
OUT_DIR="outputs/ampc/prior_rtb/${ARM}"
RUN_NAME="${ARM//_/-}-${SLURM_JOB_ID:-local}"

cd "${HOME}/activelearning"
source .venv/bin/activate

export PYTHONUNBUFFERED=1
# GP-MoLFormer and its tokenizer must already be in the Hugging Face cache; a
# compute node resolving them over the network is a failure mode worth catching
# here rather than mid-run.
export HF_HUB_OFFLINE=1
# Mila compute nodes reach the internet, so W&B runs online. Do NOT copy Nibi's
# WANDB_MODE=offline / wandb sync pair into this script.
export WANDB_DIR="${OUT_DIR}"

REQUIRED=("data/ampc_svgpscale_300000.csv")
# The surrogate, its feature cache and the MiniMol encoder are needed only by the
# on-policy stratum; demanding them for arm 1 would block a run that never reads
# them.
if [[ "${ON_POLICY_FRACTION}" != "0.0" && "${ON_POLICY_FRACTION}" != "0" ]]; then
  REQUIRED+=(
    "${STATE_PATH}"
    "cache/ampc/exact_dkl_top_n/svgpscale_300000.npy"
    "minimol_ampc_encoder/model/final.pt"
  )
fi
for required in "${REQUIRED[@]}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing ${required}." >&2
    exit 2
  fi
done

# The checkpoint is the deliverable and costs a GPU-hour; never overwrite one.
if [[ -e "${OUT_DIR}/policy" ]]; then
  echo "${OUT_DIR}/policy already exists." >&2
  echo "Move it aside, or set ARM_SUFFIX to write beside it:" >&2
  echo "  ARM_SUFFIX=_rerun2 sbatch jobs/pretrain_prior_rtb_mila.sh ${BETA} ${N_TRAIN_STEPS} ${ON_POLICY_FRACTION} ${SEED}" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}" slurm_logs

TAGS="prior-rtb,no-docking,exponential,beta${BETA},vargrad,steps${N_TRAIN_STEPS},onpolicy${ON_POLICY_FRACTION},svgp300k,true-y,mila${SEED_TAG:+,seed${SEED}}"

read -r -d '' WANDB_NOTES_TEXT <<EOF || true
Off-policy RTB prior pretraining, arm ${ARM}.

TARGET. pi_1 proportional to pi_0 * exp(${BETA} * y), with y the TRUE
docking-derived binding probability stored in data/ampc_svgpscale_300000.csv
(hit_rate.py maps a DOCK3 score to P(pKi >= 6.5); Dock3Oracle records it as y).
Loss is vargrad, so there is no log_z. Transform is exponential, so r = y and
log R = ${BETA} * y -- bounded, hence no log_score_floor and none of the
underflow-gap apparatus.

WHY. The acquisition arms spend ~750 steps with Var(r) = 0 because 98.4% of log
GIBBON over this training set underflows to exactly 0.0 on the value scale
(REWARD_SHAPE_UNDERFLOW_GAP.md section 7). This moves the policy to where the
acquisition carries information before the acquisition is ever used.

STRATA. Every batch mixes tail (y > 0.2617) and bulk molecules, because vargrad
subtracts the batch mean and its signal is purely within-batch: a tail-only batch
would say nothing about the bulk, and the mass needed to raise the tail would come
from wherever is cheapest, which in SMILES space is valid chemistry.
on_policy_fraction=${ON_POLICY_FRACTION} of each batch is generated by the current
policy and scored by PosteriorMean instead; that stratum is the only thing that
constrains the loss on the policy's OWN support.

DATA. The y > 0.2617 tail of this file is all 16,292 molecules of the 10M library
above that threshold -- kept whole in every arm of the scaling study
(svgpscale_selection.json, keep_above / n_kept) -- so nothing was copied from
another cluster and no surrogate pre-screen was needed to find the tail. About 7%
of it fails the SA threshold of 4.0, and 2,000 of what remains is WITHHELD from
training.

HOW TO READ IT. gap/holdout_minus_bulk large and positive means the policy learned
what makes a binder; gap/train_minus_holdout large means it memorized ~13k strings,
in which case nothing downstream is worth running. Then validity_rate and
unique_rate of the generated sample, then frac_in_library.

NOT MEANINGFUL HERE. No oracle is queried and no round loop runs: the oracle is a
ConstantOracle stub because Mila has no AmpC receptors.
EOF
export WANDB_NOTES="${WANDB_NOTES_TEXT}"
printf '%s\n' "${WANDB_NOTES_TEXT}" > "${OUT_DIR}/arm_description.txt"

uv run --no-sync python scripts/pretrain_prior_rtb.py \
  "${BASE_CONFIG}" \
  --out-dir "${OUT_DIR}" \
  --beta "${BETA}" \
  --n-steps "${N_TRAIN_STEPS}" \
  --on-policy-fraction "${ON_POLICY_FRACTION}" \
  "${SEED_ARG[@]}" \
  --wandb-project "ampc-prior-rtb" \
  --wandb-entity "models-mila5723" \
  --wandb-group "prior-pretrain" \
  --wandb-tags "${TAGS}" \
  --run-name "${RUN_NAME}" \
  "${@:5}"

echo
echo "Done. The checkpoint is ${OUT_DIR}/policy -- an ordinary local Hugging Face"
echo "directory, so sampler.model_name_or_path accepts it directly."
echo
echo "Read the generalization verdict first:"
echo "  python -c \"import json;d=json.load(open('${OUT_DIR}/pretrain_summary.json'));print(json.dumps(d['gaps'],indent=2))\""
