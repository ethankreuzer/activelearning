#!/bin/bash
# Run one W&B sweep agent for the prior-pretraining Bayesian sweep.
#
# Create the sweep once from a login node, then submit agents as an array:
#
#   wandb sweep --project ampc-prior-rtb --entity models-mila5723 \
#     config/ampc/prior_rtb_sweep4.yaml
#   # prints e.g. "Creating sweep with ID: ab12cd34"
#   sbatch --array=1-2 jobs/prior_rtb_sweep_agent_mila.sh ab12cd34
#
# ARRAY SIZE 2, NOT MORE. Mila's main partition allows a user 2 GPUs at once
# (8 CPU / 48 G / 2 GPU), and each agent holds one GPU for its whole lifetime.
# A third array task simply waits on the association limit. Each agent runs
# TRIAL_COUNT trials sequentially. The default of 60 is a ceiling the 24 h wall
# clock is expected to reach first, not a target.
#
# Bayesian search is sequential by nature: with 2 agents, each proposal is made
# against the trials finished so far, so 2 is a reasonable parallelism/quality
# trade and 8 would mostly be random search.
#
# WHAT A TRIAL DOES. scripts/pretrain_prior_rtb.py fine-tunes GP-MoLFormer with
# off-policy RTB on the true binding probability, generates 65,536 molecules,
# and reports as "objective" the product
#
#   frac clearing -744.44  x  validity_rate  x  (1 - memorization share)
#
# which was sweep 2's objective. SWEEPS 3 AND 4 DO NOT USE IT: they pass
# --no-price-validity and --min-validity 0.95, so the objective is the
# exceedance rate discounted for memorization only, with validity gated rather
# than priced. The untrained prior scores 0.00142 and sweep 3's best arm
# reached 0.0142. The 1/64 = 0.0156 figure quoted here and in sweep 2's config
# was always a reference point, never a pass mark. Uniqueness and
# synthesizability remain hard gates. See config/ampc/prior_rtb_sweep4.yaml for
# the current space and why each range moved after the 2026-11-06 y relabel.
#
# Trials write no checkpoint (--skip-policy-save in the sweep command). Re-run
# the winning configuration without that flag to produce one.
#
#SBATCH --partition=main
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
# A trial is ~30-45 min, and the earlier "~13 min" estimate was wrong in both
# directions. Measured: 4.8 min pool filter (per trial -- it is not cached),
# training of 300-1200 steps at 0.2-1.3 s/step depending on the on-policy
# stratum, then ~7.5 min generating 65,536 molecules and ~3 min of GIBBON over
# them. The 65,536 evaluation sample is the reason a trial now costs about
# 18 minutes before training even counts; see prior_rtb_sweep.yaml for why it
# is not negotiable.
#
# 24 h therefore buys roughly 32 to 48 trials per agent, which is why
# TRIAL_COUNT defaults high and the wall clock is expected to be what stops an
# agent rather than the count. An agent killed at the wall loses only its trial
# in flight; finished trials are already in W&B. The first sweep's agents died
# at 3 h having completed 8 of 16 requested trials.
#SBATCH --time=24:00:00
#SBATCH --job-name=ampc_prior_rtb_sweep
#SBATCH --output=slurm_logs/ampc_prior_rtb_sweep_%A_%a.out
#SBATCH --error=slurm_logs/ampc_prior_rtb_sweep_%A_%a.err

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: sbatch --array=1-2 jobs/prior_rtb_sweep_agent_mila.sh <sweep_id> [trials]" >&2
  echo "Create the sweep first:" >&2
  echo "  wandb sweep --project ampc-prior-rtb --entity models-mila5723 \\" >&2
  echo "    config/ampc/prior_rtb_sweep4.yaml" >&2
  exit 2
fi

SWEEP_ID="$1"
TRIAL_COUNT="${2:-60}"
WANDB_ENTITY_NAME="${WANDB_ENTITY_NAME:-models-mila5723}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-ampc-prior-rtb}"

if [[ ! "${TRIAL_COUNT}" =~ ^[0-9]+$ ]] || (( TRIAL_COUNT < 1 )); then
  echo "trials must be a positive integer, got '${TRIAL_COUNT}'." >&2
  exit 2
fi
# A bare id, or the fully qualified entity/project/id that `wandb sweep` echoes.
if [[ ! "${SWEEP_ID}" =~ ^[A-Za-z0-9_/-]+$ ]]; then
  echo "sweep_id looks wrong: '${SWEEP_ID}'." >&2
  exit 2
fi
if [[ "${SWEEP_ID}" == */* ]]; then
  SWEEP_PATH="${SWEEP_ID}"
else
  SWEEP_PATH="${WANDB_ENTITY_NAME}/${WANDB_PROJECT_NAME}/${SWEEP_ID}"
fi

cd "${HOME}/activelearning"
source .venv/bin/activate

export PYTHONUNBUFFERED=1
# GP-MoLFormer and its tokenizer must already be cached; a compute node resolving
# them over the network is a failure worth catching here.
export HF_HUB_OFFLINE=1
# Per-agent, so concurrent array tasks do not race ./wandb/latest-run.
export WANDB_DIR="outputs/ampc/prior_rtb/sweep/agent_${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"

for required in \
  "config/ampc/prior_rtb_pretrain.yaml" \
  "config/ampc/reward_shape_svgp300k.yaml" \
  "data/ampc_svgpscale_300000.csv" \
  "outputs/ampc/variational_gp/svgpscale_300000_svgp_M1024_iso_matern_scaled_pm0.0042864_elbo_cosine_lr3e-3_steps10064_fp32/surrogate_state.pt" \
  "cache/ampc/exact_dkl_top_n/svgpscale_300000.npy" \
  "minimol_ampc_encoder/model/final.pt"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing ${required}." >&2
    echo "Every trial scores its samples with the reward-shape GIBBON, so the" >&2
    echo "restored surrogate, its feature cache and the MiniMol encoder are all" >&2
    echo "required here even though training itself uses no surrogate." >&2
    exit 2
  fi
done

mkdir -p "${WANDB_DIR}" outputs/ampc/prior_rtb/sweep slurm_logs

echo "Agent ${SLURM_ARRAY_TASK_ID:-0} running ${TRIAL_COUNT} trials of ${SWEEP_PATH}"
uv run --no-sync wandb agent --count "${TRIAL_COUNT}" "${SWEEP_PATH}"

echo
echo "Done. Rank the trials by 'objective' in the W&B sweep view. 0.0 means the"
echo "trial failed a gate and 'failed_gates' in its summary says which; the"
echo "untrained prior scores 0.00142 and the target is 1/64 = 0.0156, so treat"
echo "anything at or below 0.00142 as having done nothing."
echo "Then re-run the top configurations at 2-3 seeds before trusting a winner,"
echo "and re-run the chosen one without --skip-policy-save to get a checkpoint."
