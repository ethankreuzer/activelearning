#!/bin/bash
# Re-run sweep 3's REC arm at three seeds AND KEEP THE CHECKPOINTS.
#
#   sbatch --array=1-3 jobs/prior_rtb_rec_checkpoint_mila.sh
#
# WHY THIS ARM AND NOT THE SWEEP WINNER. Sweep 3's objective maximizer
# (lr 5.71e-5, beta 34.9, 704 steps, on-policy 0.25) reaches validity 0.963 --
# BELOW stock GP-MoLFormer's 0.977 -- for 930 floor-clearing molecules per
# 65,536. This arm reaches 0.9805, at or above stock, for 502. The 1.85x is
# real but it is a uniform rescaling, not a better-targeted prior: the two
# above-floor distributions have the same shape (band shares within a few
# points, above-floor medians -405 vs -357, means -376 vs -366) and the SAME
# CEILING, max -4.63 vs -4.87. Nothing in the winner's sample is better; there
# is simply more of it, and 1.8 points of validity bought it.
#
# That trade is the one PRIOR_RTB_SWEEP2_FINDINGS.md exists to refuse. The one
# data point on a sub-stock-validity prior under reward pressure is sweep 2's
# winner collapsing 0.852 -> 0.247 over 2500 steps. The winner also shows the
# first trace of copying the step penalty is there to catch:
# tanimoto_to_tail_max 1.0 (an exact training-tail duplicate; this arm 0.917)
# and frac_in_library 13x higher, 0.000198 vs 1.5e-5, even though the coarse
# memorization_share reads 0.000 for both.
#
# WHAT THIS RUN IS FOR. The reward-shape arms need a prior whose static
# validity is indistinguishable from stock, to test whether validity collapse
# under reward pressure is caused by the prior's validity at all. Every sweep 3
# trial passed --skip-policy-save, so 65 trials produced NO usable checkpoint.
# This run produces one, three times, so the chosen seed can be handed over and
# the other two say whether its numbers were a seed draw.
#
# WHY THREE SEEDS. config/ampc/prior_rtb_sweep3.yaml's closing note: the sweep
# bought zero replicates (no configuration ever ran twice), and
# REWARD_SHAPE_UNDERFLOW_GAP.md section 10 item 4 records one cell producing
# three different outcomes across seeds. 0.00766 from a single trial is not yet
# a measurement. Believe the arm only if all three clear validity 0.975 and the
# rate spread is small against the 0.00142 untrained baseline.
#
#SBATCH --partition=main
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
# A sweep 3 trial at on-policy 0.0 measured 22 min of training inside a ~30 min
# trial: ~4.8 min pool filter (not cached), 744 steps at ~1.7 s/step, ~7.5 min
# generating 65,536 molecules, ~3 min of GIBBON. 3 h is a wide margin.
#SBATCH --time=03:00:00
#SBATCH --job-name=ampc_prior_rtb_rec
#SBATCH --output=slurm_logs/ampc_prior_rtb_rec_%A_%a.out
#SBATCH --error=slurm_logs/ampc_prior_rtb_rec_%A_%a.err

set -euo pipefail

# The association allows 2 GPUs at a time. With a sweep agent still holding one,
# these three tasks serialize behind it rather than running together; that is
# expected and costs about 1.5 h in total.
SEEDS=(42 43 44)
TASK="${SLURM_ARRAY_TASK_ID:-1}"
if (( TASK < 1 || TASK > ${#SEEDS[@]} )); then
  echo "Array index must be 1-${#SEEDS[@]}, got ${TASK}. Submit with --array=1-3." >&2
  exit 2
fi
SEED="${SEEDS[TASK-1]}"

# SWEEP 3'S REC ARM, trial 11505463_417737_1793691445: objective 0.00524,
# validity 0.9805, sa_rate 0.9601, rate 0.007660 (5.4x the untrained prior's
# 0.00142), memorization_share 0.000. Pinned here rather than passed on the
# command line so the file records which configuration was replicated.
LEARNING_RATE=4.72e-5
BETA=31.5
N_STEPS=744
ON_POLICY_FRACTION=0.0

OUT_DIR="outputs/ampc/prior_rtb/rec_checkpoint/seed_${SEED}"
WANDB_ENTITY_NAME="${WANDB_ENTITY_NAME:-models-mila5723}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-ampc-prior-rtb}"

cd "${HOME}/activelearning"
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
# Per-task, so concurrent tasks do not race ./wandb/latest-run.
export WANDB_DIR="${OUT_DIR}/wandb"

for required in \
  "config/ampc/prior_rtb_pretrain.yaml" \
  "config/ampc/reward_shape_svgp300k.yaml" \
  "data/ampc_svgpscale_300000.csv" \
  "outputs/ampc/variational_gp/svgpscale_300000_svgp_M1024_iso_matern_scaled_pm0.03956_elbo_cosine_lr3e-3_steps10064_fp32/surrogate_state.pt" \
  "cache/ampc/exact_dkl_top_n/svgpscale_300000.npy" \
  "minimol_ampc_encoder/model/final.pt"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing ${required}." >&2
    echo "The run scores its samples with the reward-shape GIBBON, so the" >&2
    echo "restored surrogate, its feature cache and the MiniMol encoder are all" >&2
    echo "required here even though training itself uses no surrogate." >&2
    exit 2
  fi
done

# The script itself refuses to overwrite an existing ${OUT_DIR}/policy, but
# failing here is clearer than failing after the 4.8 min pool filter.
if [[ -e "${OUT_DIR}/policy" ]]; then
  echo "${OUT_DIR}/policy already exists; refusing to overwrite a checkpoint." >&2
  exit 2
fi

mkdir -p "${WANDB_DIR}" slurm_logs

echo "Seed ${SEED}: lr ${LEARNING_RATE}, beta ${BETA}, ${N_STEPS} steps," \
  "on-policy ${ON_POLICY_FRACTION} -> ${OUT_DIR}"

# NO --skip-policy-save. That flag is the whole reason this run exists.
# Everything else is byte-identical to the sweep command in
# config/ampc/prior_rtb_sweep3.yaml, because the headline rate is only
# comparable with sweep 3's 0.00766 if the evaluation is the same: the same
# --gibbon-config, and --eval-sample-size 65536 (at 2,048 the objective is a
# count of 2-4 molecules with Poisson error +-2).
uv run --no-sync python scripts/pretrain_prior_rtb.py \
  config/ampc/prior_rtb_pretrain.yaml \
  --gibbon-config config/ampc/reward_shape_svgp300k.yaml \
  --out-dir "${OUT_DIR}" \
  --eval-sample-size 65536 \
  --no-price-validity \
  --min-validity 0.95 \
  --min-sa-rate 0.94 \
  --learning-rate "${LEARNING_RATE}" \
  --beta "${BETA}" \
  --n-steps "${N_STEPS}" \
  --on-policy-fraction "${ON_POLICY_FRACTION}" \
  --seed "${SEED}" \
  --run-name "rec_checkpoint_seed_${SEED}" \
  --wandb-group rec_checkpoint \
  --wandb-tags rec_checkpoint,replicate \
  --wandb-project "${WANDB_PROJECT_NAME}" \
  --wandb-entity "${WANDB_ENTITY_NAME}"

echo
echo "Checkpoint: ${OUT_DIR}/policy (~179 MB, the policy weights and tokenizer)."
echo "Compare the three seeds on validity_rate (sweep 3 measured 0.9805; stock"
echo "GP-MoLFormer is 0.977) and on log_gibbon_frac_above_underflow (0.007660"
echo "there, against the untrained prior's 0.00142). If a seed lands below"
echo "validity 0.975 the arm is not the stock-validity prior it looked like,"
echo "and the reward-shape handover should wait."
