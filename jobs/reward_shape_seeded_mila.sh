#!/bin/bash
# One reward-shape arm with the positive replay buffer SEEDED before step 1.
#
#   sbatch jobs/reward_shape_seeded_mila.sh          # seed 42
#   sbatch jobs/reward_shape_seeded_mila.sh 7        # a replicate
#
# WHAT IS DIFFERENT FROM jobs/reward_shape_mila.sh. That script sweeps
# (transform, beta, floor, steps) against an EMPTY positive buffer. This one
# fixes all four at tvdet6ii's values -- power, beta 0.015, floor -1000, 2500
# steps, the best arm of the 22 -- and changes exactly one thing: the positive
# replay buffer starts holding known high-acquisition molecules instead of
# nothing. The shape is held constant so the seeding is the only variable.
#
# WHY. With an empty buffer the start of every round is a lottery. Until some
# generation happens to clear the reward floor, every molecule in the batch is
# tied at it; a constant reward contributes nothing to a variance, so the
# policy is trained by the prior term alone and the reward says nothing at all.
# tvdet6ii waited 750 steps of its 2500 for that first breach, elw5yh9a ~700,
# and the floor -500 arms broke through only 3 times in 8. The policy and the
# buffer are both rebuilt per round (_new_round_model deep-copies the
# pretrained template, _create_replay_buffers runs inside the round), so that
# lottery is re-rolled once per round -- ten independent draws in a ten-round
# run. Seeding removes it rather than managing it.
#
# WHICH MOLECULES, AND WHY SO FEW. data/seed_molecules_svgp300k.csv holds ~256
# molecules chosen by scripts/build_seed_molecules.py from the 4,753 of the
# 300,000 training molecules whose log GIBBON clears the float64 underflow edge
# at -744.44 nats (measured 2026-10-30, job 11456679: median -1864, p99.9
# -149.1, max -4.75, which is the GIBBON ceiling the arms saturate at). They are
# strided across the reward-sorted list, not taken from the top, for two
# reasons. The replay loss is a variance, so it needs SPREAD inside the replay
# batch: the strided set spans ~730 nats (11 nats of beta*spread) where any one
# band of similar size would span a third of that. And replay draws are weighted
# by score - min + 1e-6, so seed COUNT decides how much replay mass is left for
# the policy's own discoveries -- measured, at 256 seeds a discovery at -500
# takes 14.2% of the draws, at 4,753 seeds it takes 0.87%. Reinforcing those
# discoveries is the mechanism behind tvdet6ii's batch max climbing -86.85 ->
# -39.01 -> -10.13 over the 500 steps after its breach, so the seed set is kept
# deliberately small.
#
# THE SEEDS ARE RE-SCORED IN-RUN. Only the SMILES column of the file is read.
# GIBBON redraws its max-value samples on every acquisition update, so the
# scores in that file are NOT on the same scale as the ones this round assigns
# to generated molecules -- and the replay weighting acts on exactly the
# difference between the two. The log_acq column there is provenance only.
#
# WHAT TO READ, in order:
#   reward/above_floor_fraction AT STEP 1. Non-zero means the seeding worked and
#     the lottery is gone. This is the primary readout; everything else is about
#     whether that was worth having.
#   sampler/s3gfn/train/seeded_buffer, and the INFO line printed before training
#     starts, which gives read/synthesizable/retained counts. Retained is below
#     256: the buffer rejects a molecule within 0.75 Tanimoto of a stored one
#     unless it scores higher, so the count depends on how diverse the file is.
#   POOL OVERLAP WITH THE TRAINING SET. The seeds are already-labelled
#     molecules and _generate_final_candidates does not exclude them, so a
#     policy that collapses onto its seeds returns 100,000 candidates whose y is
#     already known while posting an excellent acq/median. Join
#     generated_pool_round_0001.csv against data/ampc_svgpscale_300000.csv on
#     SMILES after the run. This is the failure mode seeding introduces.
#   validity at the step raw_max reaches the GIBBON ceiling (-4.47 to -4.55),
#     against tvdet6ii's 0.9767, and acq/median against its -154 final batch.
#     NOTE tvdet6ii got ~1,750 GRADED steps out of 2500 because it waited 750
#     for its breach; a seeded arm is graded from step 1, so 2500 steps here is
#     ~43% more training-against-reward. Compare at matched graded steps, not
#     matched total steps, and read the per-step line for where validity turns.
#
# ORACLE IS A STUB, as in jobs/reward_shape_mila.sh: ConstantOracle, budget 1.
# Mila has no AmpC receptors and no dock64. IGNORE dataset/general/docked/*,
# surrogate/general/held_out/* and every other y-derived number.
#
# Resources follow jobs/reward_shape_mila.sh: a100, 4 CPUs, 24G -- half of
# main's per-user quota, so two arms run concurrently. 2500 steps is ~43-53 min
# of training at the measured 0.73-0.80 s/step, plus a generation pass of 8-25
# min and ~3.5 min of selection, so 3 h leaves margin.
#SBATCH --partition=main
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=3:00:00
#SBATCH --job-name=ampc_reward_shape_seeded
#SBATCH --output=slurm_logs/ampc_reward_shape_seeded_%j.out
#SBATCH --error=slurm_logs/ampc_reward_shape_seeded_%j.err

set -euo pipefail

SEED="${1:-42}"
if [[ ! "${SEED}" =~ ^[0-9]+$ ]]; then
  echo "usage: sbatch jobs/reward_shape_seeded_mila.sh [seed]" >&2
  exit 2
fi

# Fixed at tvdet6ii's shape. These are deliberately NOT arguments: the point of
# this script is that seeding is the only thing that varies.
TRANSFORM="power"
BETA="0.015"
LOG_SCORE_FLOOR="-1000"
N_TRAIN_STEPS="2500"
LOSS="vargrad"
INITIAL_LOG_Z="1000"
LOG_Z_LR="0.01"

BASE_CONFIG="config/ampc/reward_shape_svgp300k.yaml"
FITTED_DIR="outputs/ampc/variational_gp/svgpscale_300000_svgp_M1024_iso_matern_scaled_pm0.03956_elbo_cosine_lr3e-3_steps10064_fp32"
SURROGATE_TAGS="svgp,M1024,elbo,svgpscale-300k,lr3e-3,steps10064,pinned-mean,fp32"
# A SEPARATE PROJECT from ampc-reward-shape-svgp. Those 22 arms measure shape
# against an empty buffer and are not comparable with these; mixing them is how
# the two reversals in REWARD_SHAPE_UNDERFLOW_GAP.md happened.
WANDB_PROJECT="${WANDB_PROJECT:-ampc-reward-shape-svgp-seeded}"
SEED_BUFFER_PATH="${SEED_BUFFER_PATH:-data/seed_molecules_svgp300k.csv}"

ARM="seeded_reward_${TRANSFORM}_beta${BETA}_floor${LOG_SCORE_FLOOR}_steps${N_TRAIN_STEPS}_s${SEED}"
ARM_SUFFIX="${ARM_SUFFIX:-}"
if [[ -n "${ARM_SUFFIX}" && ! "${ARM_SUFFIX}" =~ ^_[A-Za-z0-9.-]+$ ]]; then
  echo "ARM_SUFFIX must look like _name (letters, digits, '.', '-')." >&2
  exit 2
fi
ARM="${ARM}${ARM_SUFFIX}"

cd "${HOME}/activelearning"
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1   # nothing is downloaded: the prior and encoder are cached

STATE_PATH="${FITTED_DIR}/surrogate_state.pt"
OUTPUT_DIR="outputs/ampc/reward_shape_seeded_mila/${ARM}"
REJECT_LOG_PATH="${OUTPUT_DIR}/rejects_round_{round}.csv"
POOL_LOG_PATH="${OUTPUT_DIR}/generated_pool_round_{round}.csv"

for required in \
  "${STATE_PATH}" \
  "${SEED_BUFFER_PATH}" \
  "data/ampc_svgpscale_300000.csv" \
  "cache/ampc/exact_dkl_top_n/svgpscale_300000.npy" \
  "minimol_ampc_encoder/model/final.pt"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing ${required}." >&2
    echo "This arm restores a surrogate rather than fitting one and seeds its" >&2
    echo "replay buffer from a prepared molecule list. Build that list with:" >&2
    echo "  python scripts/build_seed_molecules.py \\" >&2
    echo "      outputs/ampc/log_acq_distribution/svgp300k_mila.csv \\" >&2
    echo "      --out data/seed_molecules_svgp300k.csv" >&2
    exit 2
  fi
done
# Never overwrite an earlier run's generated pool: it is the arm's result.
if compgen -G "${OUTPUT_DIR}/generated_pool_round_*.csv" > /dev/null; then
  echo "${OUTPUT_DIR} already holds a generated pool from an earlier run." >&2
  echo "Rerun with ARM_SUFFIX=_rerun2 sbatch jobs/reward_shape_seeded_mila.sh $*" >&2
  exit 2
fi

export WANDB_DIR="${OUTPUT_DIR}"
mkdir -p "${WANDB_DIR}" slurm_logs

export WANDB_NOTES="SEEDED REWARD-SHAPE arm ${ARM}, on Mila, WITHOUT DOCKING. One active-learning round training the S3-GFN policy against a fixed GIBBON acquisition on a frozen surrogate, with the positive replay buffer SEEDED before step 1.
THE QUESTION THIS RUN ASKS: does seeding the positive replay buffer remove the warmup lottery -- the 700-1,150 steps an unseeded arm spends with every reward tied at the floor, contributing nothing to a variance -- without the policy simply reproducing the molecules it was seeded with.
WHAT CHANGED, AND ONLY THIS. The shape is fixed at tvdet6ii's (power, beta ${BETA}, floor ${LOG_SCORE_FLOOR}, ${N_TRAIN_STEPS} steps, vargrad), the best of the 22 arms in ampc-reward-shape-svgp: validity 0.9767, yield 0.593, prior divergence 6.65, pool max -4.524 at the GIBBON ceiling, final batch median -154. The single difference is sampler.seed_buffer_path=${SEED_BUFFER_PATH}.
THE SEEDS. ~256 molecules strided across the reward-sorted list of the 4,753 (of 300,000 training molecules) whose log GIBBON clears the float64 underflow edge at -744.44 nats. That distribution was measured on this cluster on 2026-10-30 (job 11456679): median -1864, p99.9 -149.1, max -4.75, 98.4% underflowing. Strided rather than top-N because the replay loss is a variance and needs spread inside the batch (~730 nats here), and because replay draws are weighted by score - min + 1e-6, so seed count decides how much replay mass is left for the policy's own discoveries: measured, 14.2% at 256 seeds against 0.87% at all 4,753. Seeds are RE-SCORED IN-RUN with this round's acquisition -- GIBBON redraws its max-value samples on every update, so stored scores would be on a different scale from the ones this round gives generated molecules.
PRIMARY READOUT: reward/above_floor_fraction AT STEP 1. Non-zero means the lottery is gone. Then sampler/s3gfn/train/seeded_buffer for how many seeds survived 0.75-Tanimoto diversity filtering (below 256; the INFO line before training gives read/synthesizable/retained).
THE FAILURE MODE THIS INTRODUCES: the seeds are already-labelled molecules and _generate_final_candidates does not exclude them, so a policy that collapses onto its seeds returns a pool whose y is already known while posting an excellent acq/median. Join ${POOL_LOG_PATH} against data/ampc_svgpscale_300000.csv on SMILES before believing any acquisition number from this run.
STEP BUDGET CAVEAT: tvdet6ii got ~1,750 GRADED steps of its 2500 because it waited 750 for its first breach. A seeded arm is graded from step 1, so ${N_TRAIN_STEPS} steps here is ~43% more training-against-reward than the arm it is being compared with. By the phase argument in REWARD_SHAPE_UNDERFLOW_GAP.md section 7, that may push the policy into off-manifold reward hacking earlier in wall-clock steps. Compare at matched graded steps and read the per-step reward line for where validity turns over.
ORACLE IS A STUB. oracle=ConstantOracle, budget 1. Mila has no AmpC receptor files and no dock64. IGNORE dataset/general/docked/*, surrogate/general/held_out/* and every other y-derived number.
Arm name: ${ARM}. Submitted with: sbatch jobs/reward_shape_seeded_mila.sh $*   (Slurm job ${SLURM_JOB_ID:-unknown})"
printf '%s\n' "${WANDB_NOTES}" > "${OUTPUT_DIR}/arm_description.txt"

RUN_NAME="${ARM//_/-}-${SLURM_JOB_ID:-local}"
TAGS="reward-shape,seeded-buffer,no-docking,${TRANSFORM},beta${BETA},vargrad,floor${LOG_SCORE_FLOOR},steps${N_TRAIN_STEPS},${SURROGATE_TAGS},restored-state,1-round,seed${SEED}${ARM_SUFFIX:+,suffix${ARM_SUFFIX}}"

ORACLE="oracle={type: ConstantOracle, value: 0.0, fidelity_costs: {1: 1.0}}"
BUDGET="budget={available_budget: 1.0, schedule: {type: constant, value: 1.0}, max_rounds: 1}"
LOGGER="logger={type: MultiLogger, loggers: [{type: ConsoleLogger, project_name: ${WANDB_PROJECT}, run_name: ${RUN_NAME}}, {type: WandbLogger, project_name: ${WANDB_PROJECT}, run_name: ${RUN_NAME}, entity: models-mila5723, group: seeded-buffer, tags: [mila,${TAGS}]}]}"

uv run --no-sync python scripts/run_with_sigterm_flush.py \
  "${BASE_CONFIG}" \
  "sampler.reward_transform=${TRANSFORM}" \
  "sampler.beta=${BETA}" \
  "sampler.n_train_steps=${N_TRAIN_STEPS}" \
  "sampler.initial_log_z=${INITIAL_LOG_Z}" \
  "sampler.loss=${LOSS}" \
  "sampler.log_score_floor=${LOG_SCORE_FLOOR}" \
  "sampler.log_z_learning_rate=${LOG_Z_LR}" \
  "sampler.seed_buffer_path=${SEED_BUFFER_PATH}" \
  "runtime.seed=${SEED}" \
  "run_writer.output_dir=${OUTPUT_DIR}" \
  "sampler.reject_log_path=${REJECT_LOG_PATH}" \
  "sampler.pool_log_path=${POOL_LOG_PATH}" \
  "${ORACLE}" \
  "${BUDGET}" \
  "${LOGGER}" \
  "${@:2}"
