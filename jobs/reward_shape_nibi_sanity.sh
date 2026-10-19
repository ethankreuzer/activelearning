#!/bin/bash
# Nibi sanity check of one reward-shape arm: does the whole round run end to end
# on this cluster, not does it train well. Same code path as jobs/reward_shape.sh
# (vargrad, fixed floor -150, restored exact-GP surrogate, GIBBON), cut down to
# 100 training steps and 500 Dock3 evaluations instead of 10,000.
#
#   sbatch jobs/reward_shape_nibi_sanity.sh <exponential|power> <beta> <initial_log_z> <n_train_steps>
#
# Reproduces Narval's `sbatch jobs/reward_shape.sh power 0.1 1000 5000` as
#   sbatch jobs/reward_shape_nibi_sanity.sh power 0.1 1000 100
# initial_log_z is unused under vargrad and kept only so the arguments line up
# with jobs/reward_shape.sh.
#
# Inputs that git does not carry, copied from Narval on 2026-10-15:
#   data/ampc_balanced_25000.csv
#   cache/ampc/exact_dkl_top_n/balanced_25000.npy
#   outputs/ampc/exact_dkl_top_n/balanced_25000_nolayer/surrogate_state.pt
#   minimol_ampc_encoder/
#
# One H100 bundle on Nibi: a GPU node is 8 x H100, 112 cores and 2000G, so one
# GPU's share is 14 cores and 250G. Asking for more cores would bill the job as
# more than one GPU. num_workers: null sizes the Dock3 pool to the 14 CPUs held,
# under the 48-worker cap docking needs on Nibi.
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=250G
# Narval measured ~82 min per 10,000 training steps and ~2.9 h for training plus
# the 100k-candidate generation, so ~1.5 h of that is generation. 500 dockings at
# ~32 core-seconds each is ~20 min on 14 CPUs.
#SBATCH --time=4:00:00
#SBATCH --job-name=ampc_reward_shape_nibi_sanity
# The best-placed GPU account on Nibi on 2026-10-15 (FairShare 0.54, no usage
# here yet). Re-check with `sshare -l -A <acct>` before resubmitting.
#SBATCH --account=def-yvesbrun_gpu
#SBATCH --output=slurm_logs/ampc_reward_shape_nibi_sanity_%j.out
#SBATCH --error=slurm_logs/ampc_reward_shape_nibi_sanity_%j.err

set -euo pipefail

TRANSFORM="${1:-}"
BETA="${2:-}"
INITIAL_LOG_Z="${3:-}"
N_TRAIN_STEPS="${4:-}"
if [[ ! "${TRANSFORM}" =~ ^(exponential|power)$ \
      || ! "${BETA}" =~ ^[0-9]*\.?[0-9]+$ \
      || ! "${INITIAL_LOG_Z}" =~ ^-?[0-9]*\.?[0-9]+$ \
      || ! "${N_TRAIN_STEPS}" =~ ^[0-9]+$ ]]; then
  echo "usage: sbatch jobs/reward_shape_nibi_sanity.sh <exponential|power> <beta> <initial_log_z> <n_train_steps>" >&2
  exit 2
fi

LOG_SCORE_FLOOR="-150"
LOSS="vargrad"
LOG_Z_LR="0.01"
DOCK_BUDGET="500"

ARM="nibi_sanity_${TRANSFORM}_beta${BETA}_${LOSS}_floor${LOG_SCORE_FLOOR}_steps${N_TRAIN_STEPS}_dock${DOCK_BUDGET}"
# The job id is in the directory so a rerun never lands on an earlier pool.
OUTPUT_DIR="outputs/ampc/reward_shape_nibi_sanity/${ARM}_${SLURM_JOB_ID:-local}"

cd /home/ethankrz/activelearning
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1   # compute nodes have no internet
# Offline writes to ${WANDB_DIR}; sync from a login node after `wandb login`.
export WANDB_MODE=offline

STATE_PATH="outputs/ampc/exact_dkl_top_n/balanced_25000_nolayer/surrogate_state.pt"
if [[ ! -f "${STATE_PATH}" ]]; then
  echo "${STATE_PATH} not found: copy it from Narval first." >&2
  exit 2
fi

export WANDB_DIR="${OUTPUT_DIR}"
mkdir -p "${WANDB_DIR}" slurm_logs

export WANDB_NOTES="NIBI SANITY CHECK, not a result. Checks that one reward-shape round runs end to end on Nibi: ${N_TRAIN_STEPS} S3-GFN training steps (${LOSS}, ${TRANSFORM} reward, beta=${BETA}, log-score floor ${LOG_SCORE_FLOOR}), 100k-candidate generation, GIBBON selection, and ${DOCK_BUDGET} Dock3 evaluations instead of 10,000. Surrogate restored from ${STATE_PATH}, copied from Narval. Mirrors Narval's jobs/reward_shape.sh power 0.1 1000 5000 apart from the step count and docking budget.
Submitted with: sbatch jobs/reward_shape_nibi_sanity.sh $*   (Slurm job ${SLURM_JOB_ID:-unknown})"
printf '%s\n' "${WANDB_NOTES}" > "${OUTPUT_DIR}/arm_description.txt"

RUN_NAME="${ARM//_/-}-${SLURM_JOB_ID:-local}"
TAGS="nibi,sanity-check,reward-shape,${TRANSFORM},beta${BETA},${LOSS},floor${LOG_SCORE_FLOOR},steps${N_TRAIN_STEPS},dock${DOCK_BUDGET}"
# Restated whole, as in jobs/reward_shape.sh: lists are replaced on merge.
LOGGER="logger={type: MultiLogger, loggers: [{type: ConsoleLogger, project_name: ampc-reward-shape, run_name: ${RUN_NAME}}, {type: WandbLogger, project_name: ampc-reward-shape, run_name: ${RUN_NAME}, entity: models-mila5723, group: nibi-sanity, tags: [${TAGS}]}]}"

# The Nibi overlay supplies every Narval-specific oracle path: the receptor
# files, dockenv_nibi.sh (the default dockenv.sh is under Narval's
# /project/rrg-mailhoto) and the 2022 dock64 build. Its run_writer.output_dir is
# overridden below.
uv run --no-sync python scripts/run_with_sigterm_flush.py \
  config/ampc/reward_shape.yaml \
  config/ampc/overrides/nibi_single_fidelity.yaml \
  "sampler.reward_transform=${TRANSFORM}" \
  "sampler.beta=${BETA}" \
  "sampler.n_train_steps=${N_TRAIN_STEPS}" \
  "sampler.initial_log_z=${INITIAL_LOG_Z}" \
  "sampler.loss=${LOSS}" \
  "sampler.log_score_floor=${LOG_SCORE_FLOOR}" \
  "sampler.log_z_learning_rate=${LOG_Z_LR}" \
  "budget.available_budget=${DOCK_BUDGET}" \
  "budget.schedule.value=${DOCK_BUDGET}" \
  "run_writer.output_dir=${OUTPUT_DIR}" \
  "sampler.reject_log_path=${OUTPUT_DIR}/rejects_round_{round}.csv" \
  "sampler.pool_log_path=${OUTPUT_DIR}/generated_pool_round_{round}.csv" \
  "${LOGGER}"

echo
echo "Done. Sync from a login node with:"
echo "  wandb sync ${OUTPUT_DIR}/wandb/offline-run-*"
