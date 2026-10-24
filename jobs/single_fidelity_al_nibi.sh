#!/bin/bash
# Multi-round single-fidelity active learning on Nibi: the whole loop, not one
# round of it. The GP refits each round on a stratified subset of a growing
# archive, an S3-GFN policy trains against its acquisition, the selector's
# top-k are docked, and the results go back into the archive.
#
#   sbatch jobs/single_fidelity_al_nibi.sh              (the 2-round test)
#   sbatch jobs/single_fidelity_al_nibi.sh <run_name> <rounds> <docks_per_round>
#
# EVERY setting lives in config/ampc/subset_fixed_gp.yaml. This script passes
# no `sampler.*` overrides: the reward-shape arms took their real reward, beta,
# loss and floor from overrides here, so their YAML never described the run.
# The only overrides below are the run's label and, for a non-default size, the
# round count and docking budget. The validated config -- every field including
# the defaults -- is logged to W&B under `effective.*`, so the run states what
# it ran whatever is passed.
#
# Resources. One H100 and 24 CPUs: the CPUs are the docking workers, and the
# 48-worker per-node cap is the ceiling. Over 14 CPUs per GPU likely bills as
# more than one GPU on this cluster; that is accepted for the throughput.
#
# Time. The 2-round test: per round ~28 min of policy training (2,500 steps),
# ~5 min generating the 100k pool, the pool scored in chunks of 64, a ~1 min GP
# fit, and 1,000 docking evaluations at ~32 core-seconds each, so ~22 min on 24
# workers. Call it 1-1.5 h per round. 6 hours is generous for two rounds and
# leaves room for the live encoding of the part of ampc_331k that the feature
# store does not already hold, which is re-done each round.
#
# A crash does not lose finished stages: the surrogate fit, the policy training
# and the docking each write their metrics, figures and results to disk as they
# complete (stage_history.jsonl in the run directory). Re-check the fair share
# before every submit:
#   sshare -l -A def-bengioy_gpu,def-mailhoto_gpu,def-yvesbrun_gpu

#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=250G
#SBATCH --time=6:00:00
#SBATCH --job-name=single_fidelity_al
#SBATCH --account=def-yvesbrun_gpu
#SBATCH --output=slurm_logs/single_fidelity_al_%j.out
#SBATCH --error=slurm_logs/single_fidelity_al_%j.err

set -euo pipefail

RUN_NAME="${1:-test-2round-1000dock-seed42}"
ROUNDS="${2:-}"
DOCKS_PER_ROUND="${3:-}"
if [[ ! "${RUN_NAME}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "run_name must be a path- and W&B-safe label; got '${RUN_NAME}'." >&2
  exit 2
fi
if [[ -n "${ROUNDS}${DOCKS_PER_ROUND}" ]]; then
  if [[ ! "${ROUNDS}" =~ ^[0-9]+$ || ! "${DOCKS_PER_ROUND}" =~ ^[0-9]+$ \
        || "${ROUNDS}" -lt 1 || "${DOCKS_PER_ROUND}" -lt 1 ]]; then
    echo "usage: sbatch jobs/single_fidelity_al_nibi.sh <run_name> [<rounds> <docks_per_round>]" >&2
    echo "Pass both of rounds and docks_per_round, or neither to use the config's." >&2
    exit 2
  fi
fi

cd /home/ethankrz/activelearning
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1   # compute nodes have no internet

CONFIG="config/ampc/subset_fixed_gp.yaml"
OUTPUT_DIR="outputs/ampc/single_fidelity_al/${RUN_NAME}"
# One directory per run: artifacts, the stage history and the offline W&B run
# all land here, so a rerun cannot write into another run's.
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "${OUTPUT_DIR} already exists; pick another run_name or move it aside." >&2
  exit 2
fi
mkdir -p "${OUTPUT_DIR}" slurm_logs

# The login node has the network; this node does not. Offline writes under
# WANDB_DIR, which is this run's own directory, and is synced afterwards.
export WANDB_MODE=offline
export WANDB_DIR="${OUTPUT_DIR}"

OVERRIDES=("run_name=${RUN_NAME}")
if [[ -n "${ROUNDS}" ]]; then
  OVERRIDES+=(
    "budget.max_rounds=${ROUNDS}"
    "budget.schedule.value=${DOCKS_PER_ROUND}"
    "budget.available_budget=$((ROUNDS * DOCKS_PER_ROUND))"
  )
fi

export WANDB_NOTES="Single-fidelity active learning, run ${RUN_NAME}: the full loop over rounds, which the reward-shape arms only ever ran one of.
Each round: reselect the GP's training set from the archive (stratified FPS, every row at y >= 0.5 kept outright, band edges pinned to the round-0 balanced quantiles), refit the isotropic Matern GP with its mean pinned at the 10M library mean 0.03956, train the S3-GFN policy for the configured steps against GIBBON on that GP (power reward, beta 0.1, VarGrad, fixed floor -150 nats), generate 100k molecules, dock the top-k by acquisition, and add them to the archive.
THE QUESTION: does the distribution of docked y rise round over round? Read dataset/general/docked/frac_above_0.5 and its siblings. The archive STARTS as the 25,000-molecule balanced set, which holds 9 rows at y >= 0.5, so the tail has to be grown by the loop rather than handed to it.
Watch alongside it: surrogate/eval_set/ampc_331k/* is the GP scored on one fixed set of 331,480 docked molecules every round, the only constant scale on which the surrogate's own health can be compared across rounds; surrogate/fixed_feature_gp/hyperparameters/outputscale over .../noise is the signal-to-noise ratio, which already fell from ~4.3 to ~3.0 between round 0 and the round-1 simulation and is the earliest sign of the reward flattening; surrogate/subset_gp/train/frac_above_* is how enriched the set the GP was fitted on has become; sampler/s3gfn/reward/above_floor_fraction_median is the share of each batch not tied at the floor; and sampler/s3gfn/train/unique_rate is the mode-collapse check that moves while validity stays high.
Caveat for a short test run: 1,000 docks per round gives ~70 molecules above the threshold at round-1 hit rates, enough to prove the loop runs end to end and far too few to measure enrichment from.
Round 0's selection is the identity, because the archive equals the training budget, so that round has no band metrics. Expected, not a fault."
printf '%s\n' "${WANDB_NOTES}" > "${OUTPUT_DIR}/run_description.txt"

# run_with_sigterm_flush turns Slurm's SIGTERM into a KeyboardInterrupt, so the
# loop's handler commits the dying round's telemetry instead of losing it.
# --no-sync keeps uv from re-resolving dependencies over a network this node
# does not have.
uv run --no-sync python scripts/run_with_sigterm_flush.py \
  "${CONFIG}" \
  "${OVERRIDES[@]}"

echo
echo "Done. Outputs in ${OUTPUT_DIR}."
echo "Sync from a login node with:"
echo "  wandb sync ${OUTPUT_DIR}/wandb/offline-run-*"
