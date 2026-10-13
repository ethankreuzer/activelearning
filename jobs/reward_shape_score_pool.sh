#!/bin/bash
# Score a reward-shape arm's generated pool against the prior's molecules.
#
# The question a reward-shape arm asks is whether the trained S3-GFN policy
# produces molecules the acquisition rates above the ones the untrained prior
# produces. Answering it needs two distributions on each pool -- the surrogate
# posterior MEAN and log(acquisition) -- from one surrogate and ONE acquisition
# object.
#
# Why one job and not two: GIBBON ranks candidates against posterior max-values
# sampled when the acquisition is updated. Scoring the generated pool in one run
# and gp_molformer_set in another would draw different max-values, and the two
# log(acquisition) distributions would not be on one scale. The surrogate mean
# is deterministic and would have matched either way, so it is the acquisition
# that forces this. Hence --extra-set rather than a second invocation.
#
# This restores the same fitted state the arm itself restored, so nothing is
# refitted and the comparison is against the surrogate the policy was trained
# against.
#
# Two stages, in one job because both need the GPU:
#   1. exact_dkl_prepare.py resolves the pool CSV and encodes it into the shared
#      cache, as a set named generated_pool. --skip-subsets keeps it off the 10M
#      set, and --sets limits the encoding to the new set alone.
#   2. exact_dkl_top_n.py predicts and scores every study set plus the new one,
#      and pools them for the top-k shares.
#
# Outputs, under outputs/ampc/reward_shape/<arm>_scored/:
#   eval/generated_pool.csv    SMILES, mean, std_total, std_latent,
#                              score_gibbon_value, score_gibbon_log
#   eval/gp_molformer_set.csv  the same columns for the 83,811 prior samples,
#                              rescored here so both share one acquisition
#   eval_summary.json          per-set metrics and
#                              pooled/acquisition/*/top{k}/<set>_share
#
# usage: sbatch jobs/reward_shape_score_pool.sh <arm> [round]
#   arm    directory name under outputs/ampc/reward_shape/, e.g.
#          reward_power_beta0.1_vargrad_floor-150
#   round  zero-padded round index of the pool CSV, default 0001
#
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
# 3h: the fitted arm's own scoring pass fits in 2h, and this adds a 100k-molecule
# set to encode and score. Raise it if the chunked kernel proves slower than the
# arm's scoring stage suggested.
#SBATCH --time=3:00:00
#SBATCH --job-name=ampc_rs_score_pool
# RE-MEASURE BEFORE EVERY SUBMIT. def-yvesbrun_gpu went from RawShares 16287 to
# 1 within hours on 2026-10-13, so a reading from earlier the same day was
# already wrong. Compare the ACCOUNT row's LevelFS and the USER row's FairShare:
#   sshare -l -A def-bengioy_gpu,def-mailhoto_gpu,def-yvesbrun_gpu
#SBATCH --account=def-bengioy_gpu
#SBATCH --output=slurm_logs/ampc_rs_score_pool_%j.out
#SBATCH --error=slurm_logs/ampc_rs_score_pool_%j.err

set -euo pipefail

ARM="${1:-}"
ROUND="${2:-0001}"
if [[ -z "${ARM}" ]]; then
  echo "usage: sbatch jobs/reward_shape_score_pool.sh <arm> [round]" >&2
  exit 2
fi

cd /home/ethankrz/activelearning
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1   # compute nodes have no internet
export WANDB_MODE=offline

RUN_DIR="outputs/ampc/reward_shape/${ARM}"
POOL_CSV="${RUN_DIR}/generated_pool_round_${ROUND}.csv"
OUTPUT_DIR="${RUN_DIR}_scored"
export WANDB_DIR="${OUTPUT_DIR}"

# The surrogate every reward-shape arm restores. Scoring against anything else
# would answer a different question.
FITTED_DIR="outputs/ampc/exact_dkl_top_n/balanced_25000_nolayer"
STATE_PATH="${FITTED_DIR}/surrogate_state.pt"
CACHE_DIR="cache/ampc/exact_dkl_top_n"

if [[ ! -f "${POOL_CSV}" ]]; then
  echo "${POOL_CSV} not found." >&2
  echo "The arm writes it when generation finishes; check it got that far." >&2
  exit 2
fi
if [[ ! -f "${STATE_PATH}" ]]; then
  echo "${STATE_PATH} not found: fit the no-layer arm first with" >&2
  echo "  sbatch jobs/exact_dkl_balanced.sh 25000 none" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"
POOL_ROWS=$(($(wc -l < "${POOL_CSV}") - 1))
echo "Scoring ${POOL_ROWS} generated molecules from ${POOL_CSV}."

# Stage 1: resolve and encode the pool into the shared cache.
uv run --no-sync python -m scripts.exact_dkl_prepare \
  --extra-set "generated_pool=${POOL_CSV}" \
  --sets generated_pool \
  --skip-subsets \
  --cache-dir "${CACHE_DIR}" \
  --overwrite

# Stage 2: predict and score every set, including the new one, in one process.
# The overrides reproduce the balanced_25000_nolayer architecture exactly;
# --load-state refuses a state whose resolved_config.json disagrees on the
# dataset, the surrogate or the precision, which is the guard that this matches.
uv run --no-sync python -m scripts.exact_dkl_top_n \
  config/ampc/exact_dkl_top_n.yaml \
  "dataset.initial_data.path=data/ampc_balanced_25000.csv" \
  "surrogate.encoder.feature_cache_path=${CACHE_DIR}/balanced_25000.npy" \
  "surrogate.encoder.activation=none" \
  "surrogate.encoder.latent_dim=null" \
  "runtime.precision=32" \
  --load-state "${STATE_PATH}" \
  --extra-set generated_pool \
  --output-dir "${OUTPUT_DIR}" \
  --cache-dir "${CACHE_DIR}" \
  --eval-chunk-size 2000 \
  --score-chunk-size 64 \
  --overwrite \
  --wandb-project ampc-reward-shape \
  --wandb-entity models-mila5723 \
  --wandb-group reward-shape \
  --wandb-tags "reward-shape,scored,acq-vs-prior,generated-pool,n25000,nolayer,fp32,restored-state" \
  --run-name "${ARM//_/-}-pool-scored-${SLURM_JOB_ID:-local}" \
  "${@:3}"

echo
echo "Done. Sync from a login node with:"
echo "  wandb sync ${OUTPUT_DIR}/wandb/offline-run-*"
echo "Compare the two distributions with:"
echo "  ${OUTPUT_DIR}/eval/generated_pool.csv"
echo "  ${OUTPUT_DIR}/eval/gp_molformer_set.csv"
echo "and the pooled top-k shares in ${OUTPUT_DIR}/eval_summary.json."
