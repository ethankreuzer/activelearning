#!/bin/bash
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=48
# All memory on a 48-core A100 node (RealMemory=510000M). The 10M initial dataset
# peaked at ~478 GB on earlier runs, and Narval rejects --mem=0 unless the whole
# node (all 4 GPUs) is requested, and this code uses one GPU.
#SBATCH --mem=510000M
# The earlier size-study fits took ~13 min on top of loading the 10M rows and the
# cached features. Steps 5 adds per-epoch scoring of 130k rows and a final pass over
# the generated set, so 2 h; tighten once the first run reports its own timings.
#SBATCH --time=2:00:00
#SBATCH --job-name=ampc_eval_fit
#SBATCH --account=def-yvesbrun_gpu
#SBATCH --output=slurm_logs/ampc_eval_fit_%j.out/bt
#SBATCH --error=slurm_logs/ampc_eval_fit_%j.err
#
# Steps 4 and 5 of SURROGATE_EVAL_PLAN.md: fit the surrogate on the full 10M, track
# per-epoch metrics on train_random/train_top/val_set, then score the generated set
# once at the end. The objective is the only argument:
#
#   sbatch jobs/surrogate_eval_fit.sh VariationalELBO
#   sbatch jobs/surrogate_eval_fit.sh PredictiveLogLikelihood
#
# Output goes to outputs/ampc/surrogate_eval/<objective>/. Re-running an arm needs
# --overwrite, since the script refuses to replace an existing surrogate_state.pt.
# Arguments after the objective are passed through to the script:
#
#   sbatch jobs/surrogate_eval_fit.sh VariationalELBO --overwrite
#
# After the job, sync the offline run from a login node:
#
#   wandb sync outputs/ampc/surrogate_eval/<objective>/wandb/offline-run-*
OBJECTIVE="${1:?usage: sbatch jobs/surrogate_eval_fit.sh <VariationalELBO|PredictiveLogLikelihood>}"

cd /home/ethankrz/activelearning

source .venv/bin/activate

# Python block-buffers stdout when it is redirected to a file; flush progress
# lines as they happen instead.
export PYTHONUNBUFFERED=1

# Compute nodes are offline: sync from a login node first and use --no-sync, log
# W&B offline and sync it afterwards (`wandb sync <run dir>`), and keep
# transformers/HF from stalling on hub checks.
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline

# Without this both arms write into ./wandb/ at the repo root and race on its
# latest-run symlink; per-arm directories also make `wandb sync` unambiguous.
OUTPUT_DIR="outputs/ampc/surrogate_eval/${OBJECTIVE}"
export WANDB_DIR="${OUTPUT_DIR}"
mkdir -p "${WANDB_DIR}"

# log_space=true/log_output=false: value-scale GIBBON information gain computed
# without underflow. The base config ships log_output=true, whose log-scale scores
# would make the reward beta below meaningless, so the script refuses to run without
# these two overrides.
# Run as a module so the repo root is on sys.path: the script imports
# scripts.surrogate_eval_metrics, which `python scripts/...py` can't resolve.
uv run --no-sync python -m scripts.surrogate_eval_fit \
  config/ampc/s3gfn_minimol_ampc_variational_single_fidelity.yaml \
  "surrogate.training_params.variational_objective=${OBJECTIVE}" \
  acquisition.log_space=true \
  acquisition.log_output=false \
  --output-dir "${OUTPUT_DIR}" \
  --val-csv data/ampc_val_20k.csv \
  --gp-molformer-csv data/gpmolformer_prior_100k_docked.csv \
  --reward-transform exponential \
  --reward-beta 100 \
  --wandb-project ampc-surrogate-eval \
  --wandb-entity models-mila5723 \
  --wandb-group surrogate-eval-step5 \
  --wandb-tags "surrogate-eval,${OBJECTIVE}" \
  --run-name "${OBJECTIVE}-${SLURM_JOB_ID}" \
  "${@:2}"
