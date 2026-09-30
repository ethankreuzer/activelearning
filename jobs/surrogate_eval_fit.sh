#!/bin/bash
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=48
# All memory on a 48-core A100 node (RealMemory=510000M). The 10M initial dataset
# peaked at ~478 GB on earlier runs, and Narval rejects --mem=0 unless the whole
# node (all 4 GPUs) is requested, and this code uses one GPU.
#SBATCH --mem=510000M
# The earlier size-study fits took ~13 min on top of loading the 10M rows and the
# cached features; 1 h leaves room for the load and the train-set selection.
#SBATCH --time=1:00:00
#SBATCH --job-name=ampc_eval_fit
#SBATCH --account=def-yvesbrun_gpu
#SBATCH --output=slurm_logs/ampc_eval_fit_%j.out
#SBATCH --error=slurm_logs/ampc_eval_fit_%j.err
#
# Step 4 of SURROGATE_EVAL_PLAN.md: fit the surrogate once on the full 10M and save
# its state plus the two `train` eval sets. The objective is the only argument:
#
#   sbatch jobs/surrogate_eval_fit.sh VariationalELBO
#   sbatch jobs/surrogate_eval_fit.sh PredictiveLogLikelihood
#
# Output goes to outputs/ampc/surrogate_eval/<objective>/.
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

uv run --no-sync python scripts/surrogate_eval_fit.py \
  config/ampc/s3gfn_minimol_ampc_variational_single_fidelity.yaml \
  "surrogate.training_params.variational_objective=${OBJECTIVE}" \
  --output-dir "outputs/ampc/surrogate_eval/${OBJECTIVE}" \
  --wandb-project ampc-surrogate-eval \
  --run-name "fit-${OBJECTIVE}"
