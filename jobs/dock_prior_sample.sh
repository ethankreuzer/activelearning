#!/bin/bash
# CPU-only Slurm job array: each task docks one shard of the 100k GP-MoLFormer
# prior sample with Dock3Oracle. No GPU is requested.
#
# Step 2 of SURROGATE_EVAL_PLAN.md. Submit from the repo root:
#
#   sbatch jobs/dock_prior_sample.sh
#
# Size of the job (estimates, not measured here): the base config budgets ~32
# core-seconds per molecule, so 100k molecules is ~3.2M core-seconds (~890
# core-hours). 40 tasks x 32 cores is 1280 cores, about 40-45 minutes if the
# tasks all run at once, plus load imbalance.
#
# Each task's progress is appended to data/gpmolformer_prior_docking/shard_XXXX.csv
# after every chunk, so a task that times out or is killed resumes where it
# stopped. Rerun just the unfinished tasks with the same shard count:
#
#   sbatch --array=3,7 jobs/dock_prior_sample.sh
#
# To change the shard count, set it for sbatch and match the array range:
#
#   NUM_SHARDS=80 sbatch --array=0-79 jobs/dock_prior_sample.sh
#
# Then merge with jobs/merge_dock_prior_shards.sh (same NUM_SHARDS), e.g. chained:
#
#   sbatch --dependency=afterok:<array job id> jobs/merge_dock_prior_shards.sh
#SBATCH --array=0-39
#SBATCH --cpus-per-task=32
# Memory per core is a guess: docking itself is light, and the Python process
# imports torch and loads the hit-rate tables.
#SBATCH --mem-per-cpu=2000M
# ~42 min of docking per 2,500-molecule shard if ~32 core-seconds per molecule
# holds; a timed-out task just resumes.
#SBATCH --time=4:00:00
#SBATCH --job-name=dock_prior
#SBATCH --account=def-yvesbrun_cpu
#SBATCH --output=slurm_logs/dock_prior_%A_%a.out
#SBATCH --error=slurm_logs/dock_prior_%A_%a.err
cd /home/ethankrz/activelearning

source .venv/bin/activate

# Python block-buffers stdout when it is redirected to a file; flush progress
# lines as they happen instead.
export PYTHONUNBUFFERED=1

# Compute nodes are offline: sync the environment from a login node first and
# use --no-sync here.
export HF_HUB_OFFLINE=1

NUM_SHARDS="${NUM_SHARDS:-40}"

# runtime.device=cpu overrides the base config's `cuda`: nothing here uses a GPU.
# The oracle settings (dockfiles, hit-rate fits, timeouts) come from the base config;
# num_workers is null there, so the thread pool is sized from the 32 cores held.
uv run --no-sync python scripts/dock_prior_sample.py \
  config/ampc/s3gfn_minimol_ampc_variational_single_fidelity.yaml \
  runtime.device=cpu \
  --input data/gpmolformer_prior_100k.csv \
  --output-dir data/gpmolformer_prior_docking \
  --shard "${SLURM_ARRAY_TASK_ID}" \
  --num-shards "${NUM_SHARDS}"
