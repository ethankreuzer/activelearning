#!/bin/bash
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
# GP-MoLFormer is small and no training data is loaded; the config is only
# validated, never built beyond the sampler.
#SBATCH --mem=32G
# Generation speed has not been measured yet; 100k molecules at batch 128 is
# expected well under this. Check the log's generation_s to tighten it.
#SBATCH --time=2:00:00
#SBATCH --job-name=gpmolformer_prior_100k
#SBATCH --account=def-yvesbrun_gpu
#SBATCH --output=slurm_logs/gpmolformer_prior_100k_%j.out
#SBATCH --error=slurm_logs/gpmolformer_prior_100k_%j.err
cd /home/ethankrz/activelearning

source .venv/bin/activate

# Python block-buffers stdout when it is redirected to a file; flush progress
# lines as they happen instead.
export PYTHONUNBUFFERED=1

# Compute nodes are offline. GP-MoLFormer and its tokenizer are already in the
# HF cache (~/.cache/huggingface/hub); sync from a login node first and use
# --no-sync here.
export HF_HUB_OFFLINE=1

# Step 1 of SURROGATE_EVAL_PLAN.md: 100k unique molecules from the untrained
# GP-MoLFormer prior, through the same S3GFNSampler pipeline as a real round
# (n_train_steps=0 is set by the script). Writes the CSV plus a .json record.
uv run --no-sync python scripts/generate_prior_sample.py \
  config/ampc/s3gfn_minimol_ampc_variational_single_fidelity.yaml \
  --n-samples 100000 \
  --seed 42 \
  --output data/gpmolformer_prior_100k.csv
