#!/bin/bash
# Tiny single-fidelity AmpC run on Nibi (2 rounds x ~50 Dock3 evals), to check
# the full pipeline end to end before the 10M run in job_nibi.sh.
#
# One H100 bundle (~14 cores); num_workers: null sizes the docking pool to the
# 14 CPUs held. Submit from the repo root:
#   sbatch job_nibi_tiny.sh
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=120G
#SBATCH --time=2:00:00
#SBATCH --job-name=ampc_tiny_sf_nibi
#SBATCH --account=def-yvesbrun_gpu
#SBATCH --output=slurm_logs/ampc_tiny_sf_nibi_%j.out
#SBATCH --error=slurm_logs/ampc_tiny_sf_nibi_%j.err
cd /home/ethankrz/activelearning

source .venv/bin/activate

export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

uv run --no-sync python scripts/run_with_sigterm_flush.py \
  config/ampc/s3gfn_minimol_ampc_variational_single_fidelity.yaml \
  config/ampc/overrides/nibi_single_fidelity.yaml \
  config/ampc/overrides/nibi_tiny_single_fidelity.yaml