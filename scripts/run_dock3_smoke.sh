#!/bin/bash
  #SBATCH --cpus-per-task=8
#SBATCH --mem=64000M
#SBATCH --time=1:00:00
#SBATCH --job-name=dock3_smoke
#SBATCH --account=def-bengioy_cpu
#SBATCH --output=slurm_logs/dock3_smoke_%j.out
#SBATCH --error=slurm_logs/dock3_smoke_%j.err
cd /home/ethankrz/activelearning

source .venv/bin/activate

# wandb.init() defaults to online mode and blocks on api.wandb.ai, which this
# node cannot reach. Offline mode writes to ./wandb/offline-run-*; sync from a
# login node afterwards: wandb sync wandb/offline-run-*
export WANDB_MODE=offline

# The S3GFN sampler loads GP-MoLFormer from the HuggingFace cache in $HOME.
# Pre-fetch it from a login node; offline mode stops transformers from stalling
# on hub metadata checks the compute node cannot make.
export HF_HUB_OFFLINE=1

# Narval compute nodes have no outbound internet, so the environment must already
# be synced from a login node (`uv sync --all-extras`) before submitting this job.
# --no-sync keeps `uv run` from re-resolving dependencies over the network.
#
# The smoke test itself only submits and polls a nested `sbatch --array` job for
# DOCK3 (see SlurmDock3Oracle) once the 1,000 molecules are generated here.
#
# runtime.device=cpu overrides the base config's `cuda` default so this job can
# run on a CPU-only allocation (faster queue wait than an A100 node); GP-MoLFormer
# generation is slower on CPU but the smoke test only needs 1,000 molecules.
uv run --no-sync --all-extras python scripts/smoke_test_slurm_dock3.py \
  config/ampc/s3gfn_minimol_ampc_variational_multi_fidelity_narval.yaml \
  runtime.device=cpu \
  --output narval_slurm_dock3_smoke_results.csv
