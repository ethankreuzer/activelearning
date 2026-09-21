#!/bin/bash
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=48
# All memory on a 48-core A100 node (RealMemory=510000M). Narval rejects --mem=0
# unless the whole node (all 4 GPUs) is requested, and this code uses one GPU.
#SBATCH --mem=510000M
# Survival trial: does one round of the full 10M config get through encoding and
# the GP fit? Estimated encoding alone is ~3-7 h per round, and the first run
# also pays for building the persistent feature cache, so allow headroom.
#SBATCH --time=60:00:00
#SBATCH --job-name=ampc_10m_sf
#SBATCH --account=def-yvesbrun_gpu
#SBATCH --output=slurm_logs/ampc_10m_sf_%j.out
#SBATCH --error=slurm_logs/ampc_10m_sf_%j.err
cd /home/ethankrz/activelearning

source .venv/bin/activate

# wandb.init() defaults to online mode and blocks on api.wandb.ai, which this
# node cannot reach - measured hanging >9min, ignoring WANDB_INIT_TIMEOUT.
# Offline mode writes to ./wandb/offline-run-*; sync from a login node afterwards:
#   wandb sync wandb/offline-run-*
export WANDB_MODE=offline

# The S3GFN sampler loads GP-MoLFormer from the HuggingFace cache in $HOME.
# Pre-fetch it from a login node; offline mode stops transformers from stalling
# on hub metadata checks the compute node cannot make.
export HF_HUB_OFFLINE=1

# Narval compute nodes have no outbound internet, so the environment must already
# be synced from a login node (`uv sync --all-extras`) before submitting this job.
# --no-sync keeps `uv run` from re-resolving dependencies over the network.
#
# This run is expected to hit the wall clock rather than exhaust its budget.
# Slurm then sends SIGTERM (SIGKILL 60s later, cluster KillWait). Python ignores
# SIGTERM by default, which would discard the entire buffered wandb run, so go
# through the wrapper that turns SIGTERM into a clean interpreter shutdown.
#
# The single-fidelity config has no CometLogger and its own output_dir, so it
# needs no overlay.
uv run --no-sync python scripts/run_with_sigterm_flush.py \
  config/ampc/s3gfn_minimol_ampc_variational_single_fidelity.yaml
