#!/bin/bash
# Nibi copy of job.sh (single-fidelity 10M AmpC run).
#
# Nibi GPU nodes: 8 x H100-80gb, 112 cores, 2048000M. One H100 bundle is
# ~14 cores / ~250 GB. The 10M run peaks at ~478 GB, so memory is charged as
# ~2 GPU-equivalents whatever we do; 28 cores stay inside those 2 bundles.
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=28
#SBATCH --mem=510000M
#SBATCH --time=60:00:00
#SBATCH --job-name=ampc_10m_sf_nibi
#SBATCH --account=def-yvesbrun_gpu
#SBATCH --output=slurm_logs/ampc_10m_sf_nibi_%j.out
#SBATCH --error=slurm_logs/ampc_10m_sf_nibi_%j.err
cd /home/ethankrz/activelearning

source .venv/bin/activate

# Kept from Narval. Nibi compute nodes have internet, so these are optional,
# but offline wandb avoids the api.wandb.ai hang seen on Narval. Sync later with
#   wandb sync wandb/offline-run-*
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

# The wrapper turns a wall-clock SIGTERM into a clean interpreter shutdown so a
# timeout still flushes the buffered wandb run. The Nibi overlay sets the
# DOCK3 paths and a separate run_writer.output_dir.
uv run --no-sync python scripts/run_with_sigterm_flush.py \
  config/ampc/s3gfn_minimol_ampc_variational_single_fidelity.yaml \
  config/ampc/overrides/nibi_single_fidelity.yaml
