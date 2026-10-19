#!/bin/bash
# One point of the Nibi docking-throughput curve: dock the 1000 molecules a
# round would select (data/dock_scaling_top1000.csv) with as many workers as the
# job has CPUs, and append the timing to outputs/dock_scaling_nibi/summary.csv.
# See scripts/dock_scaling_nibi.py for why the result carries over to a round.
#
# The H100 is held only to land on the same GPU nodes a round runs on (Intel
# 8570, 112 cores); Nibi's CPU partitions are 192-core nodes with other CPUs.
# The GPU sits idle. Billing is set by the cores from 14 up, not the GPU.
#
# Submit one job per core count from the repo root, chained with singleton so
# the points run one at a time and never compete with each other:
#   for c in 14 24 32 48; do
#     sbatch --cpus-per-task=$c jobs/dock_scaling_nibi.sh
#   done
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=14
# The 48-worker docking check peaked at ~31 GB; within one GPU's 250 GB share.
#SBATCH --mem=64G
# 1000 molecules at the worst measured 55 core-seconds each on 14 cores is
# ~65 min; every larger count is faster.
#SBATCH --time=1:30:00
#SBATCH --job-name=dock_scaling_nibi
#SBATCH --dependency=singleton
#SBATCH --account=def-yvesbrun_gpu
#SBATCH --output=slurm_logs/dock_scaling_nibi_%j.out
#SBATCH --error=slurm_logs/dock_scaling_nibi_%j.err

set -euo pipefail

cd /home/ethankrz/activelearning
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1

uv run --no-sync python scripts/dock_scaling_nibi.py \
  --molecules data/dock_scaling_top1000.csv \
  --summary outputs/dock_scaling_nibi/summary.csv \
  --details-dir outputs/dock_scaling_nibi
