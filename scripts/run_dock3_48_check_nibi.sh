#!/bin/bash
#SBATCH --cpus-per-task=48
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --job-name=dock3_48_check
#SBATCH --account=def-mailhoto_cpu
#SBATCH --output=slurm_logs/dock3_48_check_%j.out
#SBATCH --error=slurm_logs/dock3_48_check_%j.err
# In-process Dock3Oracle with 48 workers on 192 molecules (no Slurm array).
# 48 cores on a shared node, not --exclusive, so it matches how a real 48-core
# job would run. Submit from the repo root:
#   sbatch scripts/run_dock3_48_check_nibi.sh
cd /home/ethankrz/activelearning
source .venv/bin/activate
export PYTHONUNBUFFERED=1
uv run --no-sync --all-extras python scripts/dock3_48_check.py
