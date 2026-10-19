#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=0:30:00
#SBATCH --job-name=dock3_192_check
#SBATCH --account=def-mailhoto_cpu
#SBATCH --output=slurm_logs/dock3_192_check_%j.out
#SBATCH --error=slurm_logs/dock3_192_check_%j.err
# Coordinator only -- it just submits and polls the one array task, which
# requests its own 192 cores. Submit from the repo root:
#   sbatch scripts/run_dock3_192_check_nibi.sh
cd /home/ethankrz/activelearning
source .venv/bin/activate
uv run --no-sync --all-extras python scripts/dock3_192_check.py
