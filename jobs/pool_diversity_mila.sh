#!/bin/bash
# Nearest-neighbour and median Tanimoto of every molecule in each K-ladder arm's 100k pool.
# See scripts/pool_diversity.py. Writes <arm>/pool_diversity_round_0001.npz; skips done arms.
#   sbatch jobs/pool_diversity_mila.sh
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --job-name=ampc_pool_diversity
#SBATCH --output=slurm_logs/ampc_pool_diversity_%j.out
#SBATCH --error=slurm_logs/ampc_pool_diversity_%j.err
set -euo pipefail
cd /home/mila/k/kreuzere/activelearning
PYTHONUNBUFFERED=1 .venv/bin/python scripts/pool_diversity.py
