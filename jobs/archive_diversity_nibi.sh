#!/bin/bash
# Feature-space diversity of the archive, read with y:
# analyze_archive_diversity/archive_diversity.py on a uniform random sample of
# 1M of the 10M rows. See
# analyze_archive_diversity/ARCHIVE_DIVERSITY_ANALYSIS.md for the method.
#
#   sbatch jobs/archive_diversity_nibi.sh                 # 1M uniform rows
#   sbatch jobs/archive_diversity_nibi.sh --rows 200000   # extra args pass through
#
# Writes analyze_archive_diversity/results/uniform_rows_<N>/. Cost: the first
# run (job 24679483, first 1M rows) took 1.5 min. Sampled rows are spread over
# the whole store, so loading now reads all 20.5 GB once, and the CSV's SMILES
# column is read in full for the row check; expect a few minutes more.
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:45:00
#SBATCH --job-name=ampc_archive_diversity
#SBATCH --account=def-yvesbrun_gpu
#SBATCH --output=slurm_logs/ampc_archive_diversity_%j.out
#SBATCH --error=slurm_logs/ampc_archive_diversity_%j.err

set -euo pipefail

cd /home/ethankrz/activelearning
source .venv/bin/activate
export PYTHONUNBUFFERED=1

uv run --no-sync python analyze_archive_diversity/archive_diversity.py --rows 1000000 "$@"
