#!/bin/bash
# Exact count of the 10M rows at or above the y of several pProp cut-offs, using
# the oracle's hit-rate model. Writes
# analyze_archive_diversity/results/pprop_threshold_counts.csv; the table is
# also printed to the .out log.
#
#   sbatch jobs/count_pprop_threshold_nibi.sh
#   sbatch jobs/count_pprop_threshold_nibi.sh --pprops 4.0,4.25,4.5,4.75
#
# Cost: one pass over the 693 MB CSV's y column, a minute or two on one core.
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=0:15:00
#SBATCH --job-name=ampc_count_pprop_threshold
#SBATCH --account=def-yvesbrun_cpu
#SBATCH --output=slurm_logs/ampc_count_pprop_threshold_%j.out
#SBATCH --error=slurm_logs/ampc_count_pprop_threshold_%j.err

set -euo pipefail

cd /home/ethankrz/activelearning
source .venv/bin/activate
export PYTHONUNBUFFERED=1

uv run --no-sync python analyze_archive_diversity/count_pprop_threshold.py "$@"
