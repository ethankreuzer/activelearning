#!/bin/bash
# Diagnose the restore-check failure of jobs/append_gp_mean_nibi.sh (job
# 24105437): predicts 256 val_set molecules from cached and from live MiniMol
# fingerprints and compares both with the fit's eval/val_set.csv. Writes
# nothing; read the .err log.
#
#   sbatch jobs/diagnose_gp_restore_nibi.sh
#
# Cost: same restore as the append job (~6 min in 24105437), then 256 molecules.
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --job-name=ampc_diagnose_gp_restore
#SBATCH --account=def-yvesbrun_gpu
#SBATCH --output=slurm_logs/ampc_diagnose_gp_restore_%j.out
#SBATCH --error=slurm_logs/ampc_diagnose_gp_restore_%j.err

set -euo pipefail

cd /home/ethankrz/activelearning
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1   # compute nodes have no internet

uv run --no-sync python -m scripts.diagnose_gp_restore "$@"
