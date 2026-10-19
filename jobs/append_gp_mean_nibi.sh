#!/bin/bash
# Append the reward-shape surrogate's posterior mean and total predictive std
# (columns gp_mean, gp_std) to a scored_samples.csv, for every row with a docked y. The surrogate is rebuilt from
# config/ampc/reward_shape.yaml and restored from
# outputs/ampc/exact_dkl_top_n/balanced_25000_nolayer/surrogate_state.pt, the
# same way every reward-shape arm restores it; scripts/append_gp_mean.py checks
# the restore against that fit's own val_set predictions before writing.
#
#   sbatch jobs/append_gp_mean_nibi.sh [csv]
#
# Defaults to analyze_scored_samples_csv/scored_samples.csv (the power
# beta=0.1, 5000-step arm). The CSV is copied to <csv>.bak first.
#
# Cost: restore ~15 s (job 24059495), then MiniMol encoding and prediction of
# ~10k molecules; minutes. A GPU because the exact GP and MiniMol both run on
# cuda in this config.
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:45:00
#SBATCH --job-name=ampc_append_gp_mean
#SBATCH --account=def-yvesbrun_gpu
#SBATCH --output=slurm_logs/ampc_append_gp_mean_%j.out
#SBATCH --error=slurm_logs/ampc_append_gp_mean_%j.err

set -euo pipefail

CSV="${1:-analyze_scored_samples_csv/scored_samples.csv}"

cd /home/ethankrz/activelearning
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1   # compute nodes have no internet

if [[ ! -f "${CSV}" ]]; then
  echo "${CSV} not found." >&2
  exit 2
fi
if [[ ! -e "${CSV}.bak" ]]; then cp "${CSV}" "${CSV}.bak"; fi

uv run --no-sync python -m scripts.append_gp_mean "${CSV}"
