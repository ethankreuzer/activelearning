#!/bin/bash
# Map the 331k AmpC subset's pProp to binding probabilities with the fitted
# HitRateModel, and write the validation subset (all hits + a random sample of the
# rest, with weights). Small, but still a job so nothing runs on the login node.
# No GPU.
#
#   sbatch jobs/make_val_set.sh
#
# Outputs (under data/, gitignored): ampc_331k_with_y.csv, ampc_val_20k.csv, plus
# ampc_val_20k.json (summary) and ampc_val_20k.png (the mapping and the agreement
# between the two routes to y). Read the .json before using the labels.
#SBATCH --cpus-per-task=1
#SBATCH --mem=8000M
#SBATCH --time=0:30:00
#SBATCH --job-name=ampc_val_set
#SBATCH --account=def-yvesbrun_cpu
#SBATCH --output=slurm_logs/ampc_val_set_%j.out
#SBATCH --error=slurm_logs/ampc_val_set_%j.err
cd /home/ethankrz/activelearning

source .venv/bin/activate

export PYTHONUNBUFFERED=1

uv run --no-sync python scripts/make_val_set.py \
  config/ampc/s3gfn_minimol_ampc_variational_single_fidelity.yaml \
  --input data/ampc_subset_331k.csv \
  --output-all data/ampc_331k_with_y.csv \
  --output-val data/ampc_val_20k.csv
