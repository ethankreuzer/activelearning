#!/bin/bash
# Build the data subsets and the per-set MiniMol feature caches for the
# exact-DKL top-n study. Run once, before any arm.
#
#   sbatch jobs/exact_dkl_prepare.sh
#   sbatch jobs/exact_dkl_prepare.sh --sets ampc_331k,gp_molformer_set   # resume
#
# This is the only job in the study that runs MiniMol inference: about 465k
# molecules in total (train_random 100k, train_top 10k, top-2000/3000, val 20k,
# gp_molformer ~90k, ampc_331k 331k, olivier 1.5k), once, instead of four times
# across the arms. 16 CPUs because graphium's featurization is CPU-bound and
# parallel; 64 GB covers the 10M SMILES list (~1.4 GB) plus the largest feature
# matrix (331k x 512 float32 = 679 MB) with room to spare. Not the 48-core
# 510 GB node the variational jobs needed: nothing here holds 10M encoded rows.
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --job-name=ampc_exact_dkl_prep
#SBATCH --account=def-yvesbrun_gpu
#SBATCH --output=slurm_logs/ampc_exact_dkl_prep_%j.out
#SBATCH --error=slurm_logs/ampc_exact_dkl_prep_%j.err

set -euo pipefail

cd /home/ethankrz/activelearning
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1   # compute nodes have no internet

mkdir -p slurm_logs cache/ampc/exact_dkl_top_n

uv run --no-sync python -m scripts.exact_dkl_prepare \
  --cache-dir cache/ampc/exact_dkl_top_n \
  --training-csv data/10M_unif_random_subset.csv \
  --val-csv data/ampc_val_20k.csv \
  --gp-molformer-csv data/gpmolformer_prior_100k_docked.csv \
  --ampc-331k-csv data/ampc_331k_with_y.csv \
  --olivier-csv data/Olivier_Invitro.csv \
  --train-sizes 2000,3000 \
  --n-train-random 100000 \
  --n-train-top 10000 \
  --eval-seed 42 \
  --checkpoint-path minimol_ampc_encoder/model/final.pt \
  --package-path minimol_ampc_encoder \
  --encoder-device cuda \
  --encoder-batch-size 64 \
  "$@"
