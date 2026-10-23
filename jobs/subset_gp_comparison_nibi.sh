#!/bin/bash
# Step 2 of analyze_archive_diversity/ARCHIVE_DIVERSITY_ANALYSIS.md: fit the
# loop's GP (isotropic Matern + ScaleKernel, mean pinned at the library mean,
# float32) on candidate 25k training subsets of the 10M archive and evaluate
# each on the reward-shape round's 9,782 docked molecules, the docked
# GP-MoLFormer prior set, val_set and ampc_331k.
#
# The default arms are step 4a: the balanced set's farthest-point selection
# with every library row at y >= 0.5 (pProp ~4.85) kept outright, next to the
# plain selection, on seeds 43 and 44. Step 3 group B, part 1 rebuilt the
# balanced 25k set with its own selection (scripts/stratified_fps.py, the
# settings of jobs/exact_dkl_balanced_prepare.sh) on seeds 42, 43 and 44, next
# to the balanced set itself. Seed 42 is the original's, so it should rebuild the
# balanced set (the log reports each rebuild's overlap with it); 43 and 44 are
# fresh draws.
#
#   sbatch jobs/subset_gp_comparison_nibi.sh
#   sbatch jobs/subset_gp_comparison_nibi.sh --arms balanced,stratified_fps:45
#
# Earlier steps, with --out set to their folder: step 2b was balanced plus
# random:42:<t> and kmeans_medoid_below:42:<t> for each threshold t; steps 2c
# and 3A are STEP_2C_ARMS and STEP_3A_ARMS in the script. An arm ending in
# @<floor> sets the GP's noise_std_floor (y units).
#
# Writes analyze_archive_diversity/results/subset_gp_fps_keep_above/. Cost
# (estimated): ~5 min to encode and load, under a minute per farthest-point
# selection (pools of ~910k rows, gathered from the GPU), ~1.5 min per fit
# plus predictions (five here); expect 20-30 min.
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1:30:00
#SBATCH --job-name=ampc_subset_gp_comparison
#SBATCH --account=def-yvesbrun_gpu
#SBATCH --output=slurm_logs/ampc_subset_gp_comparison_%j.out
#SBATCH --error=slurm_logs/ampc_subset_gp_comparison_%j.err

set -euo pipefail

cd /home/ethankrz/activelearning
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1   # compute nodes may have no internet

uv run --no-sync python analyze_archive_diversity/subset_gp_comparison.py "$@"
