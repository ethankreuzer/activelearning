#!/bin/bash
# Build the nested training sets for the sparse-variational-GP scaling study,
# plus the held-out generated tail set the arms are all evaluated on.
#
#   sbatch jobs/build_svgp_scaling_sets_nibi.sh
#   sbatch jobs/build_svgp_scaling_sets_nibi.sh --random-control 100000
#
# Writes data/ampc_svgpscale_{50000,100000,300000}.csv, their feature caches in
# cache/ampc/exact_dkl_top_n/, data/ampc_gentail_<k>.csv with its cache, and
# cache/ampc/exact_dkl_top_n/svgpscale_selection.json recording every band
# population, budget and spread measurement.
#
# The three sets are NESTED -- 50k is a subset of 100k is a subset of 300k --
# and all three hold the same 16,292 molecules above y 0.2617 (pProp 3.5). Only
# the bulk grows, so comparing the arms measures how much bulk the surrogate can
# absorb at fixed tail supervision. scripts/build_svgp_scaling_sets.py's
# docstring explains why the bands stop at pProp 2.0 rather than running to 0,
# and why the filler is split evenly rather than in proportion to the library.
#
# Check the design before spending the allocation:
#   python -m scripts.build_svgp_scaling_sets --dry-run
# which reports the bands and budgets without reading the store or selecting
# anything. It is light enough for a short salloc.
#
# Resources. Nothing is encoded for the three library arms: all 10M
# fingerprints are already in cache/ampc/s3gfn_minimol_ampc_fingerprints.npy,
# in library order, so the pool's features are a slice of that 20 GB file and
# opening it verifies the order by hash. The GPU is for the farthest-point
# distance updates (~82,000 sequential steps per band at the 300k budget, on a
# pool of at most 655,000 x 512) and for the one genuinely live piece of work,
# MiniMol on the few thousand generated tail molecules, which are not in the
# library. Memory is dominated by holding the 10M SMILES twice (the CSV and the
# store's manifest list) alongside the ~2.2 GB pool slice.
#
# A MIG instance, not a full H100, because neither half of the work needs one.
# FPS holds one band's pool on the device as an (N, 512) float32 matrix, and the
# largest is the 300k arm's lowest band at 8 x 81,916 = 655,328 rows = 1.34 GB,
# so ~3 GB with the per-step transient; the store is memory-mapped and never
# reaches the GPU. The distance updates come to ~180 TB of memory traffic, a
# minute at an H100's full bandwidth and ~8 minutes at a 2g instance's share,
# against the 3-hour limit. The binding resource is the host: a Nibi GPU node is
# 112 cores and 2000G across 8 H100s, so 2/8 of a GPU is a fair share of 3.5
# cores and 62G, and the CPU and memory below sit just inside it.
#SBATCH --gpus=h100_2g.20gb:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=60G
#SBATCH --time=3:00:00
#SBATCH --job-name=svgp_scale_prep
# Re-check the fair share before every submit:
#   sshare -l -A def-bengioy_gpu,def-mailhoto_gpu,def-yvesbrun_gpu
#SBATCH --account=def-yvesbrun_gpu
#SBATCH --output=slurm_logs/svgp_scale_prep_%j.out
#SBATCH --error=slurm_logs/svgp_scale_prep_%j.err

set -euo pipefail

cd /home/ethankrz/activelearning
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1   # compute nodes have no internet

mkdir -p slurm_logs

uv run --no-sync python -m scripts.build_svgp_scaling_sets "$@"

echo
echo "Selection recorded in:"
echo "  cache/ampc/exact_dkl_top_n/svgpscale_selection.json"
echo "Fit an arm with:"
echo "  sbatch jobs/variational_gp_scaling_nibi.sh pll 50000"
