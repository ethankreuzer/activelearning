#!/bin/bash
# Fit time, GPU memory and loop-inference cost of the chosen fixed-feature GP
# (isotropic Matern-5/2 + ScaleKernel, library mean pinned, float32) at
# n = 25,000 (the baseline), 40,000 and 50,000 training molecules.
#
#   sbatch jobs/gp_scaling_benchmark_nibi.sh
#
# Each size runs in its own process, so one running out of memory cannot leave
# the next with a fragmented allocator; the script records an out-of-memory
# stage instead of failing. Results: outputs/ampc/gp_scaling_benchmark/n<N>.json.
#
# Extrapolated from the 25k fits (~1 min, 19 GiB): ~4 min and ~48 GiB at 40k,
# ~8 min and ~75 GiB at 50k, plus scoring 10,000 candidates per size.
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:30:00
#SBATCH --job-name=gp_scaling_bench
#SBATCH --account=def-yvesbrun_gpu
#SBATCH --output=slurm_logs/gp_scaling_bench_%j.out
#SBATCH --error=slurm_logs/gp_scaling_bench_%j.err

set -euo pipefail

cd /home/ethankrz/activelearning
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1

OUTPUT_DIR="outputs/ampc/gp_scaling_benchmark"
mkdir -p "${OUTPUT_DIR}" slurm_logs

for N in 25000 40000 50000; do
  echo "=== n=${N} ($(date +%T))"
  uv run --no-sync python -m scripts.gp_scaling_benchmark \
    --n "${N}" --output "${OUTPUT_DIR}/n${N}.json"
done
nvidia-smi --query-gpu=name,memory.total --format=csv
