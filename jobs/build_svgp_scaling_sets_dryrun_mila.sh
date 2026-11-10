#!/bin/bash
# Report the band populations and per-arm budgets of the SVGP scaling sets
# WITHOUT selecting, encoding or writing anything.
#
#   sbatch jobs/build_svgp_scaling_sets_dryrun_mila.sh
#
# The cheap way to see what a change to the band edges, --keep-above or the fill
# would do before spending the real build's GPU hours on it. It is a separate
# job from jobs/build_svgp_scaling_sets_mila.sh because that one guards against
# overwriting existing sets and allocates a GPU, neither of which a dry run
# wants.
#
# CPU only: a dry run reads the library's y, streams the pProp column and counts
# bands. No GPU, no feature store, no farthest-point sampling. Memory is the one
# real requirement -- a pandas read of the 10M-row library needs a few GB, and
# an interactive session or a login shell will be OOM-killed (exit 137) long
# before it finishes, with no traceback.
#SBATCH --partition=long-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --job-name=svgp_scale_dryrun
#SBATCH --output=slurm_logs/svgp_scale_dryrun_%j.out
#SBATCH --error=slurm_logs/svgp_scale_dryrun_%j.err

set -euo pipefail

cd "${HOME}/activelearning"

for required in \
  data/10M_unif_random_subset.csv \
  data/ampc_unif_random_10M.csv.gz
do
  if [[ ! -e "${required}" ]]; then
    echo "Missing ${required}." >&2
    exit 2
  fi
done

source .venv/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1

mkdir -p slurm_logs

uv run --no-sync python -m scripts.build_svgp_scaling_sets \
  --skip-gentail --dry-run "$@"
