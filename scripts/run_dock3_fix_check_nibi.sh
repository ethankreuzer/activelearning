#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000M
#SBATCH --time=0:40:00
#SBATCH --job-name=dock3_fix_check_nibi
#SBATCH --account=def-mailhoto_cpu
#SBATCH --output=slurm_logs/dock3_fix_check_nibi_%j.out
#SBATCH --error=slurm_logs/dock3_fix_check_nibi_%j.err
# Validates the JAVA_TOOL_OPTIONS single-thread fix on a smaller footprint
# (20 workers/shard instead of 192) before trusting it on the full run.
# Submit from the repo root:
#   sbatch scripts/run_dock3_fix_check_nibi.sh
cd /home/ethankrz/activelearning

source .venv/bin/activate

export WANDB_MODE=offline
export HF_HUB_OFFLINE=1

MERGED="$SLURM_TMPDIR/multi_fidelity_nibi_fix_check_merged.yaml"
uv run --no-sync python -c "
import sys
from omegaconf import OmegaConf
OmegaConf.save(OmegaConf.merge(*[OmegaConf.load(p) for p in sys.argv[2:]]), sys.argv[1])
" "$MERGED" \
  config/ampc/s3gfn_minimol_ampc_variational_multi_fidelity_narval.yaml \
  config/ampc/overrides/nibi_multi_fidelity.yaml \
  config/ampc/overrides/nibi_dock3_fix_check.yaml

uv run --no-sync --all-extras python scripts/smoke_test_slurm_dock3.py \
  "$MERGED" \
  runtime.device=cpu \
  --output nibi_slurm_dock3_fix_check_results.csv
