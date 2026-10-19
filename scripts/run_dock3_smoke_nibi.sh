#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000M
#SBATCH --time=1:00:00
#SBATCH --job-name=dock3_smoke_nibi
#SBATCH --account=def-mailhoto_cpu
#SBATCH --output=slurm_logs/dock3_smoke_nibi_%j.out
#SBATCH --error=slurm_logs/dock3_smoke_nibi_%j.err
# Nibi copy of scripts/run_dock3_smoke.sh. Submit from the repo root:
#   sbatch scripts/run_dock3_smoke_nibi.sh
cd /home/ethankrz/activelearning

source .venv/bin/activate

# Offline mode is harmless on Nibi (compute nodes do have internet) and keeps
# behaviour identical to Narval.
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1

# smoke_test_slurm_dock3.py takes one config plus key=value overrides only, so
# merge the Narval base config with the Nibi overlay (lists replaced wholesale,
# interpolations kept unresolved) into a single file first.
MERGED="$SLURM_TMPDIR/multi_fidelity_nibi_merged.yaml"
uv run --no-sync python -c "
import sys
from omegaconf import OmegaConf
OmegaConf.save(OmegaConf.merge(*[OmegaConf.load(p) for p in sys.argv[2:]]), sys.argv[1])
" "$MERGED" \
  config/ampc/s3gfn_minimol_ampc_variational_multi_fidelity_narval.yaml \
  config/ampc/overrides/nibi_multi_fidelity.yaml

# runtime.device=cpu: CPU-only parent allocation, as on Narval.
uv run --no-sync --all-extras python scripts/smoke_test_slurm_dock3.py \
  "$MERGED" \
  runtime.device=cpu \
  --output nibi_slurm_dock3_smoke_results.csv
