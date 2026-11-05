#!/bin/bash
# Tests for the dataset-mode switch (subset vs full archive), latent-variance
# calibration tracking, and run provenance; then the whole suite as a
# regression check on the uncommitted main.py / diagnostics changes.
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --job-name=ampc_test_dsmode_calib
#SBATCH --account=def-yvesbrun_cpu
#SBATCH --output=slurm_logs/ampc_test_dsmode_calib_%j.out
#SBATCH --error=slurm_logs/ampc_test_dsmode_calib_%j.err

set -uo pipefail
cd /home/ethankrz/activelearning
source .venv/bin/activate
export PYTHONUNBUFFERED=1 WANDB_MODE=offline

echo "=== targeted: new and changed code ==="
uv run --no-sync python -m pytest \
  tests/test_dataset_mode_switch.py \
  tests/surrogate/test_training_subset.py \
  tests/surrogate/test_variance_calibration.py \
  tests/monitoring \
  tests/surrogate/test_fixed_feature_gp.py \
  tests/test_run_provenance.py \
  -q -rfE
targeted=$?

echo "=== full suite ==="
# These three import scripts that were never committed (absent from all git
# history), so they abort collection for the whole suite.
uv run --no-sync python -m pytest tests -q -rfE \
  --ignore=tests/scripts/test_convert_al_d0.py \
  --ignore=tests/scripts/test_diagnose_ampc_acquisition.py \
  --ignore=tests/scripts/test_diagnose_mes_gibbon.py
full=$?

echo "targeted exit=${targeted} full exit=${full}"
exit $(( targeted || full ))
