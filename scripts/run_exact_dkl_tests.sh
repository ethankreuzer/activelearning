#!/bin/bash
# Run the test suite for the exact-DKL top-n study.
#
# RUN THIS INSIDE A SLURM ALLOCATION, NEVER ON A LOGIN NODE.
#
#   salloc --account=def-yvesbrun_cpu --time=1:00:00 --cpus-per-task=4 --mem=16G
#   # then, inside the allocation:
#   bash scripts/run_exact_dkl_tests.sh
#
# The script refuses to run outside an allocation rather than trusting the
# caller, because a login-node pytest run puts the account at risk of a ban.
#
# Everything here is CPU-only and uses fakes or tiny sequence encoders; no
# MiniMol checkpoint, no GPU and no 10M-row data are needed.

set -uo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  cat >&2 <<'EOF'
REFUSING TO RUN: no Slurm allocation detected ($SLURM_JOB_ID is unset).

Running tests on a login node is forbidden for this account. Get an
allocation first, then re-run this script inside it:

  salloc --account=def-yvesbrun_cpu --time=1:00:00 --cpus-per-task=4 --mem=16G
  bash scripts/run_exact_dkl_tests.sh
EOF
  exit 2
fi

case "$(hostname -s)" in
  narval[0-9]*|beluga[0-9]*|cedar[0-9]*|graham[0-9]*)
    echo "REFUSING TO RUN: hostname $(hostname -s) looks like a login node." >&2
    exit 2
    ;;
esac

cd /home/ethankrz/activelearning

echo "Allocation ${SLURM_JOB_ID} on $(hostname -s)"
echo

# --no-sync throughout: compute nodes have no internet, so a sync would fail.
NEW_TESTS=(
  tests/scripts/test_stage_profiler.py
  tests/scripts/test_exact_dkl_prepare.py
  tests/scripts/test_exact_dkl_top_n.py
  tests/surrogate/dkl/test_dkl_exact_targets.py
  tests/applications/molecules/test_minimol_encoder_activation.py
  tests/test_encoder_config_fields.py
)

# These must pass UNCHANGED after the shared-helper extraction. If any of them
# fails, the re-export list in scripts/surrogate_eval_fit.py is incomplete --
# fix the re-exports, not the tests.
REGRESSION_TESTS=(
  tests/scripts/test_surrogate_eval_fit.py
  tests/scripts/test_surrogate_eval_fit_eval.py
  tests/scripts/test_surrogate_eval_metrics.py
  tests/scripts/test_surrogate_eval_transform.py
  tests/scripts/test_surrogate_dataset_size_study.py
  tests/surrogate/dkl/
  tests/acquisition/test_botorch_acquisition.py
  tests/acquisition/test_candidate_set.py
  tests/surrogate/test_botorch_surrogate.py
  tests/test_config_unions.py
  tests/test_example_configs.py
  tests/test_config_compatibility.py
)

status=0

echo "=== New tests for this study ==="
uv run --no-sync pytest "${NEW_TESTS[@]}" -v || status=1

echo
echo "=== Regression: must pass unchanged after the refactor ==="
uv run --no-sync pytest "${REGRESSION_TESTS[@]}" || status=1

echo
echo "=== Full suite ==="
uv run --no-sync pytest -q || status=1

echo
if [[ "${status}" -eq 0 ]]; then
  echo "All green. Next: sbatch jobs/exact_dkl_prepare.sh"
else
  echo "FAILURES above. Do not submit the prep job until they are resolved." >&2
fi
exit "${status}"
