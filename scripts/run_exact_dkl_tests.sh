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
  tests/applications/molecules/test_minimol_encoder_no_projection.py
  tests/test_encoder_config_fields.py
  tests/surrogate/dkl/test_dkl_noise_bound.py
  tests/surrogate/dkl/test_dkl_prior_mean.py
  tests/scripts/test_exact_dkl_rescore.py
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

# Pre-existing breakage, unrelated to this study. Verified against
# `git diff b6afcfa..HEAD`: none of my commits touch these files. Excluded so
# that a green run is a meaningful signal; each is listed with its own reason
# rather than swept under a blanket ignore.
#
#   test_ampc_single_fidelity_reward_transform_overlays_parse[power]
#     The test expects sampler.beta == 0.5, but
#     config/ampc/overrides/reward_power.yaml deliberately sets 1.0 and says so
#     in a comment. The test is stale, not the config.
#   test_molecule_s3gfn_minimol_slurm_dock3_config_parses
#     config/molecules/s3gfn_minimol_slurm_dock3.yaml does not exist.
#   test_convert_al_d0 / test_diagnose_ampc_acquisition / test_diagnose_mes_gibbon
#     Orphaned test modules: scripts/convert_al_d0.py,
#     scripts/diagnose_ampc_acquisition.py and scripts/diagnose_mes_gibbon.py
#     are absent, so these three fail at *collection* and abort the whole run.
PREEXISTING_DESELECT=(
  --deselect
  "tests/test_example_configs.py::test_ampc_single_fidelity_reward_transform_overlays_parse[power]"
  --deselect
  "tests/test_example_configs.py::test_molecule_s3gfn_minimol_slurm_dock3_config_parses"
)
PREEXISTING_IGNORE=(
  --ignore tests/scripts/test_convert_al_d0.py
  --ignore tests/scripts/test_diagnose_ampc_acquisition.py
  --ignore tests/scripts/test_diagnose_mes_gibbon.py
)

# Four more pre-existing failures that only the full suite reaches. Again
# verified against `git diff 0104988..HEAD`: neither source file has changed
# since this study began.
#
#   test_num_workers_auto_resolves_from_slurm
#     Environment-dependent. resolve_num_workers() takes
#     min(cpu_affinity, SLURM_CPUS_PER_TASK) on purpose, so no stale env var can
#     oversubscribe the allocation. The test monkeypatches the env var to 64 but
#     not the affinity, so it only passes on a host with >= 64 available CPUs.
#     It fails in any small salloc and would pass on a login node.
#   test_select_stratified_* / test_select_kmeans_respects_strata_and_subsampling
#     The default strata are incompatible with the test's dataset size. Quantiles
#     (0.9, 0.99, 0.999) make stratum 3 the top 0.1%, and fractions
#     (0.375, 0.25, 0.1875, 0.1875) allot 0.1875 * 64 = 12 inducing points to it.
#     The test uses 10,000 rows, so that stratum holds 10 -- fewer than its
#     allotment -- and _allocation raises. The real Step 8 runs used 10M rows,
#     where the top 0.1% is 10,000 rows, which is why this never surfaced.
#     Implication worth knowing: inducing_init="stratified" raises on any
#     training set below roughly 12,000 rows.
PREEXISTING_FULL_SUITE_DESELECT=(
  --deselect
  "tests/applications/molecules/test_dock3_oracle.py::TestDock3OracleConstruction::test_num_workers_auto_resolves_from_slurm"
  --deselect
  "tests/surrogate/test_inducing_init.py::test_select_stratified_draws_from_each_stratum"
  --deselect
  "tests/surrogate/test_inducing_init.py::test_select_stratified_is_seeded"
  --deselect
  "tests/surrogate/test_inducing_init.py::test_select_kmeans_respects_strata_and_subsampling"
)

status=0

echo "=== New tests for this study ==="
uv run --no-sync pytest "${NEW_TESTS[@]}" -v || status=1

echo
echo "=== Regression: must pass unchanged after the refactor ==="
uv run --no-sync pytest "${REGRESSION_TESTS[@]}" \
  "${PREEXISTING_DESELECT[@]}" || status=1

echo
echo "=== Full suite (pre-existing breakage excluded; see above) ==="
uv run --no-sync pytest -q \
  "${PREEXISTING_IGNORE[@]}" \
  "${PREEXISTING_DESELECT[@]}" \
  "${PREEXISTING_FULL_SUITE_DESELECT[@]}" || status=1

echo
echo "=== Pre-existing failures, confirmed unrelated to this study ==="
echo "Seven in total, each documented with its reason in this script."
echo "To see them, and decide separately whether to fix them:"
echo "  uv run --no-sync pytest \\"
echo "    tests/test_example_configs.py \\"
echo "    tests/surrogate/test_inducing_init.py \\"
echo "    tests/applications/molecules/test_dock3_oracle.py \\"
echo "    tests/scripts/test_convert_al_d0.py \\"
echo "    tests/scripts/test_diagnose_ampc_acquisition.py \\"
echo "    tests/scripts/test_diagnose_mes_gibbon.py"

echo
if [[ "${status}" -eq 0 ]]; then
  echo "All green. Next: sbatch jobs/exact_dkl_prepare.sh"
else
  echo "FAILURES above. Do not submit the prep job until they are resolved." >&2
fi
exit "${status}"
