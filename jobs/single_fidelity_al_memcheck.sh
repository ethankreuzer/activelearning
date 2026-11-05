#!/bin/bash
# Does an exact GP of a given training-set size survive TWO rounds of the real
# loop, on the real H100, with the S3-GFN policy and the MiniMol encoder
# resident on the same device? One question, one parameter.
#
#   sbatch jobs/single_fidelity_al_memcheck.sh <subset_size> [<archive_csv>] [<run_name>]
#
#   sbatch jobs/single_fidelity_al_memcheck.sh 45000
#   sbatch jobs/single_fidelity_al_memcheck.sh 40000
#
# DATASET_MODE=full probes the OTHER dataset regime, where the GP fits the whole
# archive instead of a subset of it:
#
#   DATASET_MODE=full sbatch jobs/single_fidelity_al_memcheck.sh \
#     25000 data/ampc_balanced_25000.csv sf-full-smoke
#
# In that mode subset_size is IGNORED -- there is no subset, the fit is the
# archive -- so the number is only still required to keep one usage shape, and
# the probe's two rounds fit the archive's own size and then that plus the
# round's acquisitions. It is an argument-level check that the regime is wired
# up, not a size probe: with the budget shortened to 10 docks a round, round 2
# fits ~10 rows more than round 1, nowhere near any ceiling. Read
# surrogate/subset_gp/train/archive_fraction, which must be 1.000 in both
# rounds, and the growing subset_size in the summary below.
#
# Replaces jobs/single_fidelity_al_50k_memcheck.sh, which hard-coded 50,000 and
# answered it: job 26159147 completed round 1 (fit 736 s at n=50,000) and died
# in ROUND 2's fit with `torch.OutOfMemoryError: Tried to allocate 9.31 GiB`
# (one 50,000^2 float32 matrix) with 2.83 GiB free of 79.18 GiB.
#
# WHY TWO ROUNDS, which job 26159147 proved is not optional. Round 1's fit ran
# at 12:50 and the policy did not load until 13:02, so round 1 fit on an
# effectively empty GPU and PASSED. Round 2's fit, with the policy resident,
# failed. A one-round check reports a clean pass and sends a 48 h run to its
# death about three hours in. Round 2 is the measurement; round 1 is the setup.
#
# WHAT THE SIZE DOES, and why no new dataset is needed. The GP fits
# `surrogate.training_subset.size` rows selected from the archive, so the fit
# can be sized independently of the archive: pass 45000 against a 50,000-row
# archive and the GP fits 45,000 rows. Kernel memory and Cholesky time depend
# on that number alone, not on the y distribution, so any archive at least that
# large is a valid probe and the fitted hyperparameters are not the point.
# Defaults to data/ampc_svgpscale_50000.csv (50,000 rows); for a smaller probe
# data/ampc_fps_keep05_n45000_45000.csv (45,000) also exists.
#
# WHERE THE MEASUREMENT NOW COMES FROM. The fit records its CUDA memory into
# the stage record, so this no longer infers headroom from a crash. Read, from
# stage_history.jsonl, for BOTH rounds:
#
#   profiling/surrogate/cuda_reserved_before_fit_gib   policy + encoder +
#                                                      whatever the previous
#                                                      round's acquisition left
#   profiling/surrogate/cuda_peak_reserved_fit_gib     the fit itself
#   profiling/surrogate/cuda_headroom_at_peak_gib      how close it came
#
# The round-2 'reserved_before' is the number worth having: it is the term the
# isolated benchmark in outputs/ampc/gp_scaling_benchmark/ cannot see, and the
# reason that benchmark's 50k pass did not predict the loop's 50k failure.
#
# WHAT A PASS MEANS. Two fits at this size completed with the policy and
# encoder resident. It does NOT certify round 10: the archive grows, so the
# StratifiedFPS selection works harder each round, though the GP's own fit
# stays pinned at the subset size and is the dominant term.
#
# WHAT IS SHORTENED, and what each shortcut hides:
#   sampler.n_train_steps=50     The policy need only be RESIDENT and training,
#                                not converged. 5,000 steps adds ~1 h a round
#                                and moves no memory peak.
#   sampler.n_samples=10000      Pool scoring peaked at 47.06 GiB at n=50,000,
#                                under the fit's 74.66, and it is chunked, so a
#                                tenth of the pool hides no peak. The
#                                136 s/100k scoring cost is not re-timed.
#   oracle=ConstantOracle        Docking answers nothing about VRAM and costs
#                                ~2 h a round. Every y-derived number here
#                                describes a stub: IGNORE
#                                dataset/general/docked/*, the held-out metrics
#                                and the surrogate's own fit quality.
#   budget 10/round              Enough that the archive grows, so round 2's
#                                subset selection is a real selection rather
#                                than the identity round 1 performs.
#
# The 331k held-out eval is deliberately NOT disabled. It is ~5 min a round of
# diagnostics this check does not need, but it predicts 331,480 molecules while
# the GP's cached training data is resident, so it is a genuine competitor for
# the memory being measured. Disabling it would make the job shorter and the
# check weaker.
#
# Expect ~25-50 min: the fit dominates and ran 736 s at n=50,000, less below
# that. An OOM arrives sooner than a pass, inside round 2's fit.
#
# Same GPU as the real run (h100:1) and the same host memory, because a
# different GPU model would make the measurement meaningless. CPUs dropped from
# 24 to 4: those are docking workers and there is no docking here.
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=2:00:00
#SBATCH --job-name=sf_al_memcheck
#SBATCH --account=def-alexhg_gpu
#SBATCH --output=slurm_logs/sf_al_memcheck_%j.out
#SBATCH --error=slurm_logs/sf_al_memcheck_%j.err

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: sbatch jobs/single_fidelity_al_memcheck.sh <subset_size> [<archive_csv>] [<run_name>]" >&2
  exit 2
fi

SUBSET_SIZE="$1"
ARCHIVE_CSV="${2:-data/ampc_svgpscale_50000.csv}"
RUN_NAME="${3:-sf-${SUBSET_SIZE}-memcheck}"
# From the environment, not a 4th positional, because ${@:4} is already the
# passthrough for extra overrides. sbatch exports the submitting environment.
DATASET_MODE="${DATASET_MODE:-subset}"

if [[ "${DATASET_MODE}" != "subset" && "${DATASET_MODE}" != "full" ]]; then
  echo "DATASET_MODE must be 'subset' or 'full'; got '${DATASET_MODE}'." >&2
  exit 2
fi

if [[ ! "${SUBSET_SIZE}" =~ ^[0-9]+$ ]] || (( SUBSET_SIZE < 1 )); then
  echo "subset_size must be a positive integer; got '${SUBSET_SIZE}'." >&2
  exit 2
fi
if [[ ! "${RUN_NAME}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "run_name must be a path- and W&B-safe label; got '${RUN_NAME}'." >&2
  exit 2
fi

cd /home/ethankrz/activelearning

if [[ ! -f "${ARCHIVE_CSV}" ]]; then
  echo "archive_csv does not exist: ${ARCHIVE_CSV}" >&2
  exit 2
fi

# The GP cannot fit more rows than the archive holds, and a subset size equal to
# the archive makes round 1 an identity selection. Checked here rather than left
# to fail inside the selector after the encoder and the policy have loaded.
ARCHIVE_ROWS="$(( $(wc -l < "${ARCHIVE_CSV}") - 1 ))"
if (( SUBSET_SIZE > ARCHIVE_ROWS )); then
  echo "subset_size ${SUBSET_SIZE} exceeds the ${ARCHIVE_ROWS} rows of ${ARCHIVE_CSV}." >&2
  exit 2
fi

source .venv/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1   # compute nodes have no internet

OUTPUT_DIR="outputs/ampc/single_fidelity_al/${RUN_NAME}"
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "${OUTPUT_DIR} already exists; pick another run_name or move it aside." >&2
  exit 2
fi
mkdir -p "${OUTPUT_DIR}" slurm_logs

export WANDB_MODE=offline
export WANDB_DIR="${OUTPUT_DIR}"

if [[ "${DATASET_MODE}" == "full" ]]; then
  echo "=== probe: dataset_mode=full, archive=${ARCHIVE_CSV} (${ARCHIVE_ROWS} rows)"
  echo "=== subset_size=${SUBSET_SIZE} is IGNORED in this mode; the fit is the archive"
else
  echo "=== probe: subset_size=${SUBSET_SIZE}, archive=${ARCHIVE_CSV} (${ARCHIVE_ROWS} rows)"
fi
echo "=== GPU before the run"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

# `oracle.type=...` would merge onto the config's Dock3Oracle mapping and leave
# its required docking paths in place, so the oracle is replaced wholesale, the
# same way jobs/reward_shape_nibi_nodock.sh does it.
# An OOM is a RESULT here, not an accident, so the failure must not take the
# summary below with it: `set -e` would exit before the one reading this job
# exists to produce. The exit status is kept and re-raised at the very end.
CONFIGS=("config/ampc/subset_fixed_gp.yaml")
SUBSET_OVERRIDES=("surrogate.training_subset.size=${SUBSET_SIZE}")
if [[ "${DATASET_MODE}" == "full" ]]; then
  # The FullArchive regime has no subset size -- the fit IS the archive -- so
  # the size override must not be passed, and the node has to be replaced
  # rather than merged into. See config/ampc/overrides/full_archive.yaml.
  CONFIGS+=("config/ampc/overrides/full_archive.yaml")
  SUBSET_OVERRIDES=(
    "surrogate.training_subset={type: FullArchive, top_threshold: 0.5, max_size: 40000}"
  )
fi

STATUS=0
uv run --no-sync python scripts/run_with_sigterm_flush.py \
  "${CONFIGS[@]}" \
  "run_name=${RUN_NAME}" \
  "dataset.initial_data.path=${ARCHIVE_CSV}" \
  "${SUBSET_OVERRIDES[@]}" \
  "oracle={type: ConstantOracle, value: 0.0, fidelity_costs: {1: 1.0}}" \
  "sampler.n_train_steps=50" \
  "sampler.n_samples=10000" \
  "budget.available_budget=20.0" \
  "budget.max_rounds=2" \
  "budget.schedule.value=10.0" \
  "${@:4}" || STATUS=$?

echo
echo "=== GPU after the run"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
echo
echo "=== CUDA memory per fit (round 2 is the one that matters)"
python3 - "${OUTPUT_DIR}/stage_history.jsonl" <<'PY'
import json, os, sys

path = sys.argv[1]
if not os.path.exists(path):
    # Nothing got as far as finishing a stage, so the failure is upstream of
    # the GP: read the .err rather than looking for memory numbers.
    print(f"  no stage records at {path}: the run failed before the first fit.")
    raise SystemExit(0)

for line in open(path):
    record = json.loads(line)
    if record.get("stage") != "surrogate_fit":
        continue
    metrics = record.get("metrics") or {}
    print(
        "  round %s: subset=%s  fit=%.0fs  live_before=%.2f GiB  "
        "cache_freed=%.2f GiB  peak=%.2f GiB  headroom=%.2f GiB of %.2f"
        % (
            record.get("round_index"),
            metrics.get("profiling/surrogate/subset_size"),
            metrics.get("profiling/surrogate/gp_fit_s", float("nan")),
            metrics.get("profiling/surrogate/cuda_reserved_before_fit_gib", float("nan")),
            metrics.get("profiling/surrogate/cuda_released_before_fit_gib", float("nan")),
            metrics.get("profiling/surrogate/cuda_peak_reserved_fit_gib", float("nan")),
            metrics.get("profiling/surrogate/cuda_headroom_at_peak_gib", float("nan")),
            metrics.get("profiling/surrogate/cuda_total_gib", float("nan")),
        )
    )
PY
echo
echo "PASS means: TWO surrogate_fit stages at subset_size=${SUBSET_SIZE}, and no"
echo "CUDA out-of-memory. One fit only means round 2 died: read the .err."
echo "IGNORE every y-derived metric: the oracle was a stub."
echo
echo "Loop exit status: ${STATUS}"
exit "${STATUS}"
