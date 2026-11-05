#!/bin/bash
# Multi-round single-fidelity active learning on Nibi: the whole loop, not one
# round of it. The GP refits each round on the archive of everything docked so
# far, an S3-GFN policy trains against its acquisition, the selector's top-k are
# docked, and the results go back into the archive.
#
#   sbatch jobs/single_fidelity_al_nibi.sh              (the 2-round test)
#   sbatch jobs/single_fidelity_al_nibi.sh <run_name> <rounds> <docks_per_round>
#   sbatch jobs/single_fidelity_al_nibi.sh <run_name> <rounds> <docks_per_round> <dataset_mode>
#
# DATASET_MODE picks how the labeled data is managed over rounds. The two
# regimes differ in one thing -- what the GP is fitted on -- and that one thing
# changes what the run can do.
#
#   subset  (default)  A fixed-size stratified subset of the archive, reselected
#                      every round: every row at y >= 0.5 kept outright, the rest
#                      filling target bands by farthest-point sampling. The fit
#                      stays 25,000 rows however large the archive gets, so cost
#                      and memory are flat and the round count is bounded by the
#                      docking budget. This produced
#                      sf-10round-5000dock-5000steps-seed42, whose archive grew
#                      25,000 -> 66,283 against a fit pinned at 25,000.
#
#   full               No subsetting: the GP fits the whole archive, which grows
#                      every round. Nothing mediates between what has been
#                      docked and what the surrogate learns from. Adds
#                      config/ampc/overrides/full_archive.yaml and the override
#                      that replaces the training_subset node; read that file
#                      for why the switch takes two pieces.
#
# THE CEILING ON THE TRAINING SET, which for `full` decides the round count
# rather than the budget. An exact GP's kernel matrix is O(n^2) and the fit
# peaks at 3.0068e-08 * n^2 + 0.325 GiB, which reproduces every isolated
# benchmark to within 0.15 GiB. That model is NOT the ceiling: it describes a
# fit with the card to itself, and the loop never has that. The ceiling was
# settled by probing the real loop instead, two rounds each, with the policy
# and encoder resident:
#
#   n=40,000  job 26162761  PASSED  round-2 peak 72.48 GiB, 6.70 GiB spare
#   n=45,000  job 26162757  OOM in round 2
#   n=45,000  job 26215104  OOM in round 2, after the cached-block release
#   n=50,000  job 26159147  OOM in round 2
#
# So 40,000 ROWS IS THE CAP, measured, and 45,000 is not reachable on this card.
# Round 2 at 45,000 died with 68.94 GiB allocated and 2.27 GiB free, needing one
# more contiguous 7.54 GiB matrix: the live policy, prior, encoder and
# acquisition state come to ~7.6 GiB, and the fit needs room for another full
# kernel matrix at its peak rather than just its own peak. An earlier reading of
# this as stale allocator cache was wrong -- job 26215104 released the cache and
# failed the same way.
#
# For `full`, the archive gains ~4,700 rows a round, not the full docking
# budget, because ~8% of docks fail and add no row. From a 25,000-row start:
#
#   fit 1  25,000    fit 3  34,400
#   fit 2  29,700    fit 4  39,100    fit 5  43,800  <- over the cap
#
# So `full` gets about FOUR rounds from 25,000, not ten, and one from the
# 39,999-row set. `max_size: 40000` below stops it with a named error rather
# than a CUDA out-of-memory inside the Cholesky; size the rounds for it.
#
# `subset` is unaffected: it fits a pinned 25,000 rows, well under the cap.
#
# EVERY setting lives in config/ampc/subset_fixed_gp.yaml. This script passes
# no `sampler.*` overrides: the reward-shape arms took their real reward, beta,
# loss and floor from overrides here, so their YAML never described the run.
# The only overrides below are the run's label and, for a non-default size, the
# round count and docking budget. The validated config -- every field including
# the defaults -- is logged to W&B under `effective.*`, so the run states what
# it ran whatever is passed.
#
# Resources. One H100 and 24 CPUs: the CPUs are the docking workers, and the
# 48-worker per-node cap is the ceiling. Over 14 CPUs per GPU likely bills as
# more than one GPU on this cluster; that is accepted for the throughput.
#
# Time. The 10-round run (`<run_name> 10 5000`): per round ~1 h of policy
# training (5,000 steps at the ~0.7 s/step the shard-free 2-round test
# measured), ~4 min generating the 100k pool, ~2.5 min selecting, ~1.5 min
# scoring the pool, a ~1 min GP fit, and 5,000 docking evaluations at the
# ~27 min per 1,000 that test measured on 24 workers, so ~2.3 h. Call it
# ~3.5 h per round, ~35 h for ten. 48 hours leaves room for the docking to
# slow as the archive and the policy change. There is no resume: a run that
# dies is rerun from round 0.
#
# Memory. The shard-free 2-round test peaked at ~44 GB; 80 GB leaves room for
# the archive growing to ~65k rows.
#
# WHAT THE RUN RECORDS ABOUT ITSELF. W&B gets the validated config under
# `effective.*` (every field, defaults included, so the surrogate, the training
# subset or FullArchive node, the sampler's loss/beta/reward/floor and the seed
# are all stated), plus `cli.*` and `provenance.*`: the reproduction command,
# the commit, whether the working tree was DIRTY and which files, the digest of
# the uncommitted diff, the resolved-config md5, and a path/size/md5 record for
# every input file -- the archive CSV, the MiniMol checkpoint, the feature store,
# the DOCK3 binary. The same lands on disk in run_manifest.json, and a dirty
# tree also leaves uncommitted.patch in the run directory.
#
# COMMIT BEFORE LAUNCHING. A dirty tree is recorded and warned about, loudly, in
# the first lines of the .out -- but a commit is the only thing that makes the
# run reproducible from the SHA alone.
#
# A crash does not lose finished stages: the surrogate fit, the policy training
# and the docking each write their metrics, figures and results to disk as they
# complete (stage_history.jsonl in the run directory). Re-check the fair share
# before every submit:
#   sshare -l -A def-bengioy_gpu,def-mailhoto_gpu,def-yvesbrun_gpu

#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --job-name=single_fidelity_al
#SBATCH --account=def-yvesbrun_gpu
#SBATCH --output=slurm_logs/single_fidelity_al_%j.out
#SBATCH --error=slurm_logs/single_fidelity_al_%j.err

set -euo pipefail

RUN_NAME="${1:-test-2round-1000dock-seed42}"
ROUNDS="${2:-}"
DOCKS_PER_ROUND="${3:-}"
DATASET_MODE="${4:-subset}"
if [[ ! "${RUN_NAME}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "run_name must be a path- and W&B-safe label; got '${RUN_NAME}'." >&2
  exit 2
fi
if [[ -n "${ROUNDS}${DOCKS_PER_ROUND}" ]]; then
  if [[ ! "${ROUNDS}" =~ ^[0-9]+$ || ! "${DOCKS_PER_ROUND}" =~ ^[0-9]+$ \
        || "${ROUNDS}" -lt 1 || "${DOCKS_PER_ROUND}" -lt 1 ]]; then
    echo "usage: sbatch jobs/single_fidelity_al_nibi.sh <run_name> [<rounds> <docks_per_round>] [<dataset_mode>]" >&2
    echo "Pass both of rounds and docks_per_round, or neither to use the config's." >&2
    exit 2
  fi
fi
# Rejected rather than defaulted: a typo here would silently run the other
# regime for ~30 hours, and the two are not comparable.
if [[ "${DATASET_MODE}" != "subset" && "${DATASET_MODE}" != "full" ]]; then
  echo "dataset_mode must be 'subset' or 'full'; got '${DATASET_MODE}'." >&2
  echo "  subset  fixed-size stratified subset of the archive (the config's own)" >&2
  echo "  full    fit the whole archive, which grows each round (~6 rounds max)" >&2
  exit 2
fi

cd /home/ethankrz/activelearning
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1   # compute nodes have no internet

CONFIG="config/ampc/subset_fixed_gp.yaml"
OUTPUT_DIR="outputs/ampc/single_fidelity_al/${RUN_NAME}"
# One directory per run: artifacts, the stage history and the offline W&B run
# all land here, so a rerun cannot write into another run's.
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "${OUTPUT_DIR} already exists; pick another run_name or move it aside." >&2
  exit 2
fi
mkdir -p "${OUTPUT_DIR}" slurm_logs

# The login node has the network; this node does not. Offline writes under
# WANDB_DIR, which is this run's own directory, and is synced afterwards.
export WANDB_MODE=offline
export WANDB_DIR="${OUTPUT_DIR}"

OVERRIDES=("run_name=${RUN_NAME}")
if [[ -n "${ROUNDS}" ]]; then
  OVERRIDES+=(
    "budget.max_rounds=${ROUNDS}"
    "budget.schedule.value=${DOCKS_PER_ROUND}"
    "budget.available_budget=$((ROUNDS * DOCKS_PER_ROUND))"
  )
fi

# The training_subset node must be REPLACED, not merged into, because the two
# regimes do not share a schema: FullArchive has no size/keep_above/band_edges
# and its config sets extra="forbid". OmegaConf merges mappings recursively, so
# the override alone would leave the StratifiedFPS keys beside the new type and
# the config would be refused. The overlay nulls the node first; the override
# then merges onto nothing and replaces it. Both halves are needed, and neither
# can misfire quietly -- see config/ampc/overrides/full_archive.yaml.
#
# top_threshold mirrors the subset arm's keep_above so train/top_count means the
# same thing in both; it selects nothing, since every row is fitted anyway.
# max_size is the measured ceiling itself -- 40,000 passed the two-round probe
# and 45,000 did not -- so there is no margin below it: see the header.
CONFIGS=("${CONFIG}")
DATASET_NOTE="reselect the GP's training set from the archive (stratified FPS, every row at y >= 0.5 kept outright, band edges pinned to the round-0 balanced quantiles)"
if [[ "${DATASET_MODE}" == "full" ]]; then
  CONFIGS+=("config/ampc/overrides/full_archive.yaml")
  OVERRIDES+=(
    "surrogate.training_subset={type: FullArchive, top_threshold: 0.5, max_size: 40000}"
  )
  DATASET_NOTE="refit the GP on the WHOLE archive, which grows every round (no subsetting; dataset_mode=full, capped at 40,000 rows by VRAM -- the measured ceiling, see the header -- so expect ~4 rounds from a 25,000-row start)"
fi

export WANDB_NOTES="Single-fidelity active learning, run ${RUN_NAME}: the full loop over rounds, which the reward-shape arms only ever ran one of.
Each round: ${DATASET_NOTE}, refit the isotropic Matern GP with its mean pinned at the 10M library mean 0.03956, train the S3-GFN policy for the configured steps against GIBBON on that GP (power reward, beta 0.1, VarGrad, fixed floor -150 nats), generate 100k molecules, dock the top-k by acquisition, and add them to the archive.
THE QUESTION: does the distribution of docked y rise round over round? Read dataset/general/docked/frac_above_0.5 and its siblings. The archive STARTS as the 25,000-molecule balanced set, which holds 9 rows at y >= 0.5, so the tail has to be grown by the loop rather than handed to it.
Watch alongside it: surrogate/eval_set/ampc_331k/* is the GP scored on one fixed set of 331,480 docked molecules every round, the only constant scale on which the surrogate's own health can be compared across rounds; surrogate/fixed_feature_gp/hyperparameters/outputscale over .../noise is the signal-to-noise ratio, which already fell from ~4.3 to ~3.0 between round 0 and the round-1 simulation and is the earliest sign of the reward flattening; surrogate/subset_gp/train/frac_above_* is how enriched the set the GP was fitted on has become; sampler/s3gfn/reward/above_floor_fraction_median is the share of each batch not tied at the floor; and sampler/s3gfn/train/unique_rate is the mode-collapse check that moves while validity stays high.
Caveat for a short test run: 1,000 docks per round gives ~70 molecules above the threshold at round-1 hit rates, enough to prove the loop runs end to end and far too few to measure enrichment from.
Round 0's selection is the identity, because the archive equals the training budget, so that round has no band metrics. Expected, not a fault."
if [[ "${DATASET_MODE}" == "full" ]]; then
  WANDB_NOTES="${WANDB_NOTES}
DATASET MODE IS 'full': there is no subset selection, so NO round has band metrics and surrogate/subset_gp/train/archive_fraction reads 1.000 throughout -- that is the regime working, not a fault. The reading to watch instead is profiling/surrogate/cuda_headroom_at_peak_gib, which falls every round as the kernel matrix grows; the run stops with a named ValueError at 40,000 rows rather than a CUDA OOM."
fi
printf '%s\n' "${WANDB_NOTES}" > "${OUTPUT_DIR}/run_description.txt"
echo "dataset_mode=${DATASET_MODE}"

# run_with_sigterm_flush turns Slurm's SIGTERM into a KeyboardInterrupt, so the
# loop's handler commits the dying round's telemetry instead of losing it.
# --no-sync keeps uv from re-resolving dependencies over a network this node
# does not have.
uv run --no-sync python scripts/run_with_sigterm_flush.py \
  "${CONFIGS[@]}" \
  "${OVERRIDES[@]}"

echo
echo "Done. Outputs in ${OUTPUT_DIR}."
echo "Sync from a login node with:"
echo "  wandb sync ${OUTPUT_DIR}/wandb/offline-run-*"
