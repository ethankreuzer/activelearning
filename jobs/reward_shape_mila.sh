#!/bin/bash
# Mila port of jobs/reward_shape_nibi.sh, with the docking removed: one
# reward-shape arm that trains the S3-GFN policy against a fixed GIBBON
# acquisition on the frozen svgp300k surrogate, generates its pool, and stops.
#
#   LOG_SCORE_FLOOR=-500 sbatch jobs/reward_shape_mila.sh power 0.015 1000 2500
#
# WHY THIS SCRIPT EXISTS. The reward-shape question -- which (beta, floor, step
# count) leaves the policy finding high-acquisition molecules without losing
# validity or collapsing onto a few modes -- is answered entirely by the
# sampler's own per-step diagnostics. On Nibi those arms cost ~6.6 h each, of
# which ~5.3 h is the 10,000 Dock3 evaluations, and the 2026-10-27 arms showed
# the docked-y distributions of two very different policies are nearly
# superimposed (mean 0.525 vs 0.542 at validity 0.83 vs 0.55): docking does not
# separate the arms it is paid for. Mila has no AmpC receptor files and no
# dock64, so Dock3Oracle raises FileNotFoundError while being built and the
# round never starts -- but it has far more GPU capacity, so the scan runs here
# with the oracle stubbed out and only the winner is docked back on Nibi.
#
# WHAT IS NOT MEASURED HERE. The oracle is ConstantOracle: every molecule gets
# the same made-up label. Everything downstream of it is meaningless and must
# not be read or compared against the Nibi arms -- dataset/general/docked/*,
# surrogate/general/held_out/*, and anything in the round record that depends
# on y. The budget is 1, so exactly one molecule is "labelled" and the round
# closes cleanly. The sampler diagnostics are collected before the oracle is
# reached (active_learning.py, stage "sampler"), which is what makes this work.
#
# WHAT TO READ, and against which failure mode. The 2026-10-27/28 svgp300k arms
# (REWARD_SHAPE_FLOOR_ABLATION.md and the revision below it) separated two
# knobs acting on two different axes:
#   floor depth -> VALIDITY. A deep floor pays out for being slightly unusual
#     rather than genuinely good: at floor -1000 a molecule at log acq -900
#     earns 100 nats over a floored one while still being worthless, and the
#     cheapest route to "unusual" in SMILES space is syntactic. Both floor -500
#     arms held validity >= 0.83 across a 2x range of above-floor fraction and a
#     3x range of prior_divergence; the one floor -1000 arm fell to 0.545.
#   beta -> DIVERSITY. beta=0.1 at floor -500 kept validity 0.882 and the
#     largest prior_divergence of any arm (16.5), and failed the other way:
#     0.901 duplicate rejection, 7,273 unique molecules from 2,000,000 attempts.
# So read train/batch_health for BOTH its validity and its unique curve, not the
# run-level averages -- in the beta=0.03 arm validity was 0.92 at the step the
# acquisition plateaued and 0.70 by the end, and the average of 0.83 describes
# neither. Three more: the step at which reward/above_floor_fraction first
# leaves zero (how long the policy searched before the reward said anything --
# ~1,150 steps at floor -500, ~700 at floor -1000, and the shallow direction is
# bounded by this warmup terminating at all), the step at which acq/trajectory
# plateaus (~2,300 in the beta=0.03 arm, after which every further step cost
# validity and bought nothing), and whether online RTB falls after its peak. A
# falling RTB is not automatically good news: in the floor -1000 arm it peaked
# at step 1,200 and decayed because the policy had converged onto an
# off-manifold mode, which is the same window in which validity fell off a cliff.
#
# Resources. No docking, so this is the sampler plus one selection pass. Measured
# on the Nibi H100 arms of 2026-10-27 (sampler/s3gfn/{train,generation}/duration_s
# in stage_history.jsonl): 3,974 s and 3,673 s for 5,000 training steps, i.e.
# 0.73-0.80 s/step; generation of the 100k pool 493 s and 1,505 s, a 3x spread
# that tracks how much validity the arm lost; selection ~3.5 min.
#
# A100 and not L40S or H100. Memory is not the constraint -- the svgp300k GIBBON
# pass peaked at 10.4 GB on the GPU and ~3.5 GB host RSS in the scaling study, so
# a 24 GB card fits and the 80 GB a100l only buys queue time. The constraint is
# bandwidth: this is autoregressive SMILES decoding plus a 1024-inducing-point GP
# solve per batch, small kernels that never fill the tensor cores, so H100's
# advantage over A100 is ~1.3-1.6x here rather than the headline figure and is
# not worth its contention. For the same reason skip V100 (no bf16, a third of
# A100's bandwidth) and A6000 (Ampere, but 768 GB/s against A100's 2,039).
# Expect ~52-64 min of training for 3,000 steps and ~85-105 min for 5,000, so 4 h
# covers the longest arm with its slowest generation pass. Mila names the 40 GB
# part a100 and the 80 GB part a100l; confirm with `sinfo -o "%P %l %G"`.
#
# main and not unkillable, which caps a user at one job and would run the arms
# serially. The tradeoff is preemption: a requeued arm lands on an output
# directory that already holds its generated pool, which the guard below refuses
# to reuse, so resubmit it with ARM_SUFFIX=_rerun rather than deleting the pool.
# short-unkillable (3 h, up to 4 GPUs, non-preemptible) fits a 3,000-step arm and
# avoids that entirely. If an arm OOMs, raise --mem rather than trimming the
# pool: its size is what the Nibi arms used and changing it breaks the comparison.
#
# 4 CPUs and 24G, half of main's per-user quota (8 CPUs / 48G / 2 GPUs), so two
# arms run concurrently. Four is not a compromise: nothing in this job path uses
# CPU parallelism, because resolve_num_workers is read only by Dock3Oracle and
# CxcalcOracle and the oracle here is the ConstantOracle stub, so the CPUs serve
# host-side torch ops and the selection pass's dataframes and the hour is spent
# on the GPU regardless. 24G against a measured ~3.5 GB host RSS leaves 7x; the
# largest host allocation is the 300k training set and its feature cache, under a
# gigabyte. Submit all the arms at once: a third one exceeding the quota stays
# pending on an association limit rather than being rejected.
#SBATCH --partition=main
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH --job-name=ampc_reward_shape_mila
#SBATCH --output=slurm_logs/ampc_reward_shape_mila_%j.out
#SBATCH --error=slurm_logs/ampc_reward_shape_mila_%j.err

set -euo pipefail

TRANSFORM="${1:-}"
BETA="${2:-}"
INITIAL_LOG_Z="${3:-}"
N_TRAIN_STEPS="${4:-}"
SEED="${5:-}"
if [[ ! "${TRANSFORM}" =~ ^(exponential|power)$ \
      || ! "${BETA}" =~ ^[0-9]*\.?[0-9]+$ \
      || ! "${INITIAL_LOG_Z}" =~ ^-?[0-9]*\.?[0-9]+$ \
      || ! "${N_TRAIN_STEPS}" =~ ^[0-9]+$ \
      || "${N_TRAIN_STEPS}" -lt 100 \
      || ! "${SEED}" =~ ^[0-9]*$ ]]; then
  echo "usage: sbatch jobs/reward_shape_mila.sh <exponential|power> <beta> <initial_log_z> <n_train_steps> [seed]" >&2
  echo "n_train_steps is an integer of at least 100; the seed is optional." >&2
  exit 2
fi

# Fixed for the whole study, as on Nibi. INITIAL_LOG_Z and LOG_Z_LR are unused
# under vargrad and kept only so switching back to rtb needs no edit here.
LOG_SCORE_FLOOR="${LOG_SCORE_FLOOR:--500}"
if [[ ! "${LOG_SCORE_FLOOR}" =~ ^-[0-9]+(\.[0-9]+)?$ ]]; then
  echo "LOG_SCORE_FLOOR must be a negative number, e.g. -500." >&2
  exit 2
fi
LOSS="vargrad"
LOG_Z_LR="0.01"
# The default is -500 here and -150 on Nibi, because svgp300k's log GIBBON sits
# one to two orders of magnitude below the exact GPs'. That default is NOT a
# recommendation. The first reading of it was that a shallow floor would tie
# every prior-like molecule on this surrogate; counting the trained pools of
# 2026-10-27 says otherwise. Fraction of the 100k pool clearing each floor, from
# the beta=0.03/-500 arm's scored_samples.csv: nonzero acq 0.2298, >= -500
# 0.2280, >= -250 0.2137, >= -150 0.1776, >= -100 0.1373. About 77% of the pool
# has acq exactly 0.0 -- GIBBON underflowed, so it is at -inf and tied at any
# floor -- and the band between -500 and -150 is nearly empty. Raising the floor
# therefore barely changes WHICH molecules are graded and mostly shrinks the gap
# between the tied mass and the tail, which is the mechanism by which it protects
# validity (see REWARD_SHAPE_FLOOR_ABLATION.md). Floors of -250 and -150 are the
# ones worth scanning; the cost of going shallow is a longer warmup before
# above_floor_fraction leaves zero, not a degenerate reward.

# Only svgp300k. The exact-GP and fixed-feature-GP states this study's other
# surrogates need were fitted on Nibi and do not exist here; svgp300k was
# fitted on Mila in the first place.
BASE_CONFIG="config/ampc/reward_shape_svgp300k.yaml"
FITTED_DIR="outputs/ampc/variational_gp/svgpscale_300000_svgp_M1024_iso_matern_scaled_pm0.03956_elbo_cosine_lr3e-3_steps10064_fp32"
SURROGATE_TAGS="svgp,M1024,elbo,svgpscale-300k,lr3e-3,steps10064,pinned-mean,fp32"
WANDB_PROJECT="ampc-reward-shape-svgp"

ARM="reward_${TRANSFORM}_beta${BETA}_vargrad_floor${LOG_SCORE_FLOOR}_steps${N_TRAIN_STEPS}"
SEED_OVERRIDE=()
SEED_TAG=""
if [[ -n "${SEED}" ]]; then
  ARM="${ARM}_s${SEED}"
  SEED_OVERRIDE=("runtime.seed=${SEED}")
  SEED_TAG=",seed${SEED}"
fi
ARM="${ARM}_svgp300k_m1024_fp32"
# Optional suffix for rerunning an arm whose settings are unchanged, so the
# rerun gets its own directory instead of colliding with the earlier pool.
ARM_SUFFIX="${ARM_SUFFIX:-}"
if [[ -n "${ARM_SUFFIX}" && ! "${ARM_SUFFIX}" =~ ^_[A-Za-z0-9.-]+$ ]]; then
  echo "ARM_SUFFIX must look like _name (letters, digits, '.', '-')." >&2
  exit 2
fi
ARM="${ARM}${ARM_SUFFIX}"

cd "${HOME}/activelearning"
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1   # nothing is downloaded: the prior and encoder are cached
# Mila compute nodes reach the internet, so wandb runs online and there is no
# `wandb sync` step afterwards, unlike the Nibi arms.

STATE_PATH="${FITTED_DIR}/surrogate_state.pt"
OUTPUT_DIR="outputs/ampc/reward_shape_mila/${ARM}"
REJECT_LOG_PATH="${OUTPUT_DIR}/rejects_round_{round}.csv"
POOL_LOG_PATH="${OUTPUT_DIR}/generated_pool_round_{round}.csv"

for required in \
  "${STATE_PATH}" \
  "data/ampc_svgpscale_300000.csv" \
  "cache/ampc/exact_dkl_top_n/svgpscale_300000.npy" \
  "minimol_ampc_encoder/model/final.pt"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing ${required}." >&2
    echo "This arm restores a surrogate rather than fitting one, so the fitted" >&2
    echo "state, its training set, its feature cache and the MiniMol encoder" >&2
    echo "must all be present. Fit or copy them first:" >&2
    echo "  LR=3e-3 STEPS=10000 sbatch jobs/variational_gp_scaling_mila.sh elbo 300000" >&2
    exit 2
  fi
done
# Never overwrite an earlier run's generated pool: it is the arm's result.
if compgen -G "${OUTPUT_DIR}/generated_pool_round_*.csv" > /dev/null; then
  echo "${OUTPUT_DIR} already holds a generated pool from an earlier run." >&2
  echo "Rerun with a suffix, e.g. ARM_SUFFIX=_rerun2 sbatch jobs/reward_shape_mila.sh $*" >&2
  exit 2
fi

# Per-arm, so concurrent arms never race on a shared ./wandb/latest-run symlink.
export WANDB_DIR="${OUTPUT_DIR}"
mkdir -p "${WANDB_DIR}" slurm_logs

export WANDB_NOTES="REWARD-SHAPE TUNING arm ${ARM}, on Mila, WITHOUT DOCKING. One active-learning round training the S3-GFN policy against a fixed GIBBON acquisition on a frozen surrogate, then one stubbed oracle query.
THE QUESTION THIS RUN ASKS: at this (beta, floor, step count), does the policy keep finding higher-acquisition molecules without losing validity (the floor's failure mode) or collapsing onto a few modes (beta's failure mode)? The answer is entirely in the per-step sampler figures; nothing downstream of the oracle is meaningful.
ORACLE IS A STUB. oracle=ConstantOracle returns the same label for every molecule, and the budget is 1, so one molecule is 'labelled' at a made-up value and the round closes. Mila has no AmpC receptor files and no dock64, so the real Dock3Oracle cannot even be constructed here. IGNORE dataset/general/docked/*, surrogate/general/held_out/* and every other y-derived number in this run: they describe the stub, not chemistry. The docked comparison is done on Nibi, for the winning arm only.
Surrogate: RESTORED, NOT REFITTED. The sparse variational GP is rebuilt on the 300,000-molecule nested scaling set data/ampc_svgpscale_300000.csv and its fitted parameters -- 1024 inducing points and their variational distribution, kernel, noise, input scaling and output standardization -- are loaded from ${STATE_PATH} in place of training. See ${FITTED_DIR}/arm_description.txt for how that fit was obtained. It is far more confident than the exact GPs, so log GIBBON runs much lower (gp_molformer_set median -2,688, 99th percentile -1,146): the floor of ${LOG_SCORE_FLOOR} is set for that range rather than the exact GPs'. That range is wider than it is populated: about 77% of a trained pool has acq exactly 0.0, hence -inf and tied at any floor, so the fraction graded only falls from 0.228 to 0.178 between floor -500 and -150. A shallow floor on this surrogate does not tie everything; it shrinks the gap between the tied mass and the tail, which is why it protects validity.
Acquisition: GIBBON (QLowerBoundMaxValueEntropy), log_space=true so the information gain never underflows, log_output=false so it is returned on the VALUE scale, which is what the reward transform needs. Max-value support is capped at 100,000 of the 300,000 training molecules by the deterministic fidelity/target-stratified reservoir, as in the scaling study.
Reward: transform=${TRANSFORM}, beta=${BETA}, log(acq) floored at the FIXED value ${LOG_SCORE_FLOOR} nats, the same for every batch, so a molecule's reward depends on that molecule alone and everything below the floor is tied. Loss: VARGRAD, so there is no log_z parameter and an offset shared by the whole batch contributes no gradient; INITIAL_LOG_Z (${INITIAL_LOG_Z}) and LOG_Z_LR (${LOG_Z_LR}) are passed but UNUSED.
WHAT THE THREE DOCKED ARMS OF 2026-10-27/28 ESTABLISHED, which this scan follows up: floor depth governs VALIDITY and beta governs DIVERSITY, and they fail in different directions. Both floor -500 arms held validity >= 0.83 while spanning a 2x range of above-floor fraction and a 3x range of prior_divergence; the single floor -1000 arm fell to 0.545, because a deep floor pays out for being slightly unusual rather than genuinely good and the cheapest route to unusual in SMILES space is syntactic. Separately, beta=0.1 at floor -500 kept validity 0.882 and the largest prior_divergence of any arm (16.5) but rejected 90.1% of its generations as duplicates, yielding 7,273 unique molecules from 2,000,000 attempts.
WHAT TO READ: train/batch_health for BOTH validity and uniqueness, read AT the step acq/trajectory plateaus rather than as a run-level average (in the beta=0.03 arm validity was 0.92 at the plateau and 0.70 at the end, and its average of 0.83 describes neither); the step at which reward/above_floor_fraction first leaves zero, which is how long the policy searched before the reward said anything (~1,150 steps at floor -500, ~700 at floor -1000) and is what bounds how shallow the floor can go; the step at which acq/trajectory plateaus, after which further steps cost validity and buy nothing; reward/effective_support and reward/concentration; train/prior_divergence, which on this evidence does NOT predict validity loss; and the training-loss panel. A falling online RTB is not automatically good: in the floor -1000 arm it peaked at step 1,200 and fell because the policy had converged onto an off-manifold mode, over exactly the window in which validity collapsed.
TRAINING LENGTH: this arm runs ${N_TRAIN_STEPS} steps. A shorter arm is not a longer one stopped early -- the warmup/decay schedule spans n_train_steps, so 2,500 steps decays the learning rate twice as fast as 5,000.
Arm name: ${ARM}. Submitted with: sbatch jobs/reward_shape_mila.sh $*   (Slurm job ${SLURM_JOB_ID:-unknown})"
printf '%s\n' "${WANDB_NOTES}" > "${OUTPUT_DIR}/arm_description.txt"

RUN_NAME="${ARM//_/-}-${SLURM_JOB_ID:-local}"
TAGS="reward-shape,no-docking,${TRANSFORM},beta${BETA},vargrad,floor${LOG_SCORE_FLOOR},steps${N_TRAIN_STEPS},${SURROGATE_TAGS},restored-state,1-round${SEED_TAG}${ARM_SUFFIX:+,suffix${ARM_SUFFIX}}"

# Restated as one YAML value each rather than overridden field by field:
# OmegaConf.from_dotlist reads every path segment as a dict key, so
# `oracle.type=...` would merge onto the config's Dock3Oracle mapping and leave
# its receptor paths behind, and `logger.loggers.0.run_name=...` would build a
# dict under `loggers` and die with "Cannot merge DictConfig with ListConfig".
ORACLE="oracle={type: ConstantOracle, value: 0.0, fidelity_costs: {1: 1.0}}"
# One stubbed query, so the round closes without pretending to label a pool.
# The selector still ranks the whole 100k pool by acquisition, which is the
# path worth exercising; it just keeps one molecule.
BUDGET="budget={available_budget: 1.0, schedule: {type: constant, value: 1.0}, max_rounds: 1}"
LOGGER="logger={type: MultiLogger, loggers: [{type: ConsoleLogger, project_name: ${WANDB_PROJECT}, run_name: ${RUN_NAME}}, {type: WandbLogger, project_name: ${WANDB_PROJECT}, run_name: ${RUN_NAME}, entity: models-mila5723, group: reward-shape-tuning, tags: [mila,${TAGS}]}]}"

# "${@:6}" and not "${@:5}": argument 5 is the seed, which is consumed above.
# jobs/reward_shape_nibi.sh passes "${@:5}", so an extra override there arrives
# with the seed in front of it and Hydra rejects the pair.
uv run --no-sync python scripts/run_with_sigterm_flush.py \
  "${BASE_CONFIG}" \
  "sampler.reward_transform=${TRANSFORM}" \
  "sampler.beta=${BETA}" \
  "sampler.n_train_steps=${N_TRAIN_STEPS}" \
  "sampler.initial_log_z=${INITIAL_LOG_Z}" \
  "sampler.loss=${LOSS}" \
  "sampler.log_score_floor=${LOG_SCORE_FLOOR}" \
  "sampler.log_z_learning_rate=${LOG_Z_LR}" \
  "${SEED_OVERRIDE[@]}" \
  "run_writer.output_dir=${OUTPUT_DIR}" \
  "sampler.reject_log_path=${REJECT_LOG_PATH}" \
  "sampler.pool_log_path=${POOL_LOG_PATH}" \
  "${ORACLE}" \
  "${BUDGET}" \
  "${LOGGER}" \
  "${@:6}"
