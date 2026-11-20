#!/bin/bash
# Beta scan on the ExactGP 20k prior-on surrogate, on Mila, without docking.
# One active-learning round training the S3-GFN policy against a fixed GIBBON
# acquisition on the frozen surrogate, generating its pool, and stopping.
#
#   LOG_SCORE_FLOOR=-150 sbatch jobs/reward_shape_exact20k_mila.sh power 0.1 0 5000 42
#
# THE SCAN IS A 4 x 4 GRID, 16 arms: beta in 0.05 / 0.1 / 0.2 / 0.4 crossed
# with LOG_SCORE_FLOOR in -750 / -175 / -150 / -125. Submit it with
#
#   for F in -750 -175 -150 -125; do
#     for B in 0.05 0.1 0.2 0.4; do
#       LOG_SCORE_FLOOR=$F sbatch jobs/reward_shape_exact20k_mila.sh power $B 0 5000 42
#     done
#   done
#
# Mila's per-user quota on main is 2 GPUs, so the 16 land two at a time and the
# rest stay pending on an association limit rather than being rejected.
#
# WHY A GRID AND NOT A BETA SCAN THEN A FLOOR SCAN. They are not independent.
# With log R = beta * max(s, F): F alone sets what fraction of the pool is
# tied, beta alone sets the reward ratio between two above-floor molecules, but
# the ratio between the best molecule and the tied mass -- the reward's total
# dynamic range, which is what drives how hard the policy is pushed off the
# prior -- is the PRODUCT beta * (s_max - F). Tuning one then the other walks
# an L through a space whose contours are diagonal.
#
# This is jobs/reward_shape_mila.sh with the surrogate swapped; read that
# script's header for what the sampler diagnostics mean, why the oracle is
# stubbed, and which W&B panels to look at. What differs here:
#
#   SURROGATE. The ExactGP fitted to the 20,000-molecule svgpscale set with the
#   lengthscale prior on and the mean pinned -- arm 1 of the four carried into
#   the active-learning run (thesis/surrogate_models/active_learning_model_selection.md).
#   The beta scan is run on this one model in the hope that the result
#   transfers to the variational arms. That hope is best founded for ELBO,
#   which shares this model's noise-heavy uncertainty allocation, and least
#   founded for PLL, which allocates to the latent GP instead.
#
#   FLOOR, and where the four values come from. They are sited on the log
#   GIBBON distribution this surrogate assigns to gp_molformer_set, which is
#   not a proxy for the step-0 pool -- it IS the step-0 pool, 83,811 molecules
#   sampled from the GP-MoLFormer prior the policy is initialised to. That
#   distribution is tight: max -27.6, p99 -104.9, median -147.8, p1 -184.8,
#   min -219.9. Above-floor fraction at step 0:
#
#     -750   1.0000   non-binding. -250, -400 and -750 are the SAME experiment,
#                     since nothing scores below -220; -750 is used because it
#                     also sits below float64's exp() underflow
#                     (log(5e-324) = -744.4), so only a genuinely-underflowed
#                     0.0 could ever land on it.
#     -175   ~0.93
#     -150   0.5532
#     -125   ~0.03    boundary case, expected to fail: with almost the whole
#                     batch tied the reward expresses nearly no preference.
#                     Included to find where training stops starting at all.
#
#   Floors shallower than about -100 are dead (above-floor fraction 0.006 at
#   -100, 0.0001 at -50) and deeper than about -250 are all the same arm, so
#   -100 to -200 is the entire useful range on this surrogate.
#
#   Note that `log_score_floor: null` would NOT mean "no floor" -- the power
#   transform falls back to a batch-relative floor 20 nats below each batch's
#   own maximum, which makes a molecule's reward depend on its batch-mates.
#   -750 is how "no floor" is expressed.
#
#   BETA, on the same evidence. The reward ratio across the pool with a
#   non-binding floor is exp(beta * spread), and the bulk spread is p1 to p99
#   = ~80 nats, NOT the 192 of min to max: the maximum is one outlier molecule
#   77 nats clear of p99. So 0.05 spans ~55x, 0.1 ~3,000x, 0.2 ~9e6, 0.4
#   ~8e13, and below ~0.02 the reward is nearly flat.
#
#   THE svgp300k ARMS' EMPIRICAL FINDINGS DO NOT TRANSFER. That surrogate is a
#   variational GP on 300,000 molecules; this is an exact GP on 20,000. Its log
#   GIBBON sits one to two orders of magnitude lower (median -2,688 against
#   -148) and ~77% of a trained pool underflowed to exactly 0.0, against
#   essentially none here. The -500 default there, the ~1,150-step warmup
#   before above_floor_fraction left zero, and the floor->validity /
#   beta->diversity mapping are properties of THAT reward landscape. Only the
#   algebra above carries over.
#
#   RESOURCES. Same shape as the svgp300k arms -- autoregressive SMILES
#   decoding plus a GP solve per batch -- but the GP solve differs: an exact GP
#   concatenates its 20,000 training rows onto every candidate chunk, where the
#   variational model only touches 1,024 inducing points. score_chunk_size is
#   64 for that reason (the value the 25,000-molecule exact-GP arms used), and
#   the per-step cost is expected to be at least the svgp300k arms' 0.73-0.80
#   s/step -- the GP solve is heavier here, so treat that as a floor rather
#   than an estimate.
#
#   --time=8:00:00 for the 5,000-step grid. The svgp300k arms measured 85-105
#   min of training for 5,000 steps plus 8-25 min of generation, and 4 h was
#   enough for them; this surrogate's per-step cost is unmeasured and strictly
#   higher, so the limit is doubled rather than estimated. main allows up to
#   5 days, so there is no partition reason to keep it tight, and an arm that
#   hits the wall loses its whole pool. Raise it again before raising the step
#   count.
#SBATCH --partition=main
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=8:00:00
#SBATCH --job-name=ampc_reward_shape_exact20k
#SBATCH --output=slurm_logs/ampc_reward_shape_exact20k_%j.out
#SBATCH --error=slurm_logs/ampc_reward_shape_exact20k_%j.err

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
  echo "usage: sbatch jobs/reward_shape_exact20k_mila.sh <exponential|power> <beta> <initial_log_z> <n_train_steps> [seed]" >&2
  echo "n_train_steps is an integer of at least 100; the seed is optional." >&2
  exit 2
fi

# The grid's floor axis; -750 (non-binding) is the default so an arm submitted
# without LOG_SCORE_FLOOR is the baseline column. INITIAL_LOG_Z and LOG_Z_LR
# are unused under vargrad and kept only so switching back to rtb needs no
# edit here.
LOG_SCORE_FLOOR="${LOG_SCORE_FLOOR:--750}"
if [[ ! "${LOG_SCORE_FLOOR}" =~ ^-[0-9]+(\.[0-9]+)?$ ]]; then
  echo "LOG_SCORE_FLOOR must be a negative number, e.g. -750." >&2
  exit 2
fi
LOSS="vargrad"
LOG_Z_LR="0.01"

BASE_CONFIG="config/ampc/reward_shape_exact20k.yaml"
FITTED_DIR="outputs/ampc/exact_gp_sweep/n20000_on_pinned"
SURROGATE_TAGS="exactgp,fixedfeature,svgpscale-20k,prior-on,pinned-mean,iso-matern,fp32"
WANDB_PROJECT="thesis-ampc-reward-shape-exact20k"

ARM="reward_${TRANSFORM}_beta${BETA}_vargrad_floor${LOG_SCORE_FLOOR}_steps${N_TRAIN_STEPS}"
SEED_OVERRIDE=()
SEED_TAG=""
if [[ -n "${SEED}" ]]; then
  ARM="${ARM}_s${SEED}"
  SEED_OVERRIDE=("runtime.seed=${SEED}")
  SEED_TAG=",seed${SEED}"
fi
ARM="${ARM}_exact20k_prioron_fp32"
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

STATE_PATH="${FITTED_DIR}/surrogate_state.pt"
OUTPUT_DIR="outputs/ampc/reward_shape_exact20k/${ARM}"
REJECT_LOG_PATH="${OUTPUT_DIR}/rejects_round_{round}.csv"
POOL_LOG_PATH="${OUTPUT_DIR}/generated_pool_round_{round}.csv"
# THESIS-GRADE: the per-step series and the training molecules are written to
# disk, not only to W&B. The W&B panels are PNGs and the policy is never
# checkpointed, so without these two files the within-round dynamics could be
# looked at once and never recomputed, re-binned, overlaid across arms or
# redrawn locally. Both are cheap: the step log is ~5,000 rows x 26 columns
# (a few hundred kB, uncompressed and greppable), the sample log is 320,000
# rows gzipped to ~6 MB.
#
#   training_steps_round_*.csv  one row per training step. valid/generated is
#     the VALIDITY curve, synthesizable/valid the SYNTHESIZABILITY curve, and
#     acq_max/acq_median/acq_min with reward_* and above_floor_fraction the
#     LOG GIBBON REWARD curves. Conditionally recorded series carry their own
#     step numbers, so a step that produced no valid molecules leaves a gap
#     rather than shifting every later point left -- which is worst exactly
#     when validity is collapsing, i.e. the failure this grid is looking for.
#   training_samples_round_*.csv.gz  every molecule generated WHILE training,
#     with its step, its post-transform reward_input_score and its raw
#     acquisition_score. generated_pool_round_*.csv is a single snapshot taken
#     AFTER training from the final policy; this is the trajectory.
TRAINING_STEP_LOG_PATH="${OUTPUT_DIR}/training_steps_round_{round}.csv"
TRAINING_SAMPLE_LOG_PATH="${OUTPUT_DIR}/training_samples_round_{round}.csv.gz"

for required in \
  "${STATE_PATH}" \
  "data/ampc_svgpscale_20000.csv" \
  "cache/ampc/exact_dkl_top_n/svgpscale_20000.npy" \
  "minimol_ampc_encoder/model/final.pt"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing ${required}." >&2
    echo "This arm restores a surrogate rather than fitting one, so the fitted" >&2
    echo "state, its training set, its feature cache and the MiniMol encoder" >&2
    echo "must all be present. Fit it first:" >&2
    echo "  sbatch jobs/exact_gp_sweep_mila.sh on pinned 20000" >&2
    exit 2
  fi
done
# Never overwrite an earlier run's generated pool: it is the arm's result.
if compgen -G "${OUTPUT_DIR}/generated_pool_round_*.csv" > /dev/null; then
  echo "${OUTPUT_DIR} already holds a generated pool from an earlier run." >&2
  echo "Rerun with a suffix, e.g. ARM_SUFFIX=_rerun2 sbatch jobs/reward_shape_exact20k_mila.sh $*" >&2
  exit 2
fi

# Per-arm, so concurrent arms never race on a shared ./wandb/latest-run symlink.
export WANDB_DIR="${OUTPUT_DIR}"
mkdir -p "${WANDB_DIR}" slurm_logs

export WANDB_NOTES="BETA x FLOOR GRID arm ${ARM}, on Mila, WITHOUT DOCKING. One active-learning round training the S3-GFN policy against a fixed GIBBON acquisition on a frozen surrogate, then one stubbed oracle query.
THE QUESTION THIS GRID ASKS: how do beta and the log-score floor TOGETHER shape training on this surrogate? The scan is 4 x 4 -- beta in 0.05/0.1/0.2/0.4 crossed with floor in -750/-175/-150/-125 -- and is a grid rather than two successive 1-D scans because the two knobs are not independent. With log R = beta * max(s, F): F alone sets what fraction of the pool is tied, beta alone sets the reward ratio between two above-floor molecules, but the ratio between the best molecule and the tied mass (the reward's total dynamic range, which drives how hard the policy is pushed off the prior) is the PRODUCT beta * (s_max - F). Read the response surface, not two curves.
ORACLE IS A STUB. oracle=ConstantOracle returns the same label for every molecule and the budget is 1, so one molecule is 'labelled' at a made-up value and the round closes. Mila has no AmpC receptor files and no dock64. IGNORE dataset/general/docked/*, surrogate/general/held_out/* and every other y-derived number in this run.
SURROGATE: RESTORED, NOT REFITTED. BoTorch exact GP on the frozen MiniMol AmpC fingerprints, isotropic Matern, LENGTHSCALE PRIOR ON, mean pinned at 0.0042864, fitted to the 20,000-molecule svgpscale set; parameters loaded from ${STATE_PATH}. See ${FITTED_DIR} and thesis/surrogate_models/active_learning_model_selection.md, where this is arm 1 of the four chosen for the active-learning run (the prior raises GIBBON top-100 enrichment on gp_molformer_set from 1.57x to 3.98x while leaving Validation and InVitro unchanged).
TRANSFER ASSUMPTION. This scan is run on ONE surrogate in the hope the chosen beta transfers to the variational arms. That is best founded for ELBO, which shares this model's noise-heavy uncertainty allocation (SSE negative overall, low latent-only coverage), and least founded for PLL, which allocates to the latent GP and has SSE positive overall. Treat a beta chosen here as a starting point for PLL, not a settled value.
Acquisition: GIBBON (QLowerBoundMaxValueEntropy), log_space=true so the information gain never underflows, log_output=false so it is returned on the VALUE scale, which is what the power transform logs. The max-value support is all 20,000 training molecules. score_chunk_size 64, because an exact GP concatenates its training rows onto every candidate chunk.
Reward: transform=${TRANSFORM}, beta=${BETA}, log(acq) floored at the FIXED value ${LOG_SCORE_FLOOR} nats, the same for every batch, so a molecule's reward depends on that molecule alone. 'log_score_floor: null' would NOT mean 'no floor' -- the power transform then uses a BATCH-RELATIVE floor 20 nats below each batch's own maximum, making a molecule's reward depend on its batch-mates; -750 is how no floor is expressed, sitting just below float64's exp() underflow (log(5e-324) = -744.4). Loss: VARGRAD, so there is no log_z parameter; INITIAL_LOG_Z (${INITIAL_LOG_Z}) and LOG_Z_LR (${LOG_Z_LR}) are passed but UNUSED.
WHERE THIS ARM SITS IN THE GRID. The floors are sited on the log GIBBON distribution this surrogate gives gp_molformer_set, which is not a proxy for the step-0 pool -- it IS the step-0 pool, 83,811 molecules from the GP-MoLFormer prior the policy starts as. That distribution is tight (max -27.6, p99 -104.9, median -147.8, p1 -184.8, min -219.9), so the above-floor fraction at step 0 is 1.0000 at -750 (nothing scores below -220, making -250/-400/-750 the same experiment), ~0.93 at -175, 0.5532 at -150 and ~0.03 at -125. -125 is the boundary case and is EXPECTED TO FAIL: with almost the whole batch tied the reward expresses nearly no preference. Floors shallower than -100 are dead (0.006 above floor at -100) and deeper than -250 are all the same arm. On beta: the reward ratio across the pool at a non-binding floor is exp(beta * spread) and the bulk spread is p1-to-p99 = ~80 nats, not the 192 of min-to-max, so 0.05 spans ~55x, 0.1 ~3,000x, 0.2 ~9e6 and 0.4 ~8e13.
EVERY FIGURE IN THIS RUN CAN BE REDRAWN FROM DISK, and the final pool's posterior is kept. ${OUTPUT_DIR}/artifacts/acquisition/general/round_0001/scored_samples.csv has one row per molecule of the final 100,000-molecule sample: x (the SMILES), gp_mean (the surrogate's mean prediction), gp_std (the LATENT sd, which is what GIBBON consumed), gp_total_std (latent + fitted noise), acquisition_score (the raw GIBBON value), reward_input_score (max(log(acq), ${LOG_SCORE_FLOOR}), the log GIBBON the policy was trained against) and the selector's ranking_score/selected/selection_rank. That file exists because run_writer.write_sample_scores is true; without it the pool is SMILES and fidelity only and the posterior behind every acquisition score is lost, since the policy is not checkpointed. Note that reward_input_score would be DROPPED under the power transform with a null floor, a batch-relative floor being undefined for a 100k pool that was never one of the sampler's batches -- a second reason the fixed -750 is how no floor is expressed here. ${OUTPUT_DIR} also holds training_steps_round_0001.csv (one row per training step: generated/valid/synthesizable/unique counts, the three loss series, acq_max/median/min, reward_mean/max/min/median and their ratios, reward_effective_support, above_floor_fraction, residuals and gradient norms) and training_samples_round_0001.csv.gz (every molecule generated while training, with its step, post-transform reward_input_score and raw acquisition_score), alongside generated_pool_round_0001.csv (the 100k post-training snapshot) and rejects_round_0001.csv. The three curves this grid is for -- validity (valid/generated), synthesizability (synthesizable/valid) and log GIBBON reward (acq_* and reward_*) over training steps -- all come from the step log. The policy is NOT checkpointed, so these files are the only record.
WHAT TO READ: train/batch_health for BOTH validity and uniqueness, read AT the step acq/trajectory plateaus rather than as a run-level average; reward/above_floor_fraction, both its value at step 0 (which should match the grid figure above -- if it does not, the policy's pool already differs from gp_molformer_set more than expected) and the step at which it leaves zero in the shallow-floor arms; the step at which acq/trajectory plateaus; reward/effective_support and reward/concentration; and train/prior_divergence.
THE svgp300k ARMS' FINDINGS DO NOT TRANSFER and must not be used to interpret this grid. That surrogate is a variational GP on 300,000 molecules; this is an exact GP on 20,000. Its log GIBBON sits one to two orders of magnitude lower (median -2,688 against -148) and ~77% of a trained pool underflowed to exactly 0.0, against essentially none here. Its -500 default, its ~1,150-step warmup, and its floor->validity / beta->diversity mapping are properties of that reward landscape. Only the algebra is shared.
TRAINING LENGTH: this arm runs ${N_TRAIN_STEPS} steps. A shorter arm is not a longer one stopped early -- the warmup/decay schedule spans n_train_steps.
Prior/policy: STOCK ibm-research/GP-MoLFormer-Uniq, not fine-tuned.
Arm name: ${ARM}. Submitted with: sbatch jobs/reward_shape_exact20k_mila.sh $*   (Slurm job ${SLURM_JOB_ID:-unknown})"
printf '%s\n' "${WANDB_NOTES}" > "${OUTPUT_DIR}/arm_description.txt"

RUN_NAME="${ARM//_/-}-${SLURM_JOB_ID:-local}"
TAGS="reward-shape,beta-scan,no-docking,${TRANSFORM},beta${BETA},vargrad,floor${LOG_SCORE_FLOOR},nonbinding-floor,steps${N_TRAIN_STEPS},${SURROGATE_TAGS},restored-state,1-round${SEED_TAG}${ARM_SUFFIX:+,suffix${ARM_SUFFIX}}"

# Restated as one YAML value each rather than overridden field by field:
# OmegaConf.from_dotlist reads every path segment as a dict key, so
# `oracle.type=...` would merge onto the config's Dock3Oracle mapping and leave
# its receptor paths behind.
ORACLE="oracle={type: ConstantOracle, value: 0.0, fidelity_costs: {1: 1.0}}"
BUDGET="budget={available_budget: 1.0, schedule: {type: constant, value: 1.0}, max_rounds: 1}"
LOGGER="logger={type: MultiLogger, loggers: [{type: ConsoleLogger, project_name: ${WANDB_PROJECT}, run_name: ${RUN_NAME}}, {type: WandbLogger, project_name: ${WANDB_PROJECT}, run_name: ${RUN_NAME}, entity: models-mila5723, group: beta-scan, tags: [mila,${TAGS}]}]}"

# "${@:6}": argument 5 is the seed, consumed above.
uv run --no-sync python scripts/run_with_sigterm_flush.py \
  "${BASE_CONFIG}" \
  "sampler.reward_transform=${TRANSFORM}" \
  "sampler.beta=${BETA}" \
  "sampler.n_train_steps=${N_TRAIN_STEPS}" \
  "sampler.initial_log_z=${INITIAL_LOG_Z}" \
  "sampler.loss=${LOSS}" \
  "sampler.log_score_floor=${LOG_SCORE_FLOOR}" \
  "sampler.log_z_learning_rate=${LOG_Z_LR}" \
  "sampler.max_generation_attempts=500000" \
  "${SEED_OVERRIDE[@]}" \
  "run_writer.output_dir=${OUTPUT_DIR}" \
  "sampler.reject_log_path=${REJECT_LOG_PATH}" \
  "sampler.pool_log_path=${POOL_LOG_PATH}" \
  "sampler.training_step_log_path=${TRAINING_STEP_LOG_PATH}" \
  "sampler.training_sample_log_path=${TRAINING_SAMPLE_LOG_PATH}" \
  "${ORACLE}" \
  "${BUDGET}" \
  "${LOGGER}" \
  "${@:6}"
