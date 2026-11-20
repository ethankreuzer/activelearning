#!/bin/bash
# One arm of the K ladder on Mila, without docking: one active-learning round
# training the S3-GFN policy against a fixed GIBBON acquisition on a frozen
# surrogate, generating its pool, and stopping.
#
#   sbatch jobs/k_ladder_mila.sh <exactgp|elbo|pll> <K> <seed>
#   sbatch jobs/k_ladder_mila.sh elbo 100 42
#   DRY_RUN=1 bash jobs/k_ladder_mila.sh pll 300 43    # resolve and validate, run nothing
#
# jobs/submit_k_ladder.sh submits the whole ladder: 3 surrogates x 5 K x 2 seeds.
#
# WHAT K IS. K = R(p99) / R(median): how many times larger the reward of the
# 99th-percentile molecule is than the median molecule's, on the surrogate's own
# log-GIBBON distribution on gp_molformer_set. That set is the policy's step-0
# sampling distribution (83,811 molecules drawn from the GP-MoLFormer prior the
# policy is initialised to), and in a 64-molecule batch the best molecule sits
# near the 99th percentile while a typical one sits at the median, so K is
# roughly the reward ratio between the best and a typical molecule in a batch.
#
# The reward is log R = beta * r with r the log-GIBBON score, so
#
#   K = exp(beta * (r_p99 - r_median))     beta = ln(K) / (r_p99 - r_median)
#
# beta has units of 1/nats, so the same beta is a different reward on every
# surrogate whose scores are spread differently; K is the same quantity on all of
# them. scripts/k_ladder_params.py derives beta from the arm's own
# eval/gp_molformer_set.csv. The ladder is K = 10, 30, 100, 300, 1000:
#
#                 ExactGP     ELBO      PLL      (p99 - median, nats: 42.9 / 193.9 / 42.8)
#   K =   10      0.0537    0.0119   0.0538
#   K =   30      0.0793    0.0175   0.0795
#   K =  100      0.1074    0.0238   0.1076
#   K =  300      0.1330    0.0294   0.1333
#   K = 1000      0.1611    0.0356   0.1614
#
# FLOOR. The 5th percentile of the same distribution, one number per surrogate,
# constant across steps and not swept: -174.1 ExactGP, -608.9 ELBO, -313.0 PLL.
# It is a numerical guard against log(0) and an outlier tail, not reward shaping.
#
# SURROGATES, all restored rather than refitted, all on the 20,000-molecule
# svgpscale set with the lengthscale prior on (the four carried into the
# active-learning run; the prior-off ExactGP arm is not in this ladder):
#
#   exactgp  outputs/ampc/exact_gp_sweep/n20000_on_pinned         pinned mean
#   elbo     outputs/ampc/variational_gp_sweep/n20000_elbo_on_s10000   10,000 steps
#   pll      outputs/ampc/variational_gp_sweep/n20000_pll_on_s10000    10,000 steps
#
# TRAINING LENGTH is fixed at 5,000 steps. Healthy arms plateau by about step
# 2,500-3,000, but an arm that collapses is still losing validity until about
# step 4,000, so a shorter run would understate exactly the failures this ladder
# is looking for.
#
# GENERATION CAP. max_generation_attempts defaults to 500,000. At a healthy K the
# 100,000-molecule pool takes ~105,000 attempts, so the cap never binds; at a
# collapsed one the yield is ~0 and an uncapped arm grinds to 2,000,000 attempts
# (~1.7 h) for no result. Override with MAX_GENERATION_ATTEMPTS=... if needed.
#
# RESOURCES. Measured on the ExactGP arms: ~1.8 h for a healthy arm (1.45 h of
# training, 0.12 h of generation), ~2.9 h for one that ran to the 2,000,000
# attempt cap. The variational arms are unmeasured. 8 h leaves room for both.
# Mila's per-user quota on main is 2 GPUs, so a batch of submissions runs two at
# a time and the rest wait on an association limit rather than being rejected.
#SBATCH --partition=main
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=8:00:00
#SBATCH --job-name=ampc_k_ladder
#SBATCH --output=slurm_logs/ampc_k_ladder_%j.out
#SBATCH --error=slurm_logs/ampc_k_ladder_%j.err

set -euo pipefail

USAGE="usage: sbatch jobs/k_ladder_mila.sh <exactgp|elbo|pll> <K> <seed> [extra Hydra-style overrides...]"
SURROGATE="${1:-}"
K="${2:-}"
SEED="${3:-}"
if [[ ! "${SURROGATE}" =~ ^(exactgp|elbo|pll)$ \
      || ! "${K}" =~ ^[0-9]*\.?[0-9]+$ \
      || ! "${SEED}" =~ ^[0-9]+$ ]]; then
  echo "${USAGE}" >&2
  echo "K is a number above 1 (the reward ratio of the 99th-percentile molecule to the median); the seed is an integer." >&2
  exit 2
fi

N_TRAIN_STEPS=5000
MAX_GENERATION_ATTEMPTS="${MAX_GENERATION_ATTEMPTS:-500000}"
TRANSFORM="power"
LOSS="vargrad"
# Unused under vargrad, which has no log_z parameter; passed only so these arms
# are configured exactly as the ExactGP arms of the earlier scan were.
INITIAL_LOG_Z="0"
LOG_Z_LR="0.01"
WANDB_PROJECT="thesis-ampc-k-ladder"

case "${SURROGATE}" in
  exactgp)
    BASE_CONFIG="config/ampc/reward_shape_exact20k.yaml"
    FITTED_DIR="outputs/ampc/exact_gp_sweep/n20000_on_pinned"
    SURROGATE_OVERRIDES=("surrogate.state_path=${FITTED_DIR}/surrogate_state.pt")
    SURROGATE_DESC="BoTorch exact GP, isotropic Matern, lengthscale prior ON, mean pinned at 0.0042864, fitted to the 20,000-molecule svgpscale set"
    SURROGATE_TAGS="exactgp,fixedfeature,svgpscale-20k,prior-on,pinned-mean,iso-matern,fp32"
    ;;
  elbo)
    BASE_CONFIG="config/ampc/reward_shape_variational20k.yaml"
    FITTED_DIR="outputs/ampc/variational_gp_sweep/n20000_elbo_on_s10000"
    SURROGATE_OVERRIDES=("surrogate.state_path=${FITTED_DIR}/surrogate_state.pt"
                         "surrogate.training_params.variational_objective=VariationalELBO")
    SURROGATE_DESC="sparse variational GP (1,024 inducing points), ELBO objective, lengthscale prior ON, 10,000 training steps, fitted to the 20,000-molecule svgpscale set"
    SURROGATE_TAGS="variational,elbo,svgpscale-20k,prior-on,steps10000,iso-matern,fp32"
    ;;
  pll)
    BASE_CONFIG="config/ampc/reward_shape_variational20k.yaml"
    FITTED_DIR="outputs/ampc/variational_gp_sweep/n20000_pll_on_s10000"
    SURROGATE_OVERRIDES=("surrogate.state_path=${FITTED_DIR}/surrogate_state.pt"
                         "surrogate.training_params.variational_objective=PredictiveLogLikelihood")
    SURROGATE_DESC="sparse variational GP (1,024 inducing points), predictive-log-likelihood objective, lengthscale prior ON, 10,000 training steps, fitted to the 20,000-molecule svgpscale set"
    SURROGATE_TAGS="variational,pll,svgpscale-20k,prior-on,steps10000,iso-matern,fp32"
    ;;
esac

# sbatch runs a copy of this script from a spool directory, so go home first.
cd "${HOME}/activelearning"

STATE_PATH="${FITTED_DIR}/surrogate_state.pt"
EVAL_CSV="${FITTED_DIR}/eval/gp_molformer_set.csv"

for required in \
  "${STATE_PATH}" \
  "${EVAL_CSV}" \
  "${BASE_CONFIG}" \
  "scripts/k_ladder_params.py" \
  "data/ampc_svgpscale_20000.csv" \
  "cache/ampc/exact_dkl_top_n/svgpscale_20000.npy" \
  "minimol_ampc_encoder/model/final.pt"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing ${required}." >&2
    echo "This arm restores a fitted surrogate rather than fitting one, and derives" >&2
    echo "beta and the floor from that surrogate's gp_molformer_set scores, so the" >&2
    echo "fitted state, its eval CSV, its training set, its feature cache and the" >&2
    echo "MiniMol encoder must all be present." >&2
    exit 2
  fi
done

# beta and the floor come from this surrogate's own distribution, not from a
# table: a stale number here would silently mislabel the arm.
if ! read -r BETA LOG_SCORE_FLOOR DIST_MEDIAN DIST_P99 DIST_P5 DIST_N \
    < <(python3 scripts/k_ladder_params.py "${EVAL_CSV}" --k "${K}"); then
  echo "scripts/k_ladder_params.py failed on ${EVAL_CSV}." >&2
  exit 2
fi
if [[ -z "${BETA:-}" || -z "${LOG_SCORE_FLOOR:-}" ]]; then
  echo "scripts/k_ladder_params.py printed nothing for K=${K} on ${EVAL_CSV}." >&2
  exit 2
fi

ARM="kladder_${SURROGATE}_K${K}_s${SEED}_steps${N_TRAIN_STEPS}"
# Optional suffix for rerunning an arm whose settings are unchanged, so the
# rerun gets its own directory instead of colliding with the earlier pool.
ARM_SUFFIX="${ARM_SUFFIX:-}"
if [[ -n "${ARM_SUFFIX}" && ! "${ARM_SUFFIX}" =~ ^_[A-Za-z0-9.-]+$ ]]; then
  echo "ARM_SUFFIX must look like _name (letters, digits, '.', '-')." >&2
  exit 2
fi
ARM="${ARM}${ARM_SUFFIX}"

OUTPUT_DIR="outputs/ampc/k_ladder/${ARM}"
REJECT_LOG_PATH="${OUTPUT_DIR}/rejects_round_{round}.csv"
POOL_LOG_PATH="${OUTPUT_DIR}/generated_pool_round_{round}.csv"
# Per-step series and training molecules go to disk, not only to W&B: the W&B
# panels are PNGs and the policy is never checkpointed, so these two files are
# the only record that can be re-binned, overlaid across arms or redrawn.
TRAINING_STEP_LOG_PATH="${OUTPUT_DIR}/training_steps_round_{round}.csv"
TRAINING_SAMPLE_LOG_PATH="${OUTPUT_DIR}/training_samples_round_{round}.csv.gz"
# Attempts, yield, reject reasons and the training totals, written even when the pool
# cannot be generated. round_history.jsonl is empty for exactly the arms that collapse,
# so without this file their generation numbers exist only in W&B and the Slurm log.
GENERATION_SUMMARY_PATH="${OUTPUT_DIR}/generation_summary_round_{round}.json"

RUN_NAME="${ARM//_/-}-${SLURM_JOB_ID:-local}"
TAGS="k-ladder,no-docking,${TRANSFORM},K${K},beta${BETA},vargrad,floor${LOG_SCORE_FLOOR},floor-p5,steps${N_TRAIN_STEPS},${SURROGATE_TAGS},restored-state,1-round,seed${SEED}${ARM_SUFFIX:+,suffix${ARM_SUFFIX}}"

# Restated as one YAML value each rather than overridden field by field:
# OmegaConf.from_dotlist reads every path segment as a dict key, so
# `oracle.type=...` would merge onto the config's Dock3Oracle mapping and leave
# its receptor paths behind.
ORACLE="oracle={type: ConstantOracle, value: 0.0, fidelity_costs: {1: 1.0}}"
BUDGET="budget={available_budget: 1.0, schedule: {type: constant, value: 1.0}, max_rounds: 1}"
LOGGER="logger={type: MultiLogger, loggers: [{type: ConsoleLogger, project_name: ${WANDB_PROJECT}, run_name: ${RUN_NAME}}, {type: WandbLogger, project_name: ${WANDB_PROJECT}, run_name: ${RUN_NAME}, entity: models-mila5723, group: k-ladder-${SURROGATE}, tags: [mila,${TAGS}]}]}"

OVERRIDES=(
  "${SURROGATE_OVERRIDES[@]}"
  "sampler.reward_transform=${TRANSFORM}"
  "sampler.beta=${BETA}"
  "sampler.n_train_steps=${N_TRAIN_STEPS}"
  "sampler.initial_log_z=${INITIAL_LOG_Z}"
  "sampler.loss=${LOSS}"
  "sampler.log_score_floor=${LOG_SCORE_FLOOR}"
  "sampler.log_z_learning_rate=${LOG_Z_LR}"
  "sampler.max_generation_attempts=${MAX_GENERATION_ATTEMPTS}"
  "runtime.seed=${SEED}"
  "run_writer.output_dir=${OUTPUT_DIR}"
  "sampler.reject_log_path=${REJECT_LOG_PATH}"
  "sampler.pool_log_path=${POOL_LOG_PATH}"
  "sampler.training_step_log_path=${TRAINING_STEP_LOG_PATH}"
  "sampler.training_sample_log_path=${TRAINING_SAMPLE_LOG_PATH}"
  "sampler.generation_summary_path=${GENERATION_SUMMARY_PATH}"
  "${ORACLE}"
  "${BUDGET}"
  "${LOGGER}"
  "${@:4}"
)

source .venv/bin/activate
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1   # nothing is downloaded: the prior and encoder are cached

echo "arm        ${ARM}"
echo "surrogate  ${SURROGATE}   (${FITTED_DIR})"
echo "K          ${K}   beta ${BETA}   floor ${LOG_SCORE_FLOOR} (5th percentile)"
echo "dist       median ${DIST_MEDIAN}   p99 ${DIST_P99}   p5 ${DIST_P5}   n ${DIST_N}"
echo "steps      ${N_TRAIN_STEPS}   seed ${SEED}   max_generation_attempts ${MAX_GENERATION_ATTEMPTS}"
echo "output     ${OUTPUT_DIR}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  # Parse the merged config with the project's own schema, exactly as the run
  # will, and print what it resolved to. Nothing is created and nothing runs.
  uv run --no-sync python - "${BASE_CONFIG}" "${OVERRIDES[@]}" <<'PY'
import sys
from activelearning.config import ActiveLearningConfig
from activelearning.utils.config_loader import load_and_parse

cfg = load_and_parse(sys.argv[1], ActiveLearningConfig, overrides=sys.argv[2:])
sampler, surrogate = cfg.sampler, cfg.surrogate
objective = getattr(getattr(surrogate, "training_params", None), "variational_objective", None)
print("-- config parsed; resolved values --")
print("surrogate.type        ", type(surrogate).__name__)
print("surrogate.state_path  ", surrogate.state_path)
print("surrogate objective   ", objective)
print("acq score_chunk_size  ", cfg.acquisition.score_chunk_size)
print("sampler.beta          ", sampler.beta)
print("sampler.log_score_floor", sampler.log_score_floor)
print("sampler.loss          ", sampler.loss)
print("sampler.n_train_steps ", sampler.n_train_steps)
print("sampler.max_generation_attempts", sampler.max_generation_attempts)
print("sampler.generation_summary_path", sampler.generation_summary_path)
print("runtime.seed          ", cfg.runtime.seed)
print("oracle.type           ", type(cfg.oracle).__name__)
PY
  echo "DRY_RUN=1: stopping before any output directory is created."
  exit 0
fi

# Never overwrite an earlier run's generated pool: it is the arm's result.
if compgen -G "${OUTPUT_DIR}/generated_pool_round_*.csv" > /dev/null; then
  echo "${OUTPUT_DIR} already holds a generated pool from an earlier run." >&2
  echo "Rerun with a suffix, e.g. ARM_SUFFIX=_rerun2 sbatch jobs/k_ladder_mila.sh $*" >&2
  exit 2
fi

# Per-arm, so concurrent arms never race on a shared ./wandb/latest-run symlink.
export WANDB_DIR="${OUTPUT_DIR}"
mkdir -p "${WANDB_DIR}" slurm_logs

# Machine-readable record of what this arm is, written BEFORE anything runs so a crashed
# or collapsed arm still has it. K, the distribution it was derived from and the git
# state are otherwise nowhere on disk except in the folder name: run_manifest.json holds
# beta and the floor but not K. This is what a figure script should group and label by.
GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
GIT_DIRTY="false"
if [[ -n "$(git status --porcelain --untracked-files=no 2>/dev/null)" ]]; then GIT_DIRTY="true"; fi
cat > "${OUTPUT_DIR}/arm_params.json" <<JSON
{
  "arm": "${ARM}",
  "surrogate": "${SURROGATE}",
  "K": ${K},
  "beta": ${BETA},
  "log_score_floor": ${LOG_SCORE_FLOOR},
  "floor_rule": "5th percentile of score_gibbon_log on gp_molformer_set",
  "dist_median": ${DIST_MEDIAN},
  "dist_p99": ${DIST_P99},
  "dist_p5": ${DIST_P5},
  "dist_n": ${DIST_N},
  "dist_csv": "${EVAL_CSV}",
  "seed": ${SEED},
  "n_train_steps": ${N_TRAIN_STEPS},
  "max_generation_attempts": ${MAX_GENERATION_ATTEMPTS},
  "reward_transform": "${TRANSFORM}",
  "loss": "${LOSS}",
  "state_path": "${STATE_PATH}",
  "base_config": "${BASE_CONFIG}",
  "slurm_job_id": "${SLURM_JOB_ID:-none}",
  "git_commit": "${GIT_COMMIT}",
  "git_dirty": ${GIT_DIRTY}
}
JSON
python3 -c "import json,sys; json.load(open(sys.argv[1]))" "${OUTPUT_DIR}/arm_params.json"

export WANDB_NOTES="K-LADDER arm ${ARM}, on Mila, WITHOUT DOCKING. One active-learning round training the S3-GFN policy against a fixed GIBBON acquisition on a frozen surrogate, generating its pool, then one stubbed oracle query.
THE QUESTION: how does the reward's dynamic range shape training, on a scale that means the same thing for every surrogate? K = R(p99)/R(median) is how many times larger the reward of the 99th-percentile molecule is than the median molecule's, on this surrogate's log-GIBBON distribution on gp_molformer_set (n=${DIST_N}), which is the policy's step-0 sampling distribution. The reward is log R = beta * r with r the log GIBBON, so K = exp(beta * (p99 - median)) and beta = ln(K) / (p99 - median). This arm: K=${K}, median ${DIST_MEDIAN}, p99 ${DIST_P99}, so beta=${BETA}. beta has units of 1/nats and does not transfer between surrogates; K does. The ladder is K = 10, 30, 100, 300, 1000, two seeds (42, 43) each, on ExactGP, ELBO and PLL.
FLOOR: ${LOG_SCORE_FLOOR}, the 5th percentile of the same distribution (${DIST_P5}), fixed for every batch so a molecule's reward depends on that molecule alone. It is a numerical guard, not a tuned knob, and is the same for every K on this surrogate.
ORACLE IS A STUB. oracle=ConstantOracle returns the same label for every molecule and the budget is 1, so one molecule is 'labelled' at a made-up value and the round closes. Mila has no AmpC receptor files and no dock64. IGNORE dataset/general/docked/*, surrogate/general/held_out/* and every other y-derived number in this run.
SURROGATE: RESTORED, NOT REFITTED. ${SURROGATE_DESC}; parameters loaded from ${STATE_PATH}.
Acquisition: GIBBON (QLowerBoundMaxValueEntropy), log_space=true so the information gain never underflows, log_output=false so it is returned on the VALUE scale, which is what the power transform logs. Reward: transform=${TRANSFORM}, loss=${LOSS}.
WHAT TO READ: train/batch_health for BOTH validity and uniqueness. A healthy arm has unique_rate equal to validity_rate to three digits; unique falling below valid is the first sign of collapse onto a few modes, visible within the first 500 steps. Then generation/yield (healthy ~0.95; a collapsed policy yields ~0 and hits the ${MAX_GENERATION_ATTEMPTS}-attempt cap), train/prior_divergence (near 0 means the policy never left the prior), reward/effective_support and reward/concentration. A falling vargrad loss is not by itself good news: it can mean convergence onto an off-manifold mode.
REFERENCE, the earlier ExactGP scan (project thesis-ampc-reward-shape-exact20k): K=8.5 (beta 0.05) never left the prior, K=73 (beta 0.1) was healthy with validity ~0.95 throughout, K=5,300 (beta 0.2) lost validity and diversity, and K=2.8e7 (beta 0.4) collapsed. K between 73 and 5,300 was never sampled; that is the gap this ladder fills.
TRAINING LENGTH: ${N_TRAIN_STEPS} steps. A shorter arm is not a longer one stopped early -- the warmup/decay schedule spans n_train_steps.
EVERY FIGURE CAN BE REDRAWN FROM DISK; the policy is NOT checkpointed. ${OUTPUT_DIR} holds training_steps_round_0001.csv (one row per training step), training_samples_round_0001.csv.gz (every molecule generated while training), generation_summary_round_0001.json (attempts, yield, reject counts and training totals, written even if generation fails), generated_pool_round_0001.csv (the post-training snapshot), rejects_round_0001.csv, and artifacts/acquisition/general/round_0001/scored_samples.csv (the final pool's posterior: gp_mean, gp_std, gp_total_std, acquisition_score, reward_input_score).
Prior/policy: STOCK ibm-research/GP-MoLFormer-Uniq, not fine-tuned.
Arm name: ${ARM}. Submitted with: sbatch jobs/k_ladder_mila.sh $*   (Slurm job ${SLURM_JOB_ID:-unknown})"
printf '%s\n' "${WANDB_NOTES}" > "${OUTPUT_DIR}/arm_description.txt"

uv run --no-sync python scripts/run_with_sigterm_flush.py \
  "${BASE_CONFIG}" \
  "${OVERRIDES[@]}"
