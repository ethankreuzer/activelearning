#!/bin/bash
# The latent-variance CALIBRATION grid: 24 cells, ungated, on the 300k arm.
#
#   sbatch --array=1-24 jobs/variational_gp_calibration_grid_mila.sh
#
# WHY THIS GRID EXISTS. The S3-GFN run's GIBBON rewards are near-zero for
# almost every molecule it samples, and the cause is on the SURROGATE side, not
# the prior side. Measured on the saved per-molecule eval files of the existing
# 300k fits (eval/*.csv carry mean, std_latent and y, so no refit is needed):
#
#   bin the molecules by latent variance s^2, take each decile's mean s^2 and
#   mean squared residual, fit a line with a free intercept. A calibrated model
#   gives SLOPE 1. The 5,032-step fit gives 20.8 / 27.8 / 22.2 on ampc_331k,
#   gentail and gp_molformer_set; the 10,064-step fit gives 24.9 / 33.4 / 25.6.
#
# So the latent variance understates how fast the errors actually grow by a
# factor of roughly 20, and GIBBON reads exactly that variance. s^2 averages
# 6.2e-5 against noise 2.7e-3, so a 20x correction would still leave epistemic
# variance below the noise term -- about half of what the fit now calls noise is
# uncertainty the model should own. That is the ELBO pathology (outputscale 0.23
# against noise 0.42), not a physical limit.
#
# THE HYPOTHESIS. For a Gaussian likelihood the ELBO's per-point term is
# log N(y | mu, sigma^2) - s^2 / (2 sigma^2): the variational variance is a PURE
# PENALTY, so the objective is maximized by driving it to zero. The predictive
# log likelihood's term is log N(y | mu, sigma^2 + s^2), which folds s^2 into
# the predictive distribution and therefore has an INTERIOR optimum at the
# variance matching the observed error. See src/activelearning/surrogate/
# objectives.py, which already cites Jankowiak et al. 2020 for this.
#
#   That also predicts an INTERACTION with step count, which is why both are
#   swept: under the ELBO, latent std fell 12.4% from 5,032 to 10,064 steps for
#   no accuracy gain at all (RMSE 0.07117 vs 0.07116). Under PLL that drift
#   should largely vanish, because the optimum is interior rather than at the
#   boundary.
#
# ARD IS THE OTHER CANDIDATE. The current fits share ONE lengthscale (7.1-7.6)
# across all 512 MiniMol features. An isotropic kernel must compromise over
# every direction, which flattens the function and pushes the residual into the
# noise term -- the same symptom from a different cause. ARD gives each feature
# its own lengthscale under the same prior.
#
# UNGATED, AND THERE IS NO OBJECTIVE. An earlier draft of this grid carried a
# maximand and hard gates on pearson and spearman, copied from the prior-RTB
# sweeps. Both were wrong here. A GRID HAS NO OPTIMIZER TO STEER, so gates buy
# nothing and can only discard cells before anyone has seen the trade-off; and
# the gate values would have been set at today's numbers (pearson 0.796,
# gentail spearman 0.587), which silently assumes PLL costs no fit. PLL
# plausibly does cost some. config/ampc/prior_rtb_sweep3.yaml records what that
# mistake looks like: sweep 1's 0.95 validity gate truncated the only
# productive direction and the search learned to avoid it.
#
#   So every cell runs and every cell is recorded. Read the achieved
#   (pearson, slope) frontier afterwards and decide with the numbers in hand.
#
# WHAT IS FIXED AND WHY. n stays at 300,000: the 50k and 100k arms are strictly
# worse on BOTH axes (coverage_1std 0.138 and 0.264 against 0.459, r2 -4.29 and
# -2.15 against -0.583), so lower n buys GIBBON signal only by making the model
# more ignorant. M stays at 1024: the M2048 arm changed nothing (r2 -0.590,
# coverage 0.453) and costs far more to train. noise_prior stays on: the fit
# lands at 0.42 against the prior's mode of 0.0067, about 3.1 sigma away in log
# space, so the prior is already overruled by the data and toggling it is a
# diagnostic rather than a treatment.
#
# THE SLOPE IS NOT COMPUTED HERE. It needs no code change and no GPU: every run
# writes eval/*.csv with per-molecule mean, std_latent and y, so all 24 slopes
# come from those files afterwards. Note the de-bias step is NOT optional --
# regress y on mu and rescale s^2 by that gain first, because on raw residuals
# the mean's systematic error swamps the variance signal and the slope comes out
# NEGATIVE (-11.7 on ampc_331k).
#
#SBATCH --partition=main
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
# The 300k arm at 10,064 steps fitted in 204 s and the whole run took ~9 min
# with SCORE_AMPC=0. ARD fits 512 lengthscales instead of 1 and may be several
# times slower, so 3 h is a wide margin.
#SBATCH --time=03:00:00
#SBATCH --job-name=ampc_svgp_calib
#SBATCH --output=slurm_logs/ampc_svgp_calib_%A_%a.out
#SBATCH --error=slurm_logs/ampc_svgp_calib_%A_%a.err

set -euo pipefail

# The grid, flattened so the array index names the cell. Order is chosen so that
# consecutive indices differ in SEED: with only two GPUs available the early
# indices then cover both objectives and both ARD settings first, and a grid cut
# short by anything still has a usable spread rather than only elbo cells.
OBJECTIVES=(elbo pll)
ARDS=(false true)
STEPSET=(5000 10000)
SEEDS=(42 43 44)

TASK="${SLURM_ARRAY_TASK_ID:-1}"
TOTAL=$(( ${#OBJECTIVES[@]} * ${#ARDS[@]} * ${#STEPSET[@]} * ${#SEEDS[@]} ))
if (( TASK < 1 || TASK > TOTAL )); then
  echo "Array index must be 1-${TOTAL}, got ${TASK}. Submit with --array=1-${TOTAL}." >&2
  exit 2
fi

INDEX=$(( TASK - 1 ))
SEED="${SEEDS[INDEX % ${#SEEDS[@]}]}"
INDEX=$(( INDEX / ${#SEEDS[@]} ))
STEPS_REQ="${STEPSET[INDEX % ${#STEPSET[@]}]}"
INDEX=$(( INDEX / ${#STEPSET[@]} ))
ARD="${ARDS[INDEX % ${#ARDS[@]}]}"
INDEX=$(( INDEX / ${#ARDS[@]} ))
OBJECTIVE_ARG="${OBJECTIVES[INDEX]}"

case "${OBJECTIVE_ARG}" in
  elbo) OBJECTIVE="VariationalELBO" ;;
  pll) OBJECTIVE="PredictiveLogLikelihood" ;;
  *) echo "Bad objective ${OBJECTIVE_ARG}." >&2; exit 2 ;;
esac

SIZE=300000
M=1024
LR=3e-3
SET_LABEL=svgpscale
TRAIN_CSV="data/ampc_${SET_LABEL}_${SIZE}.csv"
TRAIN_CACHE="cache/ampc/exact_dkl_top_n/${SET_LABEL}_${SIZE}.npy"

# Whole epochs of ceil(SIZE / 4096) = 74 minibatches, exactly as
# variational_gp_scaling_mila.sh computes them, so the step counts match the
# existing arms' names: 5000 -> 68 epochs = 5032 steps, 10000 -> 136 = 10064.
STEPS_PER_EPOCH=$(( (SIZE + 4095) / 4096 ))
EPOCHS=$(( (STEPS_REQ + STEPS_PER_EPOCH - 1) / STEPS_PER_EPOCH ))
TOTAL_STEPS=$(( EPOCHS * STEPS_PER_EPOCH ))

# The scaling script's arm name carries neither ard nor seed, so all 24 cells
# would land in ONE directory and overwrite each other. Hence this wrapper and
# its own name.
ARM="${SET_LABEL}_${SIZE}_svgp_M${M}_$([[ "${ARD}" == true ]] && echo ard || echo iso)_matern_scaled_pm0.03956_${OBJECTIVE_ARG}_cosine_lr${LR}_steps${TOTAL_STEPS}_seed${SEED}_fp32"
OUTPUT_DIR="outputs/ampc/variational_gp/calibration_grid/${ARM}"

# GIBBON builds an (M + 1)^2 covariance per candidate; this is the scaling
# script's own formula, ~1 GB per chunk at M = 1024.
SCORE_CHUNK=$(( 256 * 1024 * 1024 / (M * M) ))
(( SCORE_CHUNK < 1 )) && SCORE_CHUNK=1

cd "${HOME}/activelearning"
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export WANDB_DIR="${OUTPUT_DIR}/wandb"

for required in "${TRAIN_CSV}" "${TRAIN_CACHE}" \
  "cache/ampc/exact_dkl_top_n/gentail.csv" \
  "config/ampc/variational_gp_scaling.yaml"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing ${required}." >&2
    exit 2
  fi
done

if [[ -e "${OUTPUT_DIR}/eval_summary.json" ]]; then
  echo "${OUTPUT_DIR} already holds a finished run; refusing to overwrite." >&2
  exit 2
fi
mkdir -p "${WANDB_DIR}" slurm_logs

echo "Cell ${TASK}/${TOTAL}: ${OBJECTIVE_ARG} ard=${ARD} steps=${TOTAL_STEPS} seed=${SEED}"
echo "  -> ${OUTPUT_DIR}"

# ampc_331k is PREDICTED but not GIBBON-scored: its two scoring passes were 73%
# of the 50k arm's runtime, and the calibration slope needs only the prediction
# columns, which --predict-only still writes.
uv run --no-sync python -m scripts.exact_dkl_top_n \
  config/ampc/variational_gp_scaling.yaml \
  "dataset.initial_data.path=${TRAIN_CSV}" \
  "surrogate.encoder.feature_cache_path=${TRAIN_CACHE}" \
  "surrogate.training_params.variational_objective=${OBJECTIVE}" \
  "surrogate.training_params.lr=${LR}" \
  "surrogate.training_params.epochs=${EPOCHS}" \
  "surrogate.num_inducing=${M}" \
  "surrogate.ard=${ARD}" \
  "runtime.seed=${SEED}" \
  --output-dir "${OUTPUT_DIR}" \
  --cache-dir cache/ampc/exact_dkl_top_n \
  --eval-chunk-size 2000 \
  --score-chunk-size "${SCORE_CHUNK}" \
  --extra-set gentail \
  --predict-only ampc_331k \
  --wandb-project ampc-Variational-GP \
  --wandb-entity models-mila5723 \
  --wandb-group calibration-grid \
  --wandb-tags "variational-gp,svgp,M${M},calibration-grid,${SET_LABEL},n${SIZE},${OBJECTIVE_ARG},${OBJECTIVE},ard-${ARD},matern,steps${TOTAL_STEPS},seed${SEED},fp32" \
  --run-name "${ARM//_/-}-${SLURM_JOB_ID}"

echo
echo "Done: ${OUTPUT_DIR}/eval_summary.json"
echo "Compute the calibration slope from ${OUTPUT_DIR}/eval/*.csv once all 24"
echo "cells have finished; de-bias first (regress y on mean, rescale std_latent"
echo "by that gain), then bin by s^2 and fit with a free intercept."
