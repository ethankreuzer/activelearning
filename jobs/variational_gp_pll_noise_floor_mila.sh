#!/bin/bash
# PLL with a NOISE FLOOR: 80 cells on the 300k arm, following the PLL half of
# the calibration grid (jobs/variational_gp_calibration_grid_mila.sh) with the
# observation noise bounded below.
#
#   sbatch --array=1-80 jobs/variational_gp_pll_noise_floor_mila.sh
#
# SWEPT (5 x 2 x 2 x 2 x 2 = 80):
#   noise_floor, standardized (original-scale variance; var(y) = 0.0063526):
#     0.10 (0.000635), 0.25 (0.001588), 0.42 (0.002668, ELBO's learned noise),
#     0.49 (0.0031063, Gamma-test intercept), 0.67 (0.004256, identical-
#     embedding pairs). The floor is an estimate, so it is bracketed; 0.10
#     shows how low it can go before PLL collapses the noise again.
#   inducing_init: random, stratified (default strata: y quantiles 0.9 / 0.99 /
#     0.999). With a floor, PLL should put latent variance mainly on the hits,
#     which is where stratified places inducing points.
#   ard: false, true. In the grid ARD barely left the isotropic start
#     (effective dims 387-471 of 512, still falling at the last step), so it is
#     paired with a longer arm rather than dropped.
#   steps: 5,032 (68 epochs), 20,128 (272 epochs).
#   seed: 42, 43 (grid seed spread was tiny: pearson +-0.0003).
#
# PINNED: PredictiveLogLikelihood, prior_mean 0.03956, Matern, M = 1024,
# n = 300k, lr 3e-3 cosine to 1e-4, batch 4096, noise_prior, lengthscale_prior,
# scale_inputs, standardize_outputs.
#
# WHY. The calibration grid (job 11519080, 2026-11-03) found that PLL fixes the
# latent-variance calibration slope (~20 -> ~1.3 on ampc_331k) and the GIBBON
# underflow (median log GIBBON on gp_molformer_set -1823 -> -328), but it does so
# by collapsing the noise to ~1.8e-4 (standardized) and moving the unexplained
# variance into the latent variance: total coverage_1std 0.17, and a gentail
# slope of ~0.02.
#
# A nearest-neighbour Gamma test on the same 300k MiniMol features
# (~/scratch/gamma_test/gamma_test.json) puts the variance of y that the
# features cannot explain at 0.0031063 on the original scale = 0.489 of var(y),
# i.e. 0.49 standardized. DOCK3 is deterministic, so this is feature-induced,
# not measurement noise -- but to the GP it is indistinguishable from noise, and
# ELBO's learned noise (0.42) already sits near it. 334 pairs with IDENTICAL
# embeddings and different y confirm the floor is real.
#
# THE QUESTION. With noise held at a floor near 0.49, does PLL keep a calibration slope
# near 1 and non-sparse GIBBON, or does the latent variance shrink back to
# ELBO-like values everywhere except the poorly fit hits? And does the mean
# recover ELBO's pearson (~0.80 vs PLL's 0.79)? Compare against the 12 PLL cells
# of the calibration grid (random init, 5,032 steps, floor absent).
#
# The floor is given on the ORIGINAL target scale (surrogate.noise_floor, a
# variance) and converted with each fit's standardization (y_std = 0.07970 on
# this set). noise_prior stays on as in the grid; with every floor above the
# prior's mode (~0.0067) the noise should sit at the floor, so in practice the
# floor FIXES the noise.
#
# Slopes are computed afterwards from eval/*.csv exactly as for the grid.
#
#SBATCH --partition=main
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --job-name=ampc_svgp_nfloor
#SBATCH --output=slurm_logs/ampc_svgp_nfloor_%A_%a.out
#SBATCH --error=slurm_logs/ampc_svgp_nfloor_%A_%a.err

set -euo pipefail

# Flattened so the array index names the cell. The floor varies fastest, so the
# early indices cover the whole floor range first and a grid cut short still
# traces the floor curve.
NOISE_FLOORS=(0.000635 0.001588 0.002668 0.0031063 0.004256)
INITS=(random stratified)
ARDS=(false true)
STEPSET=(5032 20128)
SEEDS=(42 43)

TASK="${SLURM_ARRAY_TASK_ID:-1}"
TOTAL=$(( ${#NOISE_FLOORS[@]} * ${#INITS[@]} * ${#ARDS[@]} * ${#STEPSET[@]} * ${#SEEDS[@]} ))
if (( TASK < 1 || TASK > TOTAL )); then
  echo "Array index must be 1-${TOTAL}, got ${TASK}. Submit with --array=1-${TOTAL}." >&2
  exit 2
fi

INDEX=$(( TASK - 1 ))
NOISE_FLOOR="${NOISE_FLOORS[INDEX % ${#NOISE_FLOORS[@]}]}"
INDEX=$(( INDEX / ${#NOISE_FLOORS[@]} ))
INIT="${INITS[INDEX % ${#INITS[@]}]}"
INDEX=$(( INDEX / ${#INITS[@]} ))
ARD="${ARDS[INDEX % ${#ARDS[@]}]}"
INDEX=$(( INDEX / ${#ARDS[@]} ))
STEPS_REQ="${STEPSET[INDEX % ${#STEPSET[@]}]}"
INDEX=$(( INDEX / ${#STEPSET[@]} ))
SEED="${SEEDS[INDEX]}"

OBJECTIVE_ARG=pll
OBJECTIVE="PredictiveLogLikelihood"
SIZE=300000
M=1024
LR=3e-3
SET_LABEL=svgpscale
TRAIN_CSV="data/ampc_${SET_LABEL}_${SIZE}.csv"
TRAIN_CACHE="cache/ampc/exact_dkl_top_n/${SET_LABEL}_${SIZE}.npy"

STEPS_PER_EPOCH=$(( (SIZE + 4095) / 4096 ))
EPOCHS=$(( (STEPS_REQ + STEPS_PER_EPOCH - 1) / STEPS_PER_EPOCH ))
TOTAL_STEPS=$(( EPOCHS * STEPS_PER_EPOCH ))

ARM="${SET_LABEL}_${SIZE}_svgp_M${M}_$([[ "${ARD}" == true ]] && echo ard || echo iso)_matern_scaled_pm0.03956_${OBJECTIVE_ARG}_nfloor${NOISE_FLOOR}_init${INIT}_cosine_lr${LR}_steps${TOTAL_STEPS}_seed${SEED}_fp32"
OUTPUT_DIR="outputs/ampc/variational_gp/pll_noise_floor_grid/${ARM}"

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

echo "Cell ${TASK}/${TOTAL}: ${OBJECTIVE_ARG} noise_floor=${NOISE_FLOOR} init=${INIT} ard=${ARD} steps=${TOTAL_STEPS} seed=${SEED}"
echo "  -> ${OUTPUT_DIR}"

uv run --no-sync python -m scripts.exact_dkl_top_n \
  config/ampc/variational_gp_scaling.yaml \
  "dataset.initial_data.path=${TRAIN_CSV}" \
  "surrogate.encoder.feature_cache_path=${TRAIN_CACHE}" \
  "surrogate.training_params.variational_objective=${OBJECTIVE}" \
  "surrogate.training_params.lr=${LR}" \
  "surrogate.training_params.epochs=${EPOCHS}" \
  "surrogate.num_inducing=${M}" \
  "surrogate.ard=${ARD}" \
  "surrogate.noise_floor=${NOISE_FLOOR}" \
  "surrogate.inducing_init=${INIT}" \
  "runtime.seed=${SEED}" \
  --output-dir "${OUTPUT_DIR}" \
  --cache-dir cache/ampc/exact_dkl_top_n \
  --eval-chunk-size 2000 \
  --score-chunk-size "${SCORE_CHUNK}" \
  --extra-set gentail \
  --predict-only ampc_331k \
  --wandb-project ampc-Variational-GP \
  --wandb-entity models-mila5723 \
  --wandb-group pll-noise-floor \
  --wandb-tags "variational-gp,svgp,M${M},pll-noise-floor,${SET_LABEL},n${SIZE},${OBJECTIVE_ARG},${OBJECTIVE},nfloor${NOISE_FLOOR},init-${INIT},ard-${ARD},matern,steps${TOTAL_STEPS},seed${SEED},fp32" \
  --run-name "${ARM//_/-}-${SLURM_JOB_ID}"

echo
echo "Done: ${OUTPUT_DIR}/eval_summary.json"
