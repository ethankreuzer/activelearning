#!/bin/bash
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=48
# Whole node worth of memory, as in jobs/surrogate_eval_fit.sh (10M rows peaked at
# ~478 GB on earlier runs).
#SBATCH --mem=510000M
#SBATCH --time=2:00:00
#SBATCH --job-name=ampc_inducing_init
#SBATCH --account=def-yvesbrun_gpu
#SBATCH --output=slurm_logs/ampc_inducing_init_%j.out
#SBATCH --error=slurm_logs/ampc_inducing_init_%j.err
#
# Step 8 of SURROGATE_EVAL_PLAN.md: the baseline ELBO fit (jobs/surrogate_eval_fit.sh
# VariationalELBO) with only the starting inducing points changed. Both modes split
# the 10M rows into target-quantile strata (bottom 90%, 90-99%, 99-99.9%, top 0.1%)
# and give them 24 / 16 / 12 / 12 of the 64 points:
#
#   sbatch jobs/surrogate_eval_inducing_init.sh stratified   # random rows per stratum
#   sbatch jobs/surrogate_eval_inducing_init.sh kmeans       # k-means centres per stratum
#
# Output goes to outputs/ampc/surrogate_eval/ELBO_inducing_<mode>/. Arguments after
# the mode are passed through (e.g. --overwrite). Afterwards, from a login node:
#
#   wandb sync outputs/ampc/surrogate_eval/ELBO_inducing_<mode>/wandb/offline-run-*
MODE="${1:?usage: sbatch jobs/surrogate_eval_inducing_init.sh <stratified|kmeans>}"
case "${MODE}" in
  stratified|kmeans) ;;
  *) echo "mode must be stratified or kmeans, got '${MODE}'" >&2; exit 2 ;;
esac

cd /home/ethankrz/activelearning

source .venv/bin/activate

export PYTHONUNBUFFERED=1
# Compute nodes are offline.
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline

OBJECTIVE=VariationalELBO
OUTPUT_DIR="outputs/ampc/surrogate_eval/ELBO_inducing_${MODE}"
export WANDB_DIR="${OUTPUT_DIR}"
mkdir -p "${WANDB_DIR}"

# Everything but surrogate.inducing_init matches jobs/surrogate_eval_fit.sh.
uv run --no-sync python -m scripts.surrogate_eval_fit \
  config/ampc/s3gfn_minimol_ampc_variational_single_fidelity.yaml \
  "surrogate.training_params.variational_objective=${OBJECTIVE}" \
  "surrogate.inducing_init=${MODE}" \
  acquisition.log_space=true \
  acquisition.log_output=false \
  --output-dir "${OUTPUT_DIR}" \
  --val-csv data/ampc_val_20k.csv \
  --gp-molformer-csv data/gpmolformer_prior_100k_docked.csv \
  --reward-transform exponential \
  --reward-beta 100 \
  --wandb-project ampc-surrogate-eval \
  --wandb-entity models-mila5723 \
  --wandb-group surrogate-eval-step8 \
  --wandb-tags "surrogate-eval,${OBJECTIVE},inducing_${MODE}" \
  --run-name "ELBO-inducing-${MODE}-${SLURM_JOB_ID}" \
  "${@:2}"
