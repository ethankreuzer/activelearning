#!/bin/bash
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=48
# Whole node worth of memory, as in jobs/surrogate_eval_fit.sh (10M rows peaked at
# ~478 GB on earlier runs).
#SBATCH --mem=510000M
# The fit time at M > 64 is not measured. The M = 64 job takes 19 min in total (11 min
# of fit, 50,000 steps). The per-step GP cost grows with M^2 (16x at 256, 256x at
# 1024), but how much of the M = 64 step is that cost and how much is fixed overhead
# is unknown, so this limit is sized for 1024. For 256, a shorter limit schedules
# sooner: sbatch --time=3:00:00 jobs/surrogate_eval_num_inducing.sh 256
#SBATCH --time=12:00:00
#SBATCH --job-name=ampc_num_inducing
#SBATCH --account=def-yvesbrun_gpu
#SBATCH --output=slurm_logs/ampc_num_inducing_%j.out
#SBATCH --error=slurm_logs/ampc_num_inducing_%j.err
#
# Step 9, experiment 1 of SURROGATE_EVAL_PLAN.md: the baseline ELBO fit
# (jobs/surrogate_eval_fit.sh VariationalELBO, M = 64) with only the number of
# inducing points changed, to see whether capacity limits the fit on the top molecules:
#
#   sbatch --time=3:00:00 jobs/surrogate_eval_num_inducing.sh 256
#   sbatch jobs/surrogate_eval_num_inducing.sh 1024
#
# Output goes to outputs/ampc/surrogate_eval/ELBO_num_inducing_<M>/. Arguments after
# M are passed through (e.g. --overwrite). Afterwards, from a login node:
#
#   wandb sync outputs/ampc/surrogate_eval/ELBO_num_inducing_<M>/wandb/offline-run-*
NUM_INDUCING="${1:?usage: sbatch jobs/surrogate_eval_num_inducing.sh <num_inducing>}"
case "${NUM_INDUCING}" in
  ''|*[!0-9]*|0*) echo "num_inducing must be a positive integer, got '${NUM_INDUCING}'" >&2; exit 2 ;;
esac

cd /home/ethankrz/activelearning

source .venv/bin/activate

export PYTHONUNBUFFERED=1
# Compute nodes are offline.
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline

OBJECTIVE=VariationalELBO
OUTPUT_DIR="outputs/ampc/surrogate_eval/ELBO_num_inducing_${NUM_INDUCING}"
export WANDB_DIR="${OUTPUT_DIR}"
mkdir -p "${WANDB_DIR}"

# Everything but surrogate.num_inducing matches jobs/surrogate_eval_fit.sh. The
# inducing init is pinned to the baseline's random one (stratified is experiment 2).
uv run --no-sync python -m scripts.surrogate_eval_fit \
  config/ampc/s3gfn_minimol_ampc_variational_single_fidelity.yaml \
  "surrogate.training_params.variational_objective=${OBJECTIVE}" \
  "surrogate.num_inducing=${NUM_INDUCING}" \
  surrogate.inducing_init=random \
  acquisition.log_space=true \
  acquisition.log_output=false \
  --output-dir "${OUTPUT_DIR}" \
  --val-csv data/ampc_val_20k.csv \
  --gp-molformer-csv data/gpmolformer_prior_100k_docked.csv \
  --reward-transform exponential \
  --reward-beta 100 \
  --wandb-project ampc-surrogate-eval \
  --wandb-entity models-mila5723 \
  --wandb-group surrogate-eval-step9 \
  --wandb-tags "surrogate-eval,${OBJECTIVE},num_inducing_${NUM_INDUCING}" \
  --run-name "ELBO-M${NUM_INDUCING}-${SLURM_JOB_ID}" \
  "${@:2}"
