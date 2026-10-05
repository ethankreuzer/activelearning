#!/bin/bash
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=48
# Whole node worth of memory, as in jobs/surrogate_eval_fit.sh (10M rows peaked at
# ~478 GB on earlier runs).
#SBATCH --mem=510000M
# M = 256 fit took 851 s (job 4532636); 3 h leaves ample room for the evaluation.
#SBATCH --time=3:00:00
#SBATCH --job-name=ampc_target_transform
#SBATCH --account=def-yvesbrun_gpu
#SBATCH --output=slurm_logs/ampc_target_transform_%j.out
#SBATCH --error=slurm_logs/ampc_target_transform_%j.err
#
# Step 9, experiment 3 of SURROGATE_EVAL_PLAN.md: the baseline ELBO fit with the
# training targets transformed, to see whether the top molecules sitting ~30 standard
# deviations out is what the Gaussian fit cannot reach. M is pinned to 256 (experiment
# 1 showed M changes little, and 256 costs 851 s against 733 s at 64):
#
#   sbatch jobs/surrogate_eval_target_transform.sh log
#   sbatch jobs/surrogate_eval_target_transform.sh logit
#
# Metrics stay on the original y scale, so they compare directly with the M = 256
# run (W&B 0cne37bg). The acquisition, however, sees the transformed scale: its
# scores are not comparable with the untransformed runs.
#
# Output goes to outputs/ampc/surrogate_eval/ELBO_target_<transform>/. Arguments after
# the transform are passed through (e.g. --overwrite). Afterwards, from a login node:
#
#   wandb sync outputs/ampc/surrogate_eval/ELBO_target_<transform>/wandb/offline-run-*
TRANSFORM="${1:?usage: sbatch jobs/surrogate_eval_target_transform.sh <log|logit>}"
case "${TRANSFORM}" in
  log|logit) ;;
  *) echo "transform must be 'log' or 'logit', got '${TRANSFORM}'" >&2; exit 2 ;;
esac

cd /home/ethankrz/activelearning

source .venv/bin/activate

export PYTHONUNBUFFERED=1
# Compute nodes are offline.
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline

OBJECTIVE=VariationalELBO
NUM_INDUCING=256
OUTPUT_DIR="outputs/ampc/surrogate_eval/ELBO_target_${TRANSFORM}"
export WANDB_DIR="${OUTPUT_DIR}"
mkdir -p "${WANDB_DIR}"

# Everything but --target-transform matches jobs/surrogate_eval_num_inducing.sh 256.
uv run --no-sync python -m scripts.surrogate_eval_fit \
  config/ampc/s3gfn_minimol_ampc_variational_single_fidelity.yaml \
  "surrogate.training_params.variational_objective=${OBJECTIVE}" \
  "surrogate.num_inducing=${NUM_INDUCING}" \
  surrogate.inducing_init=random \
  acquisition.log_space=true \
  acquisition.log_output=false \
  --output-dir "${OUTPUT_DIR}" \
  --target-transform "${TRANSFORM}" \
  --val-csv data/ampc_val_20k.csv \
  --gp-molformer-csv data/gpmolformer_prior_100k_docked.csv \
  --reward-transform exponential \
  --reward-beta 100 \
  --wandb-project ampc-surrogate-eval \
  --wandb-entity models-mila5723 \
  --wandb-group surrogate-eval-step9 \
  --wandb-tags "surrogate-eval,${OBJECTIVE},target_${TRANSFORM},num_inducing_${NUM_INDUCING}" \
  --run-name "ELBO-${TRANSFORM}-M${NUM_INDUCING}-${SLURM_JOB_ID}" \
  "${@:2}"
