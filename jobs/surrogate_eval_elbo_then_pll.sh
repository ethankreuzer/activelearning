#!/bin/bash
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=48
# All memory on a 48-core A100 node (RealMemory=510000M). The 10M initial dataset
# peaked at ~478 GB on earlier runs, and Narval rejects --mem=0 unless the whole
# node (all 4 GPUs) is requested, and this code uses one GPU.
#SBATCH --mem=510000M
# The 2026-10-01 fit-and-evaluate jobs took about 20 min. This one also encodes the
# generated set before the fit and scores it every epoch, so 2 h until measured.
#SBATCH --time=2:00:00
#SBATCH --job-name=ampc_elbo_then_pll
#SBATCH --account=def-yvesbrun_gpu
#SBATCH --output=slurm_logs/ampc_elbo_then_pll_%j.out
#SBATCH --error=slurm_logs/ampc_elbo_then_pll_%j.err
#
# Step 7 of SURROGATE_EVAL_PLAN.md: load the surrogate fitted with VariationalELBO and
# continue training it with PredictiveLogLikelihood, to keep ELBO's means and get
# PLL's usable latent standard deviations. The argument picks what the PLL phase may
# train:
#
#   sbatch jobs/surrogate_eval_elbo_then_pll.sh all        # every parameter
#   sbatch jobs/surrogate_eval_elbo_then_pll.sh variance   # inducing covariance and
#                                                          # noise only: mean frozen
#
# The PLL phase uses the base config's training settings (50 epochs, lr 1e-3, batch
# 10000). Arguments after the mode are passed through to the script, for example
#
#   sbatch jobs/surrogate_eval_elbo_then_pll.sh variance surrogate.training_params.epochs=100
#   sbatch jobs/surrogate_eval_elbo_then_pll.sh all --overwrite
#
# The ELBO state and its epoch count come from the environment when they differ from
# the defaults:
#
#   INIT_STATE=path/to/surrogate_state.pt EPOCH_OFFSET=50 sbatch jobs/...sh variance
#
# Output goes to outputs/ampc/surrogate_eval/ELBO_then_PLL_<mode>/. After the job,
# sync the offline run from a login node:
#
#   wandb sync outputs/ampc/surrogate_eval/ELBO_then_PLL_<mode>/wandb/offline-run-*
MODE="${1:?usage: sbatch jobs/surrogate_eval_elbo_then_pll.sh <all|variance> [extra args]}"
INIT_STATE="${INIT_STATE:-outputs/ampc/surrogate_eval/VariationalELBO/surrogate_state.pt}"
# Epochs of the ELBO run the state came from: its last step is EPOCH_OFFSET - 1, where
# this run logs its starting point, and the PLL epochs follow from EPOCH_OFFSET.
EPOCH_OFFSET="${EPOCH_OFFSET:-50}"

cd /home/ethankrz/activelearning

source .venv/bin/activate

# Python block-buffers stdout when it is redirected to a file; flush progress
# lines as they happen instead.
export PYTHONUNBUFFERED=1

# Compute nodes are offline: sync from a login node first and use --no-sync, log
# W&B offline and sync it afterwards (`wandb sync <run dir>`), and keep
# transformers/HF from stalling on hub checks.
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline

# Per-arm directories keep the two modes from racing on wandb's latest-run symlink
# and make `wandb sync` unambiguous.
OUTPUT_DIR="outputs/ampc/surrogate_eval/ELBO_then_PLL_${MODE}"
export WANDB_DIR="${OUTPUT_DIR}"
mkdir -p "${WANDB_DIR}"

# log_space=true/log_output=false: value-scale GIBBON information gain computed
# without underflow, as in jobs/surrogate_eval_fit.sh.
# Run as a module so the repo root is on sys.path: the script imports
# scripts.surrogate_eval_metrics, which `python scripts/...py` can't resolve.
uv run --no-sync python -m scripts.surrogate_eval_fit \
  config/ampc/s3gfn_minimol_ampc_variational_single_fidelity.yaml \
  surrogate.training_params.variational_objective=PredictiveLogLikelihood \
  acquisition.log_space=true \
  acquisition.log_output=false \
  --output-dir "${OUTPUT_DIR}" \
  --init-state "${INIT_STATE}" \
  --trainable "${MODE}" \
  --epoch-offset "${EPOCH_OFFSET}" \
  --val-csv data/ampc_val_20k.csv \
  --gp-molformer-csv data/gpmolformer_prior_100k_docked.csv \
  --reward-transform exponential \
  --reward-beta 100 \
  --wandb-project ampc-surrogate-eval \
  --wandb-entity models-mila5723 \
  --wandb-group surrogate-eval-step7 \
  --wandb-tags "surrogate-eval,ELBO_then_PLL,${MODE}" \
  --run-name "ELBO_then_PLL_${MODE}-${SLURM_JOB_ID}" \
  "${@:2}"
