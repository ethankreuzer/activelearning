#!/bin/bash
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=48
# All memory on a 48-core A100 node (RealMemory=510000M). Narval rejects --mem=0
# unless the whole node (all 4 GPUs) is requested, and this code uses one GPU.
#SBATCH --mem=510000M
# 10 rounds of the single-fidelity loop: per-round S3-GFN training (10k steps),
# 100k-candidate generation, and ~10k Dock3 evaluations. Give the wall clock
# plenty of headroom; the run is expected to finish its budget, not time out.
#SBATCH --time=72:00:00
#SBATCH --job-name=ampc_sf_qlbmes_10rounds
#SBATCH --account=def-yvesbrun_gpu
#SBATCH --output=slurm_logs/ampc_sf_qlbmes_10rounds_%j.out
#SBATCH --error=slurm_logs/ampc_sf_qlbmes_10rounds_%j.err
cd /home/ethankrz/activelearning

source .venv/bin/activate

# wandb.init() defaults to online mode and blocks on api.wandb.ai, which this
# node cannot reach - measured hanging >9min, ignoring WANDB_INIT_TIMEOUT.
# Offline mode writes to ./wandb/offline-run-*; sync from a login node afterwards:
#   wandb sync wandb/offline-run-*
export WANDB_MODE=offline

# The S3GFN sampler loads GP-MoLFormer from the HuggingFace cache in $HOME.
# Pre-fetch it from a login node; offline mode stops transformers from stalling
# on hub metadata checks the compute node cannot make.
export HF_HUB_OFFLINE=1

# Narval compute nodes have no outbound internet, so the environment must already
# be synced from a login node (`uv sync --all-extras`) before submitting this job.
# --no-sync keeps `uv run` from re-resolving dependencies over the network.
#
# Slurm sends SIGTERM at the wall clock (SIGKILL 60s later); the wrapper turns
# that into a clean shutdown so buffered wandb/run-writer output is still flushed.
#
# This run is single fidelity with the QLowerBoundMaxValueEntropy (GIBBON)
# acquisition computed in log space but returned as plain information gain. The
# overrides below make each requested setting explicit even though the base
# config already carries some of them:
#   - surrogate.variational_objective=PredictiveLogLikelihood
#                                                   larger acquisitions than ELBO
#   - acquisition.type=QLowerBoundMaxValueEntropy   single-fidelity GIBBON
#   - acquisition.log_space=true                    underflow-safe computation
#   - acquisition.log_output=false                  scores are IG, not log(IG)
#   - sampler.beta=400                              log R = 400 * IG; sized from
#       outputs/ampc/dataset_size_study_pll_log_space/10m (p99.9 - median ~15 nats)
#   - sampler.performance_mode=optimized            S3-GFN model optimizations on
#   - sampler.n_train_steps=10000                   10k policy-training steps/round
#   - sampler.n_samples=100000                      generate a 100k pool per round
#   - budget 100000 / schedule 10000                10 active-learning rounds
uv run --no-sync python scripts/run_with_sigterm_flush.py \
  config/ampc/s3gfn_minimol_ampc_variational_single_fidelity.yaml \
  surrogate.variational_objective=PredictiveLogLikelihood \
  acquisition.type=QLowerBoundMaxValueEntropy \
  acquisition.log_space=true \
  acquisition.log_output=false \
  sampler.beta=400 \
  sampler.performance_mode=optimized \
  sampler.n_train_steps=10000 \
  sampler.n_samples=100000 \
  budget.available_budget=100000.0 \
  budget.schedule.value=10000.0
