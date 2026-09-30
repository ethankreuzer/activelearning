#!/bin/bash
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=48
# All memory on a 48-core A100 node (RealMemory=510000M). Narval rejects --mem=0
# unless the whole node (all 4 GPUs) is requested, and this code uses one GPU.
#SBATCH --mem=510000M
# 10 rounds of the single-fidelity loop: per-round S3-GFN training, 100k-candidate
# generation, and ~10k Dock3 evaluations. Give the wall clock plenty of headroom;
# the run is expected to finish its budget, not time out.
#SBATCH --time=72:00:00
#SBATCH --job-name=ampc_sf_reward_power
#SBATCH --account=def-yvesbrun_gpu
#SBATCH --output=slurm_logs/ampc_sf_reward_power_%j.out
#SBATCH --error=slurm_logs/ampc_sf_reward_power_%j.err
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
# Reward-shape arm: R = IG ** beta (beta=1), value-scale GIBBON. Pairs with
# job_sf_reward_exponential.sh, the same run under R = exp(beta * IG). The
# override file carries the reward-shape settings this arm needs
# (reward_transform, beta, acquisition.log_output=false, its own logger names
# and run_writer.output_dir); see config/ampc/overrides/reward_power.yaml for
# the reasoning. The dotlist args below are this experiment's own requirements
# on top of that:
#   - budget.max_rounds=1        a single active-learning round
#   - sampler.n_train_steps=10000  10k GFlowNet policy-training steps
#   - acquisition.log_space=true   underflow-safe GIBBON internals; value-scale
#       output is kept by log_output=false above, so LogSpaceQLowerBoundMaxValueEntropy
#       is used rather than the log-output variant (see botorch_entropy.py)
# (surrogate.variational_objective=PredictiveLogLikelihood and
# sampler.performance_mode=optimized already come from the base config.)
uv run --no-sync python scripts/run_with_sigterm_flush.py \
  config/ampc/s3gfn_minimol_ampc_variational_single_fidelity.yaml \
  config/ampc/overrides/reward_power.yaml \
  budget.max_rounds=1 \
  sampler.n_train_steps=10000 \
  acquisition.log_space=true
