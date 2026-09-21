#!/bin/bash
#SBATCH --gres=gpu:a100:1
# 48 CPUs so the Dock3 thread pool (sized from the job's CPUs) matches the earlier
# 10M runs: ~5,900 s per 10k dockings.
#SBATCH --cpus-per-task=48
# All memory on a 48-core A100 node (RealMemory=510000M). The 10M initial dataset
# peaked at ~478 GB on earlier runs, and Narval rejects --mem=0 unless the whole
# node (all 4 GPUs) is requested, and this code uses one GPU.
#SBATCH --mem=510000M
# One round with 10000 S3-GFN training steps on the 10M initial dataset. The
# persistent feature cache is already complete, so there is no encoding pass, but
# the variational GP still fits on 10M rows. The eager/float32 step time (~6.6 s,
# job 3461294) is an upper bound for the optimized preset, which has not been
# measured here, so this asks for headroom rather than an estimate.
#
# Narval queue buckets are what the walltime actually selects: b4 covers 24 h to
# 72 h, so 72 h queues the same as 36 h and is the most headroom available.
#SBATCH --time=72:00:00
#SBATCH --job-name=ampc_10m_optimized
#SBATCH --account=def-yvesbrun_gpu
#SBATCH --output=slurm_logs/ampc_10m_optimized_%j.out
#SBATCH --error=slurm_logs/ampc_10m_optimized_%j.err
cd /home/ethankrz/activelearning

source .venv/bin/activate

# wandb.init() blocks on api.wandb.ai, which compute nodes cannot reach. Offline
# runs go to ./wandb/offline-run-*; sync from a login node afterwards:
#   wandb sync wandb/offline-run-*
export WANDB_MODE=offline

# Python block-buffers stdout when it is redirected to a file, which left the
# .out file empty for the whole of job 3461294 and hid every
# "S3-GFN training step N/10000" progress line. Flush it line by line instead.
export PYTHONUNBUFFERED=1

# The S3GFN sampler loads GP-MoLFormer from the HuggingFace cache in $HOME.
# Pre-fetch it from a login node; offline mode stops transformers from stalling
# on hub metadata checks the compute node cannot make.
export HF_HUB_OFFLINE=1

RUN_NAME=molecules-s3gfn-minimol-ampc-fixed-feature-variational-gp-10M-optimized-10000steps-gibbon

# Compute nodes are offline: sync the environment from a login node first and
# keep `uv run` from re-resolving dependencies with --no-sync.
#
# The wrapper turns a wall-clock SIGTERM into a clean interpreter shutdown so a
# timeout still flushes the buffered wandb run.
#
# The single-fidelity base config already has the 10M initial dataset, its
# persistent feature cache and performance_mode=optimized, so no overlay is used.
# The key=value overrides below set the rest:
#   - the single-fidelity GIBBON acquisition (QLowerBoundMaxValueEntropy),
#   - 10000 training steps (base: 2500) and a single round,
#   - a run_writer output_dir of its own, since starting a run truncates
#     round_history.jsonl in the base config's directory (the 10M run's),
#   - logger names of their own. Lists are replaced on merge, so the whole
#     logger is restated as one value.
uv run --no-sync python scripts/run_with_sigterm_flush.py \
  config/ampc/s3gfn_minimol_ampc_variational_single_fidelity.yaml \
  sampler.performance_mode=optimized \
  sampler.n_train_steps=10000 \
  acquisition.type=QLowerBoundMaxValueEntropy \
  budget.max_rounds=1 \
  'run_writer.output_dir=outputs/ampc/10m_optimized_10000steps_gibbon/seed_${runtime.seed}' \
  "logger={type: MultiLogger, loggers: [{type: ConsoleLogger, project_name: ampc_sanity_check, run_name: ${RUN_NAME}}, {type: WandbLogger, project_name: ampc_sanity_check, run_name: ${RUN_NAME}-1-round}]}"
