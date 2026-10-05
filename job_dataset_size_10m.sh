#!/bin/bash
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=48
# All memory on a 48-core A100 node (RealMemory=510000M). The 10M initial dataset
# peaked at ~478 GB on earlier runs, and Narval rejects --mem=0 unless the whole
# node (all 4 GPUs) is requested, and this code uses one GPU.
#SBATCH --mem=510000M
# The persistent feature cache already covers the 10M training set, so the fit
# does no encoding, but the variational GP still trains on 10M rows. Narval's b4
# bucket covers 24 h to 72 h, so 72 h queues the same as 36 h.
#SBATCH --time=2:00:00
#SBATCH --job-name=ampc_size_study_10m
#SBATCH --account=def-yvesbrun_gpu
#SBATCH --output=slurm_logs/ampc_size_study_10m_%j.out
#SBATCH --error=slurm_logs/ampc_size_study_10m_%j.err
cd /home/ethankrz/activelearning

source .venv/bin/activate

# Python block-buffers stdout when it is redirected to a file; flush progress
# lines as they happen instead.
export PYTHONUNBUFFERED=1

# Compute nodes are offline; keep transformers/HF from stalling on hub checks.
export HF_HUB_OFFLINE=1

# Fit the surrogate on the 10M subset (the base config's initial dataset), then
# score the 331k AmpC subset and the Olivier in-vitro set. Outputs:
# outputs/ampc/dataset_size_study/10m/, plus surrogate_state.pt saved right
# after the fit. The evaluation SMILES don't match the feature cache's prefix,
# so they are encoded live and the cache is left untouched.
# Compute nodes are offline: sync from a login node first and use --no-sync.
# log_space (not log_output) per today's colleague-provided config: keeps
# scores on the value scale (unlike the base config's log_output: true) while
# still avoiding the exact-zero underflow at 10M candidates.
uv run --no-sync python scripts/surrogate_dataset_size_study.py \
  config/ampc/s3gfn_minimol_ampc_variational_single_fidelity.yaml \
  acquisition.type=QLowerBoundMaxValueEntropy \
  acquisition.log_space=true \
  acquisition.log_output=true \
  --train-label 10m \
  --output-dir outputs/ampc/dataset_size_study_log_output \
  --eval-csv ampc_331k=data/ampc_subset_331k.csv \
  --eval-csv olivier_invitro=data/Olivier_Invitro.csv
