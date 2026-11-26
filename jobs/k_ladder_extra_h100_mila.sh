#!/bin/bash
# K=200 and K=250 for ExactGP and ELBO, two seeds: 8 more arms of the K ladder
# (W&B thesis-ampc-k-ladder), run in ONE short-unkillable allocation on an H100 node.
#
#   sbatch jobs/k_ladder_extra_h100_mila.sh
#
# short-unkillable requires 4 GPUs per job and allows 3 h. The H100s are cn-n001/cn-n002
# (8 per node, only 4 usable per job), so the GPU type must be asked for by name.
# The 8 arms start together, PER_GPU=2 to a card, each pinned to one GPU with
# CUDA_VISIBLE_DEVICES, and the job waits for all of them. Each arm is exactly
# `bash jobs/k_ladder_mila.sh <surrogate> <K> <seed>`: same settings, names, W&B project
# and outputs as the other 30 arms. Each arm's console output goes to its own file,
# slurm_logs/k_ladder_extra_<surrogate>_K<K>_s<seed>_<jobid>.log.
#
# TIME IS THE RISK. On A100/L40S an arm took 1.6-2.0 h alone. Two arms sharing a card will
# each be slower than that, and the 3 h limit kills whatever is unfinished with no partial
# result (the pool is only written at the end). An ExactGP arm that hits its generation cap
# costs ~2.9 h alone and cannot finish here.
#SBATCH --partition=short-unkillable
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=3:00:00
#SBATCH --job-name=kl_extra_K200_K250
#SBATCH --output=slurm_logs/k_ladder_extra_%j.out
#SBATCH --error=slurm_logs/k_ladder_extra_%j.err

set -uo pipefail
cd "${HOME}/activelearning"
mkdir -p slurm_logs
nvidia-smi --query-gpu=index,name,memory.total --format=csv

SURROGATES="${SURROGATES:-exactgp elbo}"
KS="${KS:-200 250}"
SEEDS="${SEEDS:-42 43}"
N_GPUS=4

i=0
pids=()
names=()
for seed in ${SEEDS}; do
  for surrogate in ${SURROGATES}; do
    for k in ${KS}; do
      gpu=$(( i % N_GPUS ))
      log="slurm_logs/k_ladder_extra_${surrogate}_K${k}_s${seed}_${SLURM_JOB_ID:-local}.log"
      echo "launch ${surrogate} K=${k} seed=${seed} on GPU ${gpu} -> ${log}"
      CUDA_VISIBLE_DEVICES="${gpu}" bash jobs/k_ladder_mila.sh "${surrogate}" "${k}" "${seed}" > "${log}" 2>&1 &
      pids+=($!)
      names+=("${surrogate}_K${k}_s${seed}")
      i=$(( i + 1 ))
    done
  done
done

status=0
for idx in "${!pids[@]}"; do
  if wait "${pids[$idx]}"; then
    echo "done   ${names[$idx]}"
  else
    echo "FAILED ${names[$idx]} (exit $?)"
    status=1
  fi
done
exit "${status}"
