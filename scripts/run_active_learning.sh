#!/usr/bin/env bash
# Usage: sbatch scripts/run_active_learning.sh <config.yaml> [key=value ...]
#SBATCH --job-name=active_learning
#SBATCH --output=active_learning_%j.out
#SBATCH --error=active_learning_%j.err
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --mem=32GB
#SBATCH --partition=main

set -euo pipefail

if [[ -n "${REPO_ROOT:-}" ]]; then
  REPO_ROOT="$(cd -- "${REPO_ROOT}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="$(cd -- "${SLURM_SUBMIT_DIR}" && pwd)"
else
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "${REPO_ROOT}"

if [[ $# -eq 0 ]]; then
  echo "Usage: sbatch scripts/run_active_learning.sh <config.yaml> [key=value ...]" >&2
  exit 1
fi

CONFIG_PATH="$1"
shift

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config file not found: ${CONFIG_PATH}" >&2
  exit 1
fi



# Or prepend an installation directory:
export PATH="/path/to/xtb/bin:${PATH}"

if ! command -v xtb >/dev/null 2>&1; then
  echo "xtb not found on PATH" >&2
  exit 1
fi


export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export PYTHONUNBUFFERED=1

echo "Running active learning"
echo "  host: ${HOSTNAME:-unknown}"
echo "  job: ${SLURM_JOB_ID:-unknown}"
echo "  config: ${CONFIG_PATH}"

uv run --all-extras activelearning "${CONFIG_PATH}" "$@"
