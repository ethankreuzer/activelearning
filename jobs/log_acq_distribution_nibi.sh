#!/bin/bash
# Measure the log GIBBON distribution over the 300,000 molecules the svgp300k
# surrogate was fitted to.
#
#   sbatch jobs/log_acq_distribution_nibi.sh
#
# WHY. Every arm in ampc-reward-shape-svgp picks log_score_floor against a
# distribution measured on a DIFFERENT set: the 83,811 molecules of
# gp_molformer_set (REWARD_SHAPE_UNDERFLOW_GAP.md section 3 -- median -2,688,
# p99.9 -752.6, max -158.9). The floor, the underflow edge at -744.4 and any
# tail_threshold a seeded replay buffer would use are all quantiles of this
# distribution, so measure it on the set the surrogate actually saw.
#
# This run SCORES ONLY. No policy is trained, no molecule is generated and no
# oracle is called; the sampler, selector, oracle and budget sections of the
# config are never built.
#
# log_output=true IS THE POINT, and is the one way this differs from an arm. The
# arms run log_output=false, which returns exp(L): in float64 that is exactly
# 0.0 below ln(5e-324) = -744.44, so about 77% of the set arrives as a single
# tied value and the bulk cannot be measured. log_output=true returns L itself.
# The scores are therefore on the same scale and from the same acquisition as
# the arms', just not pushed through an exponential that cannot hold them.
#
# One H100. The restore and the encoder are GPU work; the 300,000 feature rows
# come from feature_cache_path, so MiniMol is never run and nothing here uses
# CPU parallelism. 4 cores and 32G serve host-side torch and the output frame,
# matching jobs/reward_shape_nibi_nodock.sh.
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
# The docked Nibi arms of 2026-10-27 ranked a 100,000-molecule pool by this same
# acquisition in 3.3 min, so 300,000 is ~10 min, plus the surrogate restore and
# acquisition update (under a minute) and one chunked posterior pass for the
# latent mean and std. 1 h is generous for ~15 min of work and fits a backfill
# window. The script prints a rate and an ETA every 50 chunks, so the first run
# measures this properly; and because it scores in a seeded shuffled order and
# flushes per chunk, a job killed at the wall clock still leaves a uniform
# random subsample on disk rather than a biased prefix.
#SBATCH --time=1:00:00
#SBATCH --job-name=ampc_log_acq_distribution
# Re-check before every submit: `sshare -U -u $USER`.
#SBATCH --account=def-yvesbrun_gpu
#SBATCH --output=slurm_logs/ampc_log_acq_distribution_%j.out
#SBATCH --error=slurm_logs/ampc_log_acq_distribution_%j.err

set -euo pipefail

BASE_CONFIG="config/ampc/reward_shape_svgp300k.yaml"
FITTED_DIR="outputs/ampc/variational_gp/svgpscale_300000_svgp_M1024_iso_matern_scaled_pm0.03956_elbo_cosine_lr3e-3_steps10064_fp32"
STATE_PATH="${FITTED_DIR}/surrogate_state.pt"
OUT_DIR="outputs/ampc/log_acq_distribution"
# Optional suffix for a rerun, so it gets its own file instead of colliding
# with the earlier scoring.
OUT_SUFFIX="${OUT_SUFFIX:-}"
if [[ -n "${OUT_SUFFIX}" && ! "${OUT_SUFFIX}" =~ ^_[A-Za-z0-9.-]+$ ]]; then
  echo "OUT_SUFFIX must look like _name (letters, digits, '.', '-')." >&2
  exit 2
fi
# Named by cluster: the Mila/Nibi split of 2026-10-28 is still unexplained, and
# two scorings of the same state file are the cheapest test of whether it lives
# in the surrogate restore or the acquisition. Diff against the Mila summary.
OUT_CSV="${OUT_DIR}/svgp300k_nibi${OUT_SUFFIX}.csv"

cd "${HOME}/activelearning"
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1   # nothing is downloaded: the encoder is cached

for required in \
  "${STATE_PATH}" \
  "data/ampc_svgpscale_300000.csv" \
  "cache/ampc/exact_dkl_top_n/svgpscale_300000.npy" \
  "minimol_ampc_encoder/model/final.pt"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing ${required}." >&2
    echo "This job restores a surrogate rather than fitting one, so the fitted" >&2
    echo "state, its training set, its feature cache and the MiniMol encoder" >&2
    echo "must all be present." >&2
    exit 2
  fi
done
# The CSV is the result and is expensive to recompute; never overwrite one.
if [[ -e "${OUT_CSV}" ]]; then
  echo "${OUT_CSV} already exists." >&2
  echo "Move it aside, or set OUT_SUFFIX to write beside it:" >&2
  echo "  OUT_SUFFIX=_rerun2 sbatch jobs/log_acq_distribution_nibi.sh" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}" slurm_logs

uv run --no-sync python scripts/log_acq_distribution.py \
  "${BASE_CONFIG}" \
  "acquisition.log_output=true" \
  --out "${OUT_CSV}" \
  "$@"

echo
echo "Done. Plot with, on a login node:"
echo "  uv run --no-sync python -m scripts.plot_log_acq_distribution ${OUT_CSV}"
