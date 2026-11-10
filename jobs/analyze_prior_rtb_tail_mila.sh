#!/bin/bash
# Re-read the prior-RTB runs as a tail problem. CPU only -- no model is loaded
# and no GIBBON is recomputed; this reads the per-molecule log GIBBON scores
# already written to each run's policy_sample.csv.
#
#   sbatch jobs/analyze_prior_rtb_tail_mila.sh
#
# WHY A JOB AND NOT A LOGIN SHELL. The bootstrap at n = 65,536 is small if
# chunked and 2.1 GB if not, and an interactive mila-code session is allocated
# 1 CPU / 2 GB -- enough that a careless array is OOM-killed (exit 137) with no
# other symptom. 16 G here is slack, not need.
#
# WHAT IT ANSWERS. The sweep ranked trials by log_gibbon_p90, which lives in the
# bulk. The downstream requirement is tail mass: S3-GFN draws 64 at a time, so
# the acquisition reward stops being silent once ~1/64 = 1.56% of samples clear
# -744.44. This prints both, so a trial that fattens the tail without moving the
# bulk is visible rather than averaged away.
#SBATCH --partition=main-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --job-name=prior_rtb_tail
#SBATCH --output=slurm_logs/prior_rtb_tail_%j.out
#SBATCH --error=slurm_logs/prior_rtb_tail_%j.err

set -euo pipefail

cd "${HOME}/activelearning"
source .venv/bin/activate
export PYTHONUNBUFFERED=1

BASELINE="${BASELINE:-outputs/ampc/prior_rtb/prior_rtb_beta0_steps0_onpolicy0.0_s42_svgp300k_n65536}"
OUT_DIR="outputs/ampc/prior_rtb/tail_analysis"

if [[ ! -e "${BASELINE}/policy_sample.csv" ]]; then
  echo "Missing ${BASELINE}/policy_sample.csv." >&2
  echo "Run the control arm first:" >&2
  echo "  ARM_SUFFIX=_n65536 sbatch jobs/pretrain_prior_rtb_mila.sh 0 0 0.0 42 \\" >&2
  echo "    --gibbon-config config/ampc/reward_shape_svgp300k.yaml \\" >&2
  echo "    --skip-policy-save --eval-sample-size 65536" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}" slurm_logs

# Every sweep trial plus the 2048 control, so the two controls can be read
# against each other as a sanity check on the sample size itself.
mapfile -t TRIALS < <(
  ls -d outputs/ampc/prior_rtb/sweep/trial_* \
         outputs/ampc/prior_rtb/prior_rtb_beta0_steps0_onpolicy0.0_s42_svgp300k \
    2>/dev/null
)
if (( ${#TRIALS[@]} == 0 )); then
  echo "No trial directories found under outputs/ampc/prior_rtb/sweep." >&2
  exit 2
fi

echo "Baseline: ${BASELINE}"
echo "Comparing ${#TRIALS[@]} directories."

python scripts/analyze_prior_rtb_tail.py \
  --baseline "${BASELINE}" \
  --trials "${TRIALS[@]}" \
  --out "${OUT_DIR}/tail_report.json" \
  "$@"
