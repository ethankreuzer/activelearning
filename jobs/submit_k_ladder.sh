#!/bin/bash
# Submit the K ladder: 3 surrogates x 5 K x 2 seeds = 30 arms of jobs/k_ladder_mila.sh.
#
#   bash jobs/submit_k_ladder.sh             # print the 30 sbatch commands, submit nothing
#   SUBMIT=1 bash jobs/submit_k_ladder.sh    # submit them
#
# Restrict it with SURROGATES, KS and SEEDS (space-separated), e.g. to run one rung or
# one surrogate first:
#
#   SURROGATES="pll" KS="100" SEEDS="42" SUBMIT=1 bash jobs/submit_k_ladder.sh
#
# ORDER. The seed is the OUTER loop, so the first 15 arms to start are seed 42 on every
# surrogate and every K, and a whole seed's picture exists before the second seed
# begins. Mila's quota on main is 2 GPUs, so the 30 run two at a time. An arm whose
# generated pool already exists is skipped, so the script can be re-run after a failure.
set -euo pipefail

SURROGATES="${SURROGATES:-exactgp elbo pll}"
KS="${KS:-10 30 100 300 1000}"
SEEDS="${SEEDS:-42 43}"

cd "${HOME}/activelearning"
submitted=0
skipped=0
for seed in ${SEEDS}; do
  for surrogate in ${SURROGATES}; do
    for k in ${KS}; do
      arm="kladder_${surrogate}_K${k}_s${seed}_steps5000"
      if compgen -G "outputs/ampc/k_ladder/${arm}/generated_pool_round_*.csv" > /dev/null; then
        echo "skip   ${arm} (generated pool already exists)"
        skipped=$((skipped + 1))
        continue
      fi
      cmd=(sbatch "--job-name=kl_${surrogate}_K${k}_s${seed}" jobs/k_ladder_mila.sh "${surrogate}" "${k}" "${seed}")
      if [[ "${SUBMIT:-0}" == "1" ]]; then
        "${cmd[@]}"
      else
        echo "${cmd[*]}"
      fi
      submitted=$((submitted + 1))
    done
  done
done
verb="would submit"
[[ "${SUBMIT:-0}" == "1" ]] && verb="submitted"
echo "${verb} ${submitted}, skipped ${skipped}"
