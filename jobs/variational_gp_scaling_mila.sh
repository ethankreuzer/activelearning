#!/bin/bash
# Fit one arm of the sparse-variational-GP training-set scaling study, then
# evaluate and score it exactly as every other arm in ampc-Variational-GP.
#
#   sbatch jobs/variational_gp_scaling_mila.sh pll 50000
#   sbatch jobs/variational_gp_scaling_mila.sh pll 100000
#   sbatch jobs/variational_gp_scaling_mila.sh pll 300000
#   sbatch jobs/variational_gp_scaling_mila.sh pll 100000 --extra-set svgpscale_random_100000
#
# The Mila copy of jobs/variational_gp_scaling_nibi.sh: same arm, same command,
# only the Slurm request, the repo path and W&B's mode differ. Submit from the
# repo root, after `mkdir -p slurm_logs`, since Slurm opens the log files there
# before the script runs.
#
# The training sets come from jobs/build_svgp_scaling_sets_nibi.sh (built on
# Nibi, where the 10M fingerprint store lives, and copied here) and are
# nested: the 50k set is a subset of the 100k set, which is a subset of the 300k
# set, and all three hold the same 16,292 molecules above y 0.2617 (pProp 3.5).
# Only the bulk grows, so what the arms measure is how much bulk this surrogate
# absorbs at fixed tail supervision -- which is what decides whether the AL loop
# can drop the archive/train split that the exact GP's O(n^3) fit forced on it.
#
# What to read, in this order:
#   1. gentail/* -- the held-out tail. These molecules were docked by the
#      10-round run's policy and are NOT in the 10M library, so no arm trained
#      on them. r2, bias and coverage_2std here are the answer to "does it still
#      fit the high scorers".
#   2. gp_molformer_set/acquisition/gibbon_*/top1000_enrichment,
#      effective_support and fraction_at_floor -- "are the GIBBON values still
#      usable as a reward". ampc_331k is predicted but NOT scored here
#      (--predict-only below): its GIBBON pass was 73% of the 50k arm's 49
#      minutes on an RTX 8000, while predicting it takes seconds.
#      The support is capped at 100,000 (see the config): the whole training
#      set at 50k and 100k, a stratified 100,000 at 300k -- what each size would
#      get in an AL round. run/acquisition/*/acqf_support_size records it.
#   3. ampc_331k and gp_molformer_set coverage_2std and spearman_std_error, and
#      the *_unseen variants.
#   4. run/hyperparameters/* across arms. The balanced PLL arm collapsed its
#      noise onto the 1e-4 floor; whether more data pulls it off the floor is
#      the single most informative number here.
#
# Epochs are NOT comparable across arms: the fit is minibatched at 4096 rows, so
# an epoch is 13, 25 or 74 steps. Compare train/epoch/* against the step count.
#
# Resources. The fit is a 4096 x 1024 cross-covariance per step regardless of
# arm, so the fit cost scales only with the step count. The memory and wall time
# are dominated by GIBBON scoring, as they were for the balanced arm (peak 10.4
# GB allocated at a 25,000-row support); the support here is capped at 100,000,
# up to four times that, so --score-chunk-size stays at 256 and the first 300k arm
# measures the real wall time (stage_profile.csv) for later ones to trim.
#
# Any GPU with 32 GB or more is enough for that peak, but an untyped request
# lands on whichever card is free first, usually an RTX 8000. The fit runs in
# plain float32 (no TF32), where an L40S (~90 TFLOPS) is about five times an
# RTX 8000 and faster than an A100, and Mila has ~90 L40S nodes in both main
# and unkillable. Override with --gres=gpu:<type>:1 on the sbatch line.
# The balanced arms took 7-8 minutes at 3.5 GB host RSS on Nibi, so 32G and 3
# hours leave ample room and fit unkillable's 6 CPU / 32G / 1 GPU cap.
# unkillable because it is never preempted: a preempted arm would be requeued
# onto an output directory that already holds its state, which the script
# refuses to reuse.
#SBATCH --partition=unkillable
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --job-name=ampc_svgp_scale
#SBATCH --output=slurm_logs/ampc_svgp_scale_%j.out
#SBATCH --error=slurm_logs/ampc_svgp_scale_%j.err

set -euo pipefail

OBJECTIVE_ARG="${1:-}"
case "${OBJECTIVE_ARG}" in
  elbo)
    OBJECTIVE="VariationalELBO"
    OBJECTIVE_NOTE="the variational ELBO, sum_i E_q[log p(y_i | f_i)] - KL(q || p)"
    ;;
  pll)
    OBJECTIVE="PredictiveLogLikelihood"
    OBJECTIVE_NOTE="the predictive log likelihood, sum_i log E_q[p(y_i | f_i)] - KL(q || p) (Jankowiak et al., 2020), which won the preliminary comparison on the 25,000-molecule balanced set"
    ;;
  *)
    echo "usage: sbatch jobs/variational_gp_scaling_mila.sh elbo|pll <size> [extra script flags or overrides...]" >&2
    exit 2
    ;;
esac

SIZE="${2:-}"
if [[ ! "${SIZE}" =~ ^[0-9]+$ ]]; then
  echo "The second argument must be the arm's training-set size, e.g. 50000." >&2
  exit 2
fi

SET_LABEL="${SET_LABEL:-svgpscale}"
TRAIN_CSV="data/ampc_${SET_LABEL}_${SIZE}.csv"
TRAIN_CACHE="cache/ampc/exact_dkl_top_n/${SET_LABEL}_${SIZE}.npy"
GENTAIL_SET="${GENTAIL_SET:-gentail}"

cd "${HOME}/activelearning"

for required in "${TRAIN_CSV}" "${TRAIN_CACHE}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing ${required}. Build the sets on Nibi and copy them here:" >&2
    echo "  sbatch jobs/build_svgp_scaling_sets_nibi.sh" >&2
    exit 2
  fi
done

for argument in "${@:3}"; do
  if [[ "${argument}" == dataset.initial_data.path=* \
     || "${argument}" == surrogate.encoder.feature_cache_path=* ]]; then
    echo "Set the training set with the second argument (the size), not" \
      "${argument}, so the directory, tags and notes follow it." >&2
    exit 2
  fi
  if [[ "${argument}" == surrogate.training_params.variational_objective=* ]]; then
    echo "Set the objective with the first argument (elbo|pll), not ${argument}." >&2
    exit 2
  fi
done

ARM="${SET_LABEL}_${SIZE}_svgp_M1024_iso_matern_scaled_pm0.03956_${OBJECTIVE_ARG}_cosine_fp32"

source .venv/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1   # nothing is downloaded: every model input is a cached feature
# Online, unlike on Nibi: Mila compute nodes reach api.wandb.ai, so the run
# streams to the project as it goes and nothing needs syncing afterwards.

OUTPUT_DIR="outputs/ampc/variational_gp/${ARM}"
# Per-arm, so concurrent runs never race on a shared ./wandb/latest-run symlink.
export WANDB_DIR="${OUTPUT_DIR}"
mkdir -p "${WANDB_DIR}" slurm_logs

# Shown on the W&B run's Overview page and kept next to the outputs.
export WANDB_NOTES="Sparse variational GP with 1024 learned inducing points, fitted to the ${SIZE}-molecule arm of the training-set scaling study (${TRAIN_CSV}).
The arms are NESTED subsets of data/10M_unif_random_subset.csv built by scripts/build_svgp_scaling_sets.py: every arm holds the same 16292 molecules above y 0.2617 (pProp 3.5) whole, and the filler budget is split evenly across the four bands cut at pProp 2.0, 2.5, 3.0 and 3.5, by farthest point sampling in the frozen MiniMol space. Bands stop at pProp 2.0 because the shipped score/pProp table's eight-decimal cumul_prop column makes pProp constant at 1.5988 for every score from -7.045 to +71.665, which is the library's minimum and holds 101313 molecules; recovered DOCK scores for all 10M put the median at 1.6443 and the 99th percentile at 2.7299. Tail supervision is therefore constant across arms while bulk grows, so the arms isolate how much bulk the surrogate absorbs; the tail's share falls from 32.4% at 50k to 5.4% at 300k by construction.
Surrogate, unchanged across arms and taken from the winning balanced arm: Matern-5/2 with ONE lengthscale shared by all 512 features under BoTorch's dimension-scaled LogNormal prior and 0.025 floor; inputs min-max scaled to [0, 1] on the training rows' range; a learned ScaleKernel output scale; BoTorch's LogNormal(-4, 1) noise prior and 1e-4 floor; the constant mean PINNED at 0.03956042999405787 (the 10M library mean); float32. Both floors use gpytorch's softplus rather than BoTorch's untransformed bound, because Adam cannot enforce a bound.
Fitted on ${OBJECTIVE_NOTE}, by Adam on 4096-row minibatches for 300 epochs with the learning rate cosine-annealed from 1e-2 to 1e-4. Minibatched, unlike the balanced arm, so a step costs the same in every arm: an epoch is ceil(${SIZE}/4096) steps, so train/epoch/* is comparable across arms only against the step count.
GIBBON's max-value support is CAPPED at 100000 observations (acquisition.candidate_set_spec.max_support_size), as it would be in an AL round: the 50k and 100k arms use their whole training set, and the 300k arm a 100000-row subset chosen by the deterministic fidelity/target-stratified reservoir, which keeps the arm's best molecule. run/acquisition/*/acqf_support_size records the size used.
Evaluated on the held-out generated tail set ${GENTAIL_SET} -- molecules the 10-round run docked that are NOT in the 10M, so no arm trained on them -- alongside the standard sets. ampc_331k is predicted but not scored with GIBBON, so it has prediction metrics and no acquisition metrics and is not in the cross-set pool. Fit and GIBBON scoring in one job.
Submitted with: sbatch jobs/variational_gp_scaling_mila.sh $*   (Slurm job ${SLURM_JOB_ID:-unknown})"
printf '%s\n' "${WANDB_NOTES}" > "${OUTPUT_DIR}/arm_description.txt"

# --score-chunk-size: each candidate is its own q=1 batch and the variational
# posterior builds a (1024 + 1)^2 covariance per candidate, so 256 is ~1 GB in
# float32. Unchanged from the balanced arm; the support is four times larger
# here, which costs time rather than per-chunk memory.
uv run --no-sync python -m scripts.exact_dkl_top_n \
  config/ampc/variational_gp_scaling.yaml \
  "dataset.initial_data.path=${TRAIN_CSV}" \
  "surrogate.encoder.feature_cache_path=${TRAIN_CACHE}" \
  "surrogate.training_params.variational_objective=${OBJECTIVE}" \
  --output-dir "${OUTPUT_DIR}" \
  --cache-dir cache/ampc/exact_dkl_top_n \
  --eval-chunk-size 2000 \
  --score-chunk-size 256 \
  --extra-set "${GENTAIL_SET}" \
  --predict-only ampc_331k \
  --wandb-project ampc-Variational-GP \
  --wandb-entity models-mila5723 \
  --wandb-group variational-gp-scaling \
  --wandb-tags "variational-gp,svgp,M1024,scaling,${SET_LABEL},n${SIZE},${OBJECTIVE_ARG},${OBJECTIVE},isotropic,matern,outputscale,pinned-mean,noise-prior,lengthscale-prior,scale-inputs,adam,minibatch4096,lr-cosine,fp32,nested,even-fill,keep-pprop3.5,support-cap100k,ampc331k-predict-only" \
  --run-name "${ARM//_/-}-${SLURM_JOB_ID}" \
  "${@:3}"

echo
echo "Fitted and scored into ${OUTPUT_DIR}. Summary:"
echo "  ${OUTPUT_DIR}/eval_summary.json"
