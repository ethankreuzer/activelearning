#!/bin/bash
# Fit the sparse variational GP (VariationalGPSurrogate, 1024 inducing points)
# on the balanced 25,000-molecule training set, then evaluate and score it
# exactly as the exact-DKL and fixed-feature-GP arms in ampc-exact-dkl-balanced
# were, into the W&B project ampc-Variational-GP.
#
#   sbatch jobs/variational_gp_balanced_nibi.sh elbo
#   sbatch jobs/variational_gp_balanced_nibi.sh pll
#   LR_SCHEDULE=cosine sbatch jobs/variational_gp_balanced_nibi.sh pll
#
# LR_SCHEDULE=cosine anneals Adam's rate from the config's 1e-2 to LR_MIN
# (default 1e-4) over the 2000 steps, instead of holding it constant. It was
# added after the first two arms: the ELBO arm's loss spiked at steps 1435-1450
# and settled worse (outputscale 0.88 -> 0.41), and the PLL arm's lengthscale
# and outputscale were still drifting at step 2000 on a flat loss. The rate is
# charted as train/epoch/lr.
#
# The two arms differ in the training objective and nothing else; see
# activelearning.surrogate.objectives for why it matters to GIBBON. Everything
# else is in config/ampc/variational_gp_balanced.yaml: the isotropic Matern,
# output scale, noise prior, [0, 1] inputs and pinned mean of the exact GP the
# 10-round single-fidelity run used, with 1024 inducing points and 2000
# full-batch Adam steps.
#
# Unlike the exact-GP arms, fitting and scoring run in ONE job: they were split
# because an exact posterior concatenates all 25,000 training rows onto every
# candidate, whereas this one only carries the inducing points.
#
# What to read, as for the exact arms: ampc_331k and gp_molformer_set
# coverage_2std, spearman_std_error and top1pct_overlap; the *_unseen variants;
# ampc_331k/acquisition/gibbon_*/top1000_enrichment, effective_support and
# fraction_at_floor; and run/hyperparameters/* against the exact GP's
# (lengthscale, outputscale, noise). train/epoch/loss shows whether 2000 steps
# reached a plateau. The loss includes the priors and the KL term, so its level
# is comparable between these two arms only in shape, not in value: ELBO and
# PLL are different functionals.
#
# Resources: the fit is GPU linear algebra on a 25,000 x 1024 cross-covariance
# and 4 cores are plenty. The largest evaluation cache is 648 MB and the DKL
# arm peaked at 2.2 GB of host memory. The wall time is not yet measured for
# this model; the first run measures it (stage_profile.csv), so trim --time
# from it for later arms.
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --job-name=ampc_svgp_bal
# Re-check the fair share before every submit:
#   sshare -l -A def-bengioy_gpu,def-mailhoto_gpu,def-yvesbrun_gpu
#SBATCH --account=def-yvesbrun_gpu
#SBATCH --output=slurm_logs/ampc_svgp_bal_%j.out
#SBATCH --error=slurm_logs/ampc_svgp_bal_%j.err

set -euo pipefail

OBJECTIVE_ARG="${1:-}"
case "${OBJECTIVE_ARG}" in
  elbo)
    OBJECTIVE="VariationalELBO"
    OBJECTIVE_NOTE="the variational ELBO, sum_i E_q[log p(y_i | f_i)] - KL(q || p), which penalizes the variational variance directly"
    ;;
  pll)
    OBJECTIVE="PredictiveLogLikelihood"
    OBJECTIVE_NOTE="the predictive log likelihood, sum_i log E_q[p(y_i | f_i)] - KL(q || p) (Jankowiak et al., 2020), which folds the variational variance into the predictive distribution and typically calibrates it better"
    ;;
  *)
    echo "usage: sbatch jobs/variational_gp_balanced_nibi.sh elbo|pll [extra script flags or overrides...]" >&2
    exit 2
    ;;
esac
LR_SCHEDULE="${LR_SCHEDULE:-constant}"
LR_MIN="${LR_MIN:-1e-4}"
case "${LR_SCHEDULE}" in
  constant)
    SCHEDULE_SUFFIX=""
    SCHEDULE_NOTE="a constant learning rate of 1e-2"
    ;;
  cosine)
    if [[ ! "${LR_MIN}" =~ ^[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$ ]]; then
      echo "LR_MIN must be a non-negative number." >&2
      exit 2
    fi
    SCHEDULE_SUFFIX="_cosine${LR_MIN}"
    SCHEDULE_NOTE="a learning rate cosine-annealed from 1e-2 to ${LR_MIN} over the 2000 steps"
    ;;
  *)
    echo "LR_SCHEDULE must be constant or cosine." >&2
    exit 2
    ;;
esac
for argument in "${@:2}"; do
  if [[ "${argument}" == surrogate.training_params.variational_objective=* ]]; then
    echo "Set the objective with the first argument (elbo|pll), not ${argument}," \
      "so the directory, tags and notes follow it." >&2
    exit 2
  fi
  if [[ "${argument}" == surrogate.training_params.lr_schedule=* || "${argument}" == surrogate.training_params.lr_min=* ]]; then
    echo "Set the schedule with LR_SCHEDULE=constant|cosine and LR_MIN=<value>, not" \
      "${argument}, so the directory, tags and notes follow it." >&2
    exit 2
  fi
done

ARM="balanced_25000_svgp_M1024_iso_matern_scaled_pm0.03956_${OBJECTIVE_ARG}${SCHEDULE_SUFFIX}_fp32"

cd /home/ethankrz/activelearning
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1   # compute nodes have no internet
# wandb.init() defaults to online and blocks on api.wandb.ai, which this node
# cannot reach. Offline writes to ${WANDB_DIR}; sync from a login node after.
export WANDB_MODE=offline

OUTPUT_DIR="outputs/ampc/variational_gp/${ARM}"
# Per-arm, so concurrent runs never race on a shared ./wandb/latest-run symlink.
export WANDB_DIR="${OUTPUT_DIR}"
mkdir -p "${WANDB_DIR}" slurm_logs

# Shown on the W&B run's Overview page and kept next to the outputs.
export WANDB_NOTES="Sparse variational GP with 1024 learned inducing points (VariationalGPSurrogate), fitted to the 25000-molecule BALANCED training set data/ampc_balanced_25000.csv -- the same set, fingerprints and evaluation as the exact GP the 10-round single-fidelity run used (balanced_25000_fixedgp_iso_matern_scaled_pm0.03956_fp32).
Borrowed from that exact GP: Matern-5/2 with ONE lengthscale shared by all 512 features under BoTorch's dimension-scaled LogNormal prior and 0.025 floor; inputs min-max scaled to [0, 1] with the training rows' range; a learned ScaleKernel output scale; BoTorch's LogNormal(-4, 1) noise prior and 1e-4 floor; the constant mean PINNED at 0.03956042999405787 (the 10M library mean); float32. Both floors use gpytorch's softplus rather than BoTorch's untransformed bound, because Adam cannot enforce a bound.
Different: the posterior is carried by 1024 inducing points started at uniform training rows and then learned, instead of conditioning on all 25000 rows; it is fitted by 2000 full-batch Adam steps, with ${SCHEDULE_NOTE}, on ${OBJECTIVE_NOTE}, instead of L-BFGS-B on the exact marginal likelihood.
train/epoch/* here is indexed by Adam step (full batch, so one step per epoch). The loss includes the KL term and the priors (divided by n), so it is not comparable in level with the exact GP's, nor between the ELBO and PLL arms.
Fit and GIBBON scoring in one job (no separate scoring run).
Submitted with: LR_SCHEDULE=${LR_SCHEDULE} LR_MIN=${LR_MIN} sbatch jobs/variational_gp_balanced_nibi.sh $*   (Slurm job ${SLURM_JOB_ID:-unknown})"
printf '%s\n' "${WANDB_NOTES}" > "${OUTPUT_DIR}/arm_description.txt"

# --score-chunk-size: each candidate is its own q=1 batch, and the variational
# posterior builds a (1024 + 1)^2 covariance per candidate, so a chunk of 256 is
# ~1 GB in float32. scripts/surrogate_eval_fit.py's budget for the same thing
# (325,000 kernel rows) gives 317 at M = 1024.
uv run --no-sync python -m scripts.exact_dkl_top_n \
  config/ampc/variational_gp_balanced.yaml \
  "surrogate.training_params.variational_objective=${OBJECTIVE}" \
  "surrogate.training_params.lr_schedule=${LR_SCHEDULE}" \
  "surrogate.training_params.lr_min=${LR_MIN}" \
  --output-dir "${OUTPUT_DIR}" \
  --cache-dir cache/ampc/exact_dkl_top_n \
  --eval-chunk-size 2000 \
  --score-chunk-size 256 \
  --wandb-project ampc-Variational-GP \
  --wandb-entity models-mila5723 \
  --wandb-group variational-gp-balanced \
  --wandb-tags "variational-gp,svgp,M1024,${OBJECTIVE_ARG},${OBJECTIVE},isotropic,matern,outputscale,pinned-mean,noise-prior,lengthscale-prior,scale-inputs,adam,full-batch,lr-${LR_SCHEDULE},fp32,n25000,proportional,top5pct" \
  --run-name "${ARM//_/-}-${SLURM_JOB_ID}" \
  "${@:2}"

echo
echo "Fitted and scored into ${OUTPUT_DIR}. Summary:"
echo "  ${OUTPUT_DIR}/eval_summary.json"
echo "Sync from a login node with:"
echo "  wandb sync ${OUTPUT_DIR}/wandb/offline-run-*"
