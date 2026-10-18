#!/bin/bash
# Fit BoTorch's default exact GP (FixedFeatureGPSurrogate) on the balanced
# 25,000-molecule training set, evaluate it exactly as the exact-DKL arms are,
# and save the state for the reward-shape arms to restore.
#
#   sbatch jobs/fixed_feature_gp_balanced_nibi.sh [extra script flags or overrides...]
#
# The counterpart of jobs/exact_dkl_balanced.sh 25000 none (the Narval arm
# balanced_25000_nolayer, which the reward-shape arms restore today). Same
# training set, fingerprints, evaluation sets, runner and W&B project, so the
# two land side by side in ampc-exact-dkl-balanced. What differs is the GP; see
# config/ampc/fixed_feature_gp_balanced.yaml.
#
# What to read first: run/fit/converged (1 only if the last L-BFGS-B attempt met
# scipy's convergence test; BoTorch also accepts running out of iterations) and
# run/fit/lbfgs_attempts (above 1 means BoTorch restarted from prior draws).
# Then the same comparison as the DKL arms: ampc_331k and gp_molformer_set
# coverage_2std, spearman_std_error and top1pct_overlap, and the lengthscale
# spread in run/hyperparameters.
#
# One H100 bundle on Nibi is 14 cores and 250G. The fit is GPU linear algebra
# and L-BFGS-B's own steps are trivial, so 4 cores suffice; the DKL arm peaked
# at 2.2 GB of host memory and the largest evaluation cache is 648 MB.
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
# Measured by the float64 smoke run (job 24351124, 5 L-BFGS-B steps): ~1.5 s
# per objective evaluation, ~70 s of setup, ~70 s to predict all six sets, and
# a 38.8 GB GPU peak. 3,000 evaluations is ~75 min, and BoTorch's worst case of
# five such attempts ~6 h, so submit with --time=6:00:00; the 12 h here is the
# ceiling.
#SBATCH --time=12:00:00
#SBATCH --job-name=ampc_fixed_gp_bal
# Re-check the fair share before every submit:
#   sshare -l -A def-bengioy_gpu,def-mailhoto_gpu,def-yvesbrun_gpu
#SBATCH --account=def-yvesbrun_gpu
#SBATCH --output=slurm_logs/ampc_fixed_gp_bal_%j.out
#SBATCH --error=slurm_logs/ampc_fixed_gp_bal_%j.err

set -euo pipefail

ARM="balanced_25000_fixedgp"
# Fit precision, 64 (the config's) or 32 (the DKL no-layer arm's). It sets the
# override, the directory, the tags and the notes together, so a float32 run is
# never labelled float64:
#   PRECISION=32 sbatch --time=6:00:00 jobs/fixed_feature_gp_balanced_nibi.sh
# A float32 fit may end without converging: L-BFGS-B's stopping tolerance is
# below float32's resolution of the objective. run/fit/converged records it.
PRECISION="${PRECISION:-64}"
if [[ "${PRECISION}" != 64 && "${PRECISION}" != 32 ]]; then
  echo "PRECISION must be 64 or 32." >&2
  exit 2
fi
for argument in "$@"; do
  if [[ "${argument}" == runtime.precision=* ]]; then
    echo "Set the precision with PRECISION=${argument#runtime.precision=}, not" \
      "${argument}, so the directory, tags and notes follow it." >&2
    exit 2
  fi
done
if [[ "${PRECISION}" == 32 ]]; then
  ARM="${ARM}_fp32"
fi
# Optional suffix for a run that must not land in the real arm's directory, such
# as a smoke run capped at a few L-BFGS-B iterations:
#   ARM_SUFFIX=_smoke sbatch --time=0:45:00 jobs/fixed_feature_gp_balanced_nibi.sh \
#     surrogate.fit_kwargs.optimizer_kwargs.options.maxiter=5
# sbatch exports the submitting shell's environment, so this reaches the job.
ARM_SUFFIX="${ARM_SUFFIX:-}"
if [[ -n "${ARM_SUFFIX}" && ! "${ARM_SUFFIX}" =~ ^_[A-Za-z0-9.-]+$ ]]; then
  echo "ARM_SUFFIX must look like _name (letters, digits, '.', '-')." >&2
  exit 2
fi
ARM="${ARM}${ARM_SUFFIX}"

cd /home/ethankrz/activelearning
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1   # compute nodes have no internet
# wandb.init() defaults to online and blocks on api.wandb.ai, which this node
# cannot reach. Offline writes to ${WANDB_DIR}; sync from a login node after.
export WANDB_MODE=offline

OUTPUT_DIR="outputs/ampc/exact_dkl_top_n/${ARM}"
# Per-arm, so concurrent runs never race on a shared ./wandb/latest-run symlink.
export WANDB_DIR="${OUTPUT_DIR}"
mkdir -p "${WANDB_DIR}" slurm_logs

# Shown on the W&B run's Overview page and kept next to the outputs.
export WANDB_NOTES="Exact GP (no inducing points, no minibatching) fitted to the 25000-molecule BALANCED training set data/ampc_balanced_25000.csv -- the same set, fingerprints and evaluation as balanced-25000-nolayer, the exact-DKL arm the reward-shape study restores.
Surrogate: FixedFeatureGPSurrogate, i.e. BoTorch's default SingleTaskGP on the frozen 512-d MiniMol AmpC fingerprints. Kernel: ARD RBF with BoTorch's dimension-scaled LogNormal lengthscale prior and no output scale. Likelihood: Gaussian with the LogNormal(-4, 1) noise prior. Mean: constant, learned. Inputs min-max scaled per dimension to [0, 1] (scale_inputs=true); targets standardized.
Fit: BoTorch fit_gpytorch_mll (scipy L-BFGS-B) run to convergence, on an exact Cholesky marginal likelihood (max_cholesky_size=30000), float${PRECISION} on one H100. The balanced-25000-nolayer arm instead used a prior-free Matern-5/2 kernel with an output scale on fingerprints divided by one fixed scalar, fitted by 1000 Adam steps in float32 that had not converged.
train/epoch/* here is indexed by L-BFGS-B objective evaluation, line searches included, not by epoch. The loss includes the hyperparameter priors, so its level is not comparable with the DKL arm's.
Acquisition scoring skipped, as for the DKL arms.
Arm suffix: ${ARM_SUFFIX:-none}. Submitted with: sbatch jobs/fixed_feature_gp_balanced_nibi.sh $*   (Slurm job ${SLURM_JOB_ID:-unknown})"
printf '%s\n' "${WANDB_NOTES}" > "${OUTPUT_DIR}/arm_description.txt"

uv run --no-sync python -m scripts.exact_dkl_top_n \
  config/ampc/fixed_feature_gp_balanced.yaml \
  "runtime.precision=${PRECISION}" \
  --output-dir "${OUTPUT_DIR}" \
  --cache-dir cache/ampc/exact_dkl_top_n \
  --eval-chunk-size 2000 \
  --skip-scoring \
  --wandb-project ampc-exact-dkl-balanced \
  --wandb-entity models-mila5723 \
  --wandb-group fixed-feature-gp-balanced \
  --wandb-tags "fixed-feature-gp,botorch-default,rbf,lbfgs,exact-cholesky,fp${PRECISION},n25000,proportional,top5pct${ARM_SUFFIX:+,${ARM_SUFFIX#_}}" \
  --run-name "${ARM//_/-}-${SLURM_JOB_ID}" \
  "$@"

echo
echo "Fitted into ${OUTPUT_DIR}. Check run/fit/converged in:"
echo "  ${OUTPUT_DIR}/eval_summary.json"
echo "Sync from a login node with:"
echo "  wandb sync ${OUTPUT_DIR}/wandb/offline-run-*"
