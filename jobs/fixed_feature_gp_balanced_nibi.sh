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
# TRAIN_SET=fps_keep05 fits the same GP on the FPS + keep-y>=0.5 set instead,
# in its own W&B project (ampc-fixed-gp-fps-keep-above); see TRAIN_SET below.
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
# ceiling. KERNEL=iso has three hyperparameters instead of 514 and is expected
# to converge in far fewer evaluations; its first run measures how many.
#SBATCH --time=12:00:00
#SBATCH --job-name=ampc_fixed_gp_bal
# Re-check the fair share before every submit:
#   sshare -l -A def-bengioy_gpu,def-mailhoto_gpu,def-yvesbrun_gpu
#SBATCH --account=def-yvesbrun_gpu
#SBATCH --output=slurm_logs/ampc_fixed_gp_bal_%j.out
#SBATCH --error=slurm_logs/ampc_fixed_gp_bal_%j.err

set -euo pipefail

# Training set, balanced (data/ampc_balanced_25000.csv), fps_keep05 (the same
# stratified farthest-point selection with every 10M row at y >= 0.5 kept, built
# by jobs/exact_dkl_fps_keep05_prepare.sh), fps_keep05_r1 (that selection over
# the round-1 archive) or fps_keep05_45k (it at 45,000 rows). It sets the config, the directory,
# the W&B project and group, the training-set tags and the notes together, and
# nothing else, so the two sets are fitted and evaluated identically:
#   TRAIN_SET=fps_keep05 PRECISION=32 KERNEL=iso FAMILY=matern OUTPUTSCALE=1 \
#     PRIOR_MEAN=0.03956042999405787 sbatch --time=2:00:00 jobs/fixed_feature_gp_balanced_nibi.sh
TRAIN_SET="${TRAIN_SET:-balanced}"
case "${TRAIN_SET}" in
  balanced)
    CONFIG="config/ampc/fixed_feature_gp_balanced.yaml"
    ARM="balanced_25000_fixedgp"
    WANDB_PROJECT="ampc-exact-dkl-balanced"
    WANDB_GROUP="fixed-feature-gp-balanced"
    SET_TAGS="n25000,proportional,top5pct"
    MAX_CHOLESKY=30000
    SET_NOTE="the 25000-molecule BALANCED training set data/ampc_balanced_25000.csv -- the same set, fingerprints and evaluation as balanced-25000-nolayer, the exact-DKL arm the reward-shape study restores."
    ;;
  fps_keep05)
    CONFIG="config/ampc/fixed_feature_gp_fps_keep05.yaml"
    ARM="fps_keep05_25000_fixedgp"
    WANDB_PROJECT="ampc-fixed-gp-fps-keep-above"
    WANDB_GROUP="fixed-feature-gp-fps-keep05"
    SET_TAGS="n25000,proportional,top5pct,keep-above-0.5"
    MAX_CHOLESKY=30000
    SET_NOTE="the 25000-molecule training set data/ampc_fps_keep05_25000.csv: every 10M row at y >= 0.5 (pProp ~4.85) kept outright, then the balanced set's stratified farthest-point selection for the rest (library quantile bands at 50/75/90/95/99/99.75%, the top band [0.232, 0.5) at 5% of the budget, the lower bands proportional to the library, FPS within each band, seed 42). Same fingerprints and evaluation as the balanced arms."
    ;;
  fps_keep05_r1)
    # The same selection over the ROUND-1 archive: the 10M library plus the
    # 10,000 molecules the first reward-shape round docked
    # (jobs/exact_dkl_fps_keep05_r1_prepare.sh). Same project and group as the
    # round-0 arm, so the two fits compare directly.
    CONFIG="config/ampc/fixed_feature_gp_fps_keep05_r1.yaml"
    ARM="fps_keep05_r1_25000_fixedgp"
    WANDB_PROJECT="ampc-fixed-gp-fps-keep-above"
    WANDB_GROUP="fixed-feature-gp-fps-keep05"
    SET_TAGS="n25000,proportional,top5pct,keep-above-0.5,round1-archive"
    MAX_CHOLESKY=30000
    SET_NOTE="the 25000-molecule training set data/round1/ampc_fps_keep05_r1_25000.csv: the fps_keep05 selection run again over the ROUND-1 archive (the 10M library plus the 9,887 molecules the first reward-shape round docked, 699 of them at y >= 0.5), so every row at y >= 0.5 kept outright is now ~1212 rather than 513, and the bands draw from library and generated molecules together. Selector, hyperparameters, fingerprints and evaluation otherwise identical to the round-0 arm -- this simulates the GP the next active-learning round would fit."
    ;;
  fps_keep05_45k)
    # The same selection at 45,000 rows (SIZE=45000 jobs/exact_dkl_fps_keep05_prepare.sh),
    # in the same W&B project and group as the 25,000 arm so the two compare
    # directly.
    CONFIG="config/ampc/fixed_feature_gp_fps_keep05_45k.yaml"
    ARM="fps_keep05_45000_fixedgp"
    WANDB_PROJECT="ampc-fixed-gp-fps-keep-above"
    WANDB_GROUP="fixed-feature-gp-fps-keep05"
    SET_TAGS="n45000,proportional,top5pct,keep-above-0.5"
    MAX_CHOLESKY=50000
    SET_NOTE="the 45000-molecule training set data/ampc_fps_keep05_n45000_45000.csv: the fps_keep05 selection at 45,000 rows instead of 25,000 -- every 10M row at y >= 0.5 (pProp ~4.85) kept outright, then the balanced set's stratified farthest-point selection for the rest (library quantile bands at 50/75/90/95/99/99.75%, the top band [0.232, 0.5) at 5% of the budget, the lower bands proportional to the library, FPS within each band, seed 42). Same fingerprints and evaluation as the balanced arms."
    ;;
  *)
    echo "TRAIN_SET must be balanced, fps_keep05, fps_keep05_r1 or fps_keep05_45k." >&2
    exit 2
    ;;
esac
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
# Kernel, ard (BoTorch's default, one lengthscale per feature) or iso (one
# lengthscale shared by all 512, same prior). Like PRECISION it sets the
# override, the directory, the tags and the notes together:
#   KERNEL=iso sbatch --time=2:00:00 jobs/fixed_feature_gp_balanced_nibi.sh
KERNEL="${KERNEL:-ard}"
if [[ "${KERNEL}" != ard && "${KERNEL}" != iso ]]; then
  echo "KERNEL must be ard or iso." >&2
  exit 2
fi
# Kernel family, rbf (BoTorch's default) or matern (Matern-5/2 under the same
# lengthscale prior and constraint):
#   FAMILY=matern KERNEL=iso sbatch --time=2:00:00 jobs/fixed_feature_gp_balanced_nibi.sh
FAMILY="${FAMILY:-rbf}"
if [[ "${FAMILY}" != rbf && "${FAMILY}" != matern ]]; then
  echo "FAMILY must be rbf or matern." >&2
  exit 2
fi
# Output scale, 0 (BoTorch's default: none, so the signal variance is fixed at
# the training targets' variance) or 1 (a learned, prior-free ScaleKernel):
#   OUTPUTSCALE=1 KERNEL=iso sbatch --time=2:00:00 jobs/fixed_feature_gp_balanced_nibi.sh
OUTPUTSCALE="${OUTPUTSCALE:-0}"
if [[ "${OUTPUTSCALE}" != 0 && "${OUTPUTSCALE}" != 1 ]]; then
  echo "OUTPUTSCALE must be 0 or 1." >&2
  exit 2
fi
# Prior mean on the original target scale, pinned through the fit; empty learns
# it. The AmpC library value is the mean y of data/10M_unif_random_subset.csv
# (n = 10,000,000, mean 0.03956042999405787, std 0.021780709138335935):
#   PRIOR_MEAN=0.03956042999405787 KERNEL=iso sbatch ...
PRIOR_MEAN="${PRIOR_MEAN:-}"
if [[ -n "${PRIOR_MEAN}" && ! "${PRIOR_MEAN}" =~ ^-?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$ ]]; then
  echo "PRIOR_MEAN must be a number." >&2
  exit 2
fi
for argument in "$@"; do
  if [[ "${argument}" == surrogate.outputscale=* || "${argument}" == surrogate.prior_mean=* || "${argument}" == surrogate.kernel=* ]]; then
    echo "Set ${argument%%=*} with FAMILY=rbf|matern, OUTPUTSCALE=0|1 or PRIOR_MEAN=<value>, not" \
      "${argument}, so the directory, tags and notes follow it." >&2
    exit 2
  fi
  if [[ "${argument}" == runtime.precision=* ]]; then
    echo "Set the precision with PRECISION=${argument#runtime.precision=}, not" \
      "${argument}, so the directory, tags and notes follow it." >&2
    exit 2
  fi
  if [[ "${argument}" == surrogate.ard=* ]]; then
    echo "Set the kernel with KERNEL=ard|iso, not ${argument}, so the" \
      "directory, tags and notes follow it." >&2
    exit 2
  fi
done
if [[ "${FAMILY}" == matern ]]; then
  FAMILY_NAME="Matern-5/2"
else
  FAMILY_NAME="RBF"
fi
if [[ "${KERNEL}" == iso ]]; then
  ARM="${ARM}_iso"
  ARD_OVERRIDE="surrogate.ard=false"
  KERNEL_NOTE="ISOTROPIC ${FAMILY_NAME}: one lengthscale shared by all 512 features, under BoTorch's dimension-scaled LogNormal lengthscale prior (the same prior, still scaled by d=512)"
  KERNEL_TAG="isotropic"
else
  ARD_OVERRIDE="surrogate.ard=true"
  KERNEL_NOTE="ARD ${FAMILY_NAME} (one lengthscale per feature) with BoTorch's dimension-scaled LogNormal lengthscale prior"
  KERNEL_TAG="ard"
fi
EXTRA_OVERRIDES=()
if [[ "${FAMILY}" == matern ]]; then
  ARM="${ARM}_matern"
  EXTRA_OVERRIDES+=("surrogate.kernel=matern")
fi
if [[ "${OUTPUTSCALE}" == 1 ]]; then
  ARM="${ARM}_scaled"
  EXTRA_OVERRIDES+=("surrogate.outputscale=true")
  SCALE_NOTE="a learned, prior-free output scale (ScaleKernel, initialized at 1)"
  KERNEL_TAG="${KERNEL_TAG},outputscale"
else
  SCALE_NOTE="no output scale"
fi
if [[ -n "${PRIOR_MEAN}" ]]; then
  ARM="${ARM}_pm${PRIOR_MEAN:0:7}"
  EXTRA_OVERRIDES+=("surrogate.prior_mean=${PRIOR_MEAN}")
  MEAN_NOTE="constant, PINNED at ${PRIOR_MEAN} on the original target scale (converted with each fit's standardization) and not fitted"
  KERNEL_TAG="${KERNEL_TAG},pinned-mean"
else
  MEAN_NOTE="constant, learned"
fi
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
# Fewer fragmentation OOMs near the H100's 80 GB, which the 45,000-row fit nears.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# wandb.init() defaults to online and blocks on api.wandb.ai, which this node
# cannot reach. Offline writes to ${WANDB_DIR}; sync from a login node after.
export WANDB_MODE=offline

OUTPUT_DIR="outputs/ampc/exact_dkl_top_n/${ARM}"
# Per-arm, so concurrent runs never race on a shared ./wandb/latest-run symlink.
export WANDB_DIR="${OUTPUT_DIR}"
mkdir -p "${WANDB_DIR}" slurm_logs

# Shown on the W&B run's Overview page and kept next to the outputs.
export WANDB_NOTES="Exact GP (no inducing points, no minibatching) fitted to ${SET_NOTE}
Surrogate: FixedFeatureGPSurrogate, i.e. BoTorch's default SingleTaskGP on the frozen 512-d MiniMol AmpC fingerprints. Kernel: ${KERNEL_NOTE}, and ${SCALE_NOTE}. Likelihood: Gaussian with the LogNormal(-4, 1) noise prior. Mean: ${MEAN_NOTE}. Inputs min-max scaled per dimension to [0, 1] (scale_inputs=true); targets standardized.
Fit: BoTorch fit_gpytorch_mll (scipy L-BFGS-B) run to convergence, on an exact Cholesky marginal likelihood (max_cholesky_size=${MAX_CHOLESKY}), float${PRECISION} on one H100. The balanced-25000-nolayer arm instead used a prior-free Matern-5/2 kernel with an output scale on fingerprints divided by one fixed scalar, fitted by 1000 Adam steps in float32 that had not converged.
train/epoch/* here is indexed by L-BFGS-B objective evaluation, line searches included, not by epoch. The loss includes the hyperparameter priors, so its level is not comparable with the DKL arm's.
Acquisition scoring skipped, as for the DKL arms.
Arm suffix: ${ARM_SUFFIX:-none}. Submitted with: TRAIN_SET=${TRAIN_SET} sbatch jobs/fixed_feature_gp_balanced_nibi.sh $*   (Slurm job ${SLURM_JOB_ID:-unknown})"
printf '%s\n' "${WANDB_NOTES}" > "${OUTPUT_DIR}/arm_description.txt"

uv run --no-sync python -m scripts.exact_dkl_top_n \
  "${CONFIG}" \
  "runtime.precision=${PRECISION}" \
  "${ARD_OVERRIDE}" \
  ${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"} \
  --output-dir "${OUTPUT_DIR}" \
  --cache-dir cache/ampc/exact_dkl_top_n \
  --eval-chunk-size 2000 \
  --skip-scoring \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-entity models-mila5723 \
  --wandb-group "${WANDB_GROUP}" \
  --wandb-tags "fixed-feature-gp,botorch-default,${FAMILY},${KERNEL_TAG},lbfgs,exact-cholesky,fp${PRECISION},${SET_TAGS}${ARM_SUFFIX:+,${ARM_SUFFIX#_}}" \
  --run-name "${ARM//_/-}-${SLURM_JOB_ID}" \
  "$@"

echo
echo "Fitted into ${OUTPUT_DIR}. Check run/fit/converged in:"
echo "  ${OUTPUT_DIR}/eval_summary.json"
echo "Sync from a login node with:"
echo "  wandb sync ${OUTPUT_DIR}/wandb/offline-run-*"
