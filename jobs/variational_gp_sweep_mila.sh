#!/bin/bash
# One arm of the variational-GP sweep on Mila. Twelve arms: two axes crossed
# with the three sizes of the svgpscale ladder.
#
#   sbatch jobs/variational_gp_sweep_mila.sh <pll|elbo> <on|off> <steps>
#
# FORTY-EIGHT ARMS: objective(2) x lengthscale_prior(2) x size(3) x steps(4),
# fully crossed. Steps are 625 / 1250 / 2500 / 5000 -- a ladder that points
# DOWN, on the 2026-11-10 pilot's evidence; 10,000 is deliberately not a level.
# config/ampc/variational_gp_sweep.yaml "AXIS 4" has the numbers.
#
#   sbatch jobs/variational_gp_sweep_mila.sh pll  on  2500
#   sbatch jobs/variational_gp_sweep_mila.sh elbo off 625
#   SIZE=300000 sbatch jobs/variational_gp_sweep_mila.sh pll on 1250
#
# The whole grid:
#
#   for n in 20000 100000 300000; do for o in pll elbo; do
#     for p in on off; do for s in 625 1250 2500 5000; do
#       SIZE=$n sbatch jobs/variational_gp_sweep_mila.sh $o $p $s
#   done; done; done; done
#
# THE COMPANION OF jobs/exact_gp_sweep_mila.sh, deliberately: same runner, same
# three held-out sets, same metrics, same per-arm artifacts, same W&B
# conventions. SIZE=20000 is the MATCHED cell -- the exact sweep's training set
# molecule for molecule -- and it is the default, because it is the only size at
# which the thesis can compare the two surrogates directly. 100000 and 300000
# are what the exact GP cannot reach: its fit peaked at 38.8 GB at n=25,000.
#
# config/ampc/variational_gp_sweep.yaml holds the reasoning for every fixed
# setting and for both axes; THESIS_ACTIVELEARNING.md section 5 is the decision
# record. noise_floor is deliberately UNSET in all twelve arms -- the floor
# frontier is a separate later sweep on whichever arm this one picks, and it
# has no exact-GP counterpart.
#
# SIZE must be a size the ladder actually built -- both
# data/ampc_svgpscale_<SIZE>.csv and its feature cache are checked below -- and
# it is recorded in the arm name, the output directory, the W&B group and a
# `trainset-` tag, so a run can never be mistaken for one at another size. The
# ladder is NESTED, so a smaller arm is a subset of every larger one; see
# THESIS_ACTIVELEARNING.md "The svgpscale ladder".
#
# STEPS, not epochs. EPOCHS is derived as ceil(STEPS / ceil(SIZE / 4096)), so a
# step level means the same optimization at every size: STEPS=2500 is 2,500
# steps at n=20,000 and 100,000, and 2,516 at 300,000 (34 epochs of 74 steps).
# EPOCHS ARE NOT COMPARABLE ACROSS SIZES; read train/epoch/* against the step
# count, which is in the arm name and the tags.
#
# The step count is an AXIS, and a short one is not a truncated long one:
# lr_schedule is cosine over the whole fit, so each level is a complete
# schedule annealed to lr_min. The levels cannot be read off one long run.
#
# EVALUATION, after the fit: predictions on all three held-out sets, and GIBBON
# scores (value and log scale) on gp_molformer_set and olivier_invitro. The
# sets come from STUDY_SETS in scripts/exact_dkl_top_n.py, so they are the same
# three the exact sweep used and no flag selects them:
#   - ampc_val_screen_232k   230,887  ground-truth labels + ipw. Predict-only in
#                                     STUDY_SETS itself, so NO --predict-only
#                                     flag is passed: that flag only accepts a
#                                     set that would otherwise be scored, and
#                                     naming this one is a hard error.
#   - gp_molformer_set        83,811  scored. Its labels are still wrong (it
#                                     needs data/gpmolformer_prior_100k_docked.csv
#                                     from Nibi), so its label-based metrics
#                                     must be ignored; the GIBBON DISTRIBUTION
#                                     is label-free and is the point of scoring
#                                     it, and its gibbon_log median is a
#                                     validated proxy for the prior's reward
#                                     distribution (within ~50 nats of a
#                                     65,536-sample generation pass).
#   - olivier_invitro          1,521  labelled 2026-11-09 from its DOCK scores.
#   - ampc_val_screen_232k_acq 45,850 scored, via --extra-set. THE set that
#                                     answers "is the reward flat on the
#                                     molecules we are trying to recover": all
#                                     30,850 enriched (15,767 hits) plus a
#                                     seed-42 15,000-row bulk subsample. Same
#                                     set, same rows, in every arm of both
#                                     sweeps. See THESIS section 2.0.
#
# NOTE for anyone reaching for the older jobs/variational_gp_scaling_mila.sh:
# that script passes `--predict-only ampc_331k`, and ampc_331k was RETIRED from
# STUDY_SETS, so the flag is now a hard error and the script cannot run as
# written. This one replaces it for thesis arms.
#
# --score-chunk-size is computed from M, not inherited. GIBBON scores each
# candidate as its own q=1 batch and the variational posterior builds an
# (M + 1)^2 covariance per candidate, so a chunk's memory grows with M^2 and
# NOT with n -- the opposite of the exact GP, whose posterior carries all n
# training inputs per batch element (39 MB per candidate row at n=20,000, which
# is why its chunk had to be 128). The chunk is scaled by (1024 / M)^2 to keep
# it at the ~1 GB that 256 candidates take at M = 1024, which peaked at 10.7 GB
# allocated on the 300k scaling arm.
#
# RESOURCES, all MEASURED on the 2026-11-10 pilot's stage_profile.csv rather
# than inherited: GPU peak 12.74 GiB reserved, host RSS peak 2.05 GiB, run
# 8.1 min at 2,500 steps and 10.4 min at 10,000. The fit is a 4096 x 1024
# cross-covariance per step regardless of training size, so an arm's cost is set
# by its AXIS-4 step level and nothing else; wall time is dominated by GIBBON on
# the 83,811-molecule prior set and the 45,850-row validation slice, which are
# the same in every arm.
#
# main, NOT unkillable, and mem=24G is the reason. The per-user QOS caps are
# cpu=8 / mem=48G / gpu=2, so 48/24 = 2 and 8/4 = 2 both land exactly on the GPU
# cap: TWO ARMS RUN AT ONCE, as the exact sweep's six did. The older
# variational_gp_scaling_mila.sh asked 32G, which is 16x the measured 2.05 GiB
# and would have allowed only one arm; unkillable caps at 1 GPU and would also
# have serialized the grid. At 2 at a time, 48 arms is ~3.6 h.
#
# VRAM is not the binding constraint at all: 12.74 GiB fits every card Mila has,
# including a 24 GB one. l40s is requested for SPEED -- the fit is plain float32
# (no TF32), where an L40S (~90 TFLOPS) is about five times an RTX 8000 and
# faster than an A100, and an untyped request lands on whichever card is free
# first, usually an RTX 8000. If l40s is backed up, override on the submit line
# itself: sbatch --gres=gpu:a100:1 jobs/variational_gp_sweep_mila.sh pll on 2500
#
# The cost of main over unkillable is preemption: a preempted arm requeues onto
# an output directory that already holds its state, which the script refuses to
# reuse. Re-submit that arm rather than passing --overwrite blindly.
#SBATCH --partition=main
#SBATCH --gres=gpu:l40s:1   # 12.74 GiB measured peak; override on the submit line
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --job-name=variational_gp_sweep
#SBATCH --output=slurm_logs/variational_gp_sweep_%j.out
#SBATCH --error=slurm_logs/variational_gp_sweep_%j.err

set -euo pipefail

USAGE="usage: sbatch jobs/variational_gp_sweep_mila.sh <pll|elbo> <on|off> <steps>  [SIZE=20000|100000|300000]"
OBJECTIVE_ARG="${1:?${USAGE}}"
PRIOR="${2:?${USAGE}}"
# AXIS 4, positional and REQUIRED rather than an env default: it is a swept axis
# as of 2026-11-10, and a default would let an arm of the grid be launched
# without stating which level it is.
STEPS="${3:?${USAGE}}"
SIZE="${SIZE:-20000}"
M="${M:-1024}"

case "${OBJECTIVE_ARG}" in
  pll)
    OBJECTIVE="PredictiveLogLikelihood"
    OBJECTIVE_NOTE="the predictive log likelihood, sum_i log E_q[p(y_i | f_i)] - KL(q || p) (Jankowiak et al., 2020), under which the variational variance enters the predictive distribution and has an interior optimum"
    ;;
  elbo)
    OBJECTIVE="VariationalELBO"
    OBJECTIVE_NOTE="the variational ELBO, sum_i E_q[log p(y_i | f_i)] - KL(q || p), under which the variational variance is a pure penalty and the optimum drives it to zero"
    ;;
  *) echo "axis 1 must be 'pll' or 'elbo', got '${OBJECTIVE_ARG}'" >&2; exit 2 ;;
esac

case "${PRIOR}" in
  on)  PRIOR_OVERRIDE="surrogate.lengthscale_prior=true" ;;
  off) PRIOR_OVERRIDE="surrogate.lengthscale_prior=false" ;;
  *) echo "axis 2 must be 'on' or 'off', got '${PRIOR}'" >&2; exit 2 ;;
esac

for name in SIZE STEPS M; do
  if [[ ! "${!name}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${name} must be a positive integer, got '${!name}'" >&2
    exit 2
  fi
done

cd "${HOME}/activelearning"

TRAIN_CSV="data/ampc_svgpscale_${SIZE}.csv"
FEATURE_CACHE="cache/ampc/exact_dkl_top_n/svgpscale_${SIZE}.npy"
# Checked here so a wrong SIZE fails at submit rather than after the queue wait.
# cache_only is true in the config, so a missing cache is a hard error anyway --
# but one that costs the allocation to discover.
for required in "${TRAIN_CSV}" "${FEATURE_CACHE}" "${FEATURE_CACHE}.json"; do
  if [[ ! -e "${required}" ]]; then
    echo "SIZE=${SIZE} needs ${required}, which does not exist." >&2
    echo "Built arms: $(ls data/ampc_svgpscale_*.csv 2>/dev/null | sed 's/.*svgpscale_//;s/\.csv//' | grep -E '^[0-9]+$' | sort -n | tr '\n' ' ')" >&2
    echo "Build one with: SIZES=\"${SIZE} ...\" sbatch jobs/build_svgp_scaling_sets_mila.sh" >&2
    echo "NOTE: only a SMALLER arm can be added safely -- a larger one redraws" >&2
    echo "every band's pool and changes all the existing arms." >&2
    exit 2
  fi
done

# Reject the overrides that would make the arm name a lie, the way the older
# scaling job does. Everything the name encodes has to come from an argument.
for argument in "${@:4}"; do
  case "${argument}" in
    dataset.initial_data.path=*|surrogate.encoder.feature_cache_path=*)
      echo "Set the training set with SIZE=<n>, not ${argument}, so the arm" \
        "name, directory and tags follow it." >&2; exit 2 ;;
    surrogate.training_params.variational_objective=*)
      echo "Set the objective with the first argument (pll|elbo), not ${argument}." >&2
      exit 2 ;;
    surrogate.lengthscale_prior=*)
      echo "Set the lengthscale prior with the second argument (on|off), not ${argument}." >&2
      exit 2 ;;
    surrogate.training_params.epochs=*|surrogate.num_inducing=*|--score-chunk-size*)
      echo "Set these with the third argument (steps) and M=<count>, not" \
        "${argument}, so the arm name and tags follow them." >&2; exit 2 ;;
    surrogate.noise_floor=*)
      echo "noise_floor is UNSET in all twelve arms by design; the floor frontier" \
        "is a separate sweep. See the config header." >&2; exit 2 ;;
  esac
done

# STEPS -> EPOCHS. The fit is minibatched at 4096 rows, so an epoch is 5, 25 or
# 74 steps at the three sizes and only the TOTAL step count is comparable.
STEPS_PER_EPOCH=$(( (SIZE + 4095) / 4096 ))
EPOCHS=$(( (STEPS + STEPS_PER_EPOCH - 1) / STEPS_PER_EPOCH ))
TOTAL_STEPS=$(( EPOCHS * STEPS_PER_EPOCH ))

# A chunk's memory grows with M^2 and not with n; see the header.
SCORE_CHUNK=$(( 256 * 1024 * 1024 / (M * M) ))
(( SCORE_CHUNK < 1 )) && SCORE_CHUNK=1

# The size leads the arm name so the arms of one size sort together, matching
# exact_gp_sweep_mila.sh's n<SIZE>_<axis>_<axis> convention. The STEP COUNT is
# in the name because steps are an AXIS, not a fixed setting: without it two
# step levels of the same cell collide on one output directory, and the script
# refuses to reuse a directory that already holds a surrogate_state.pt -- so the
# second level would fail at submit, or silently need --overwrite and destroy
# the first. TOTAL_STEPS rather than STEPS, since it is what actually ran
# (10000 at n=20,000 and 100,000, 10064 at 300,000).
ARM="n${SIZE}_${OBJECTIVE_ARG}_${PRIOR}_s${TOTAL_STEPS}"
OUTPUT_DIR="outputs/ampc/variational_gp_sweep/${ARM}"
WANDB_PROJECT="${WANDB_PROJECT:-VariationalGP_thesis}"
WANDB_ENTITY="${WANDB_ENTITY:-models-mila5723}"

source .venv/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1   # nothing is downloaded: every model input is a cached feature
# OFFLINE and synced afterwards, as the exact sweep's arms were. Mila compute
# nodes have reached api.wandb.ai (cn-f002 did on 2026-11-10) and the repo's own
# job comments disagree about whether they can, so this costs nothing and
# removes the question. The sync command is printed at the end.
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Per-arm, so arms never race on a shared ./wandb/latest-run symlink.
export WANDB_DIR="${OUTPUT_DIR}"

mkdir -p "${WANDB_DIR}" slurm_logs

export WANDB_NOTES="Variational GP sweep, arm ${ARM}. Sparse variational GP with ${M} learned inducing points on the frozen MiniMol AmpC fingerprints, fitted to ${TRAIN_CSV} (${SIZE} molecules, the svgpscale ladder).
THE COMPANION OF THE EXACT-GP SWEEP (W&B ExactGP_thesis, config/ampc/exact_gp_sweep.yaml). Everything the two surrogates share is pinned to the same value: the training set, the frozen encoder and its cache, Matern-5/2, one lengthscale over all 512 features (ard=false), inputs min-max scaled to [0,1] on the training rows, a learned output scale, the constant mean PINNED at 0.0042864189360438442 (the 10M library mean), standardized outputs, float32, seed 42, and GIBBON with num_mv_samples=10 scored on both the value and the log scale. n=20000 is the MATCHED cell -- the exact sweep's training set molecule for molecule, and the only size the exact GP could fit (38.8 GB at n=25000).
AXIS 1, objective: ${OBJECTIVE}. Fitted on ${OBJECTIVE_NOTE}. This axis does not exist for an exact GP and is the largest effect measured in this project: on old y at 300k it moved the gp_molformer_set log-GIBBON median -2688 -> -376 and the de-biased latent calibration slope 28.3 -> 1.28 for 0.008 of Pearson.
AXIS 2, lengthscale prior: ${PRIOR}. BoTorch's dimension-scaled LogNormal(sqrt2 + 0.5*log 512, sqrt3), mode 4.63, started at its mode; off means prior-free and started at sqrt(512) = 22.63, so the axis changes both the gradient and the starting point and a difference cannot be split between them. This is the axis that decided the exact sweep: prior-off predicted better pointwise and was unusable, with one molecule of 83811 holding 90% of the acquisition mass.
NOT SWEPT: prior_mean (the exact sweep measured its three levels inert, 16-89x below the axis-1 gap), ard, num_inducing, inducing_init, seed, steps, and noise_floor -- which is UNSET by design, for the same reason the exact sweep left noise_std_floor unset: flooring the noise pins the outputscale-to-noise ratio by fiat, and that ratio is the latent variance GIBBON consumes.
SCHEDULE: Adam on 4096-row minibatches, ${EPOCHS} epochs of ${STEPS_PER_EPOCH} steps = ${TOTAL_STEPS} steps, learning rate cosine-annealed 3e-3 -> 1e-4. Epochs are NOT comparable across sizes; steps are, and 10064 is the incumbent arm's count.
ACQUISITION: GIBBON's max-value support is capped at 100000 (max_support_size), so the 20k and 100k arms use their whole training set and only the 300k arm is drawn down; run/acquisition/*/acqf_support_size records it. At n=20000 that is the same support the exact sweep's uncapped spec gives. ${SCORE_CHUNK} candidates per chunk, scaled by (1024/M)^2 -- the variational posterior's chunk grows with M^2 and not with n.
EVALUATED on the three STUDY_SETS: ampc_val_screen_232k (230887, ground-truth labels + ipw, predict-only), gp_molformer_set (83811, scored; its labels are still wrong, so read only its label-free GIBBON distribution) and olivier_invitro (1521, labelled 2026-11-09).
Submitted with: SIZE=${SIZE} STEPS=${STEPS} M=${M} sbatch jobs/variational_gp_sweep_mila.sh $*   (Slurm job ${SLURM_JOB_ID:-unknown})"
printf '%s\n' "${WANDB_NOTES}" > "${OUTPUT_DIR}/arm_description.txt"

nvidia-smi --query-gpu=name,memory.total --format=csv

uv run --no-sync python -m scripts.exact_dkl_top_n \
  config/ampc/variational_gp_sweep.yaml \
  "surrogate.training_params.variational_objective=${OBJECTIVE}" \
  "${PRIOR_OVERRIDE}" \
  "surrogate.training_params.epochs=${EPOCHS}" \
  "surrogate.num_inducing=${M}" \
  "dataset.initial_data.path=${TRAIN_CSV}" \
  "surrogate.encoder.feature_cache_path=${FEATURE_CACHE}" \
  --output-dir "${OUTPUT_DIR}" \
  --cache-dir cache/ampc/exact_dkl_top_n \
  --eval-chunk-size 5000 \
  --score-chunk-size "${SCORE_CHUNK}" \
  --acquisition-seed 42 \
  --extra-set ampc_val_screen_232k_acq \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-entity "${WANDB_ENTITY}" \
  --wandb-group "variational-gp-n${SIZE}" \
  --wandb-tags "variational-gp-sweep,svgp,n${SIZE},trainset-svgpscale${SIZE},${OBJECTIVE_ARG},${OBJECTIVE},prior-${PRIOR},M${M},steps${TOTAL_STEPS},isotropic,matern,outputscale,pinned-mean,noise-prior,scale-inputs,no-noise-floor,adam,minibatch4096,lr-cosine,lr3e-3,fp32,nested,keep-pprop3.5,support-cap100k,chunk${SCORE_CHUNK}" \
  --run-name "${ARM}-${SLURM_JOB_ID}" \
  "${@:4}"

echo
echo "Fitted and scored into ${OUTPUT_DIR}. Summary:"
echo "  ${OUTPUT_DIR}/eval_summary.json"
echo
echo "Sync from a LOGIN node:"
echo "  cd ${HOME}/activelearning && source .venv/bin/activate"
echo "  wandb sync ${OUTPUT_DIR}/wandb/offline-run-*"
