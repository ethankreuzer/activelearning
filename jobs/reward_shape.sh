#!/bin/bash
# One arm of the reward-shape study: a single active-learning round in which the
# surrogate is restored rather than refitted, so the only thing that differs
# between arms is the reward the S3-GFN policy is trained against.
#
#   sbatch jobs/reward_shape.sh <exponential|power> <beta> <initial_log_z> [seed]
#
# initial_log_z is where the RTB normalizer starts, and it is required because
# there is no safe default: it is log E[R] under the pretrained prior, which
# depends on beta and on the log-score floor, and is not proportional to either.
# Read it per arm from the log_Z column of
#   python -m scripts.reward_shape_preview <scored run> --log-score-floor -100
# and pass 0 for exponential arms, where it measures 0.000. The four power arms
# of 2026-10-08 (jobs 5024309, 5024311, 5024318, 5024345) started at 0, 9-23
# nats from the truth, and collapsed to 0.4% validity within ~1000 updates; see
# REWARD_SHAPE_LOGZ_FIX.md.
#
# beta means different things in the two transforms and the grids are NOT
# interchangeable:
#
#   power        R = acq ** beta        beta is an exponent. The reward ratio
#                                       between two molecules follows the RATIO
#                                       of their information gains, which is what
#                                       separates a score spanning orders of
#                                       magnitude. docs/tutorials/gflownet_sampler.md
#                                       recommends 0.01-0.5 for such a score;
#                                       beta=1 collapsed the policy in September.
#   exponential  R = exp(beta * acq)    beta is an inverse temperature. The ratio
#                                       follows the DIFFERENCE of the information
#                                       gains, and a batch of 64 generated
#                                       molecules has a best acq near 1e-29, so
#                                       separating it would take beta ~ 1e29.
#                                       Measured flat (effective support exactly
#                                       64.00 of 64) at both beta=100 and
#                                       beta=500. Run it as a control, not to win.
#
# Run scripts/reward_shape_preview.py first: it predicts each arm's reward
# concentration from the acquisition scores already on disk, in minutes on a CPU
# allocation, so a beta that is certain to be flat or degenerate never reaches a
# GPU. Measured on gp_molformer_set, the stand-in for what the policy generates
# (effective support out of a batch of 64, and the best molecule's reward as a
# multiple of the batch mean):
#
#   arm                support   max(R)/mean(R)
#   power:0.05            63.1             2.6    flat
#   power:0.1             57.1             6.7    weak but not nothing
#   power:0.125           50.1            10.3
#   power:0.15            40.2            15.4
#   power:0.175           28.8            21.9
#   power:0.2             18.4            29.6
#   power:0.25             6.4            44.8    near the collapse end
#   exponential:100       64.00            1.0    flat
#   exponential:500       64.00            1.0    flat
#
# The two ends are known failures rather than guesses. September's power beta=1
# sat at a support of about 2 and collapsed the policy; exponential beta=100 sat
# at 64 with a ratio of exactly 1 and never moved it. Nothing in between has been
# run, which is what the arms below are for, so they span the interior widely
# instead of clustering.
#
# That table and the paragraph below were measured under the old PER-BATCH floor
# (20 nats below each batch's own best). The arms now run with a fixed floor of
# LOG_SCORE_FLOOR on log(acq), so rerun the preview with --log-score-floor before
# reading support numbers off it.
#
# Note what the floor fraction of 0.969 means, identically for every power arm:
# about 62 of 64 molecules in a batch are tied, so the reward is effectively
# two-level and beta only sets how hard the other ~2 are favoured. No beta in
# this family grades the reward across many molecules. That tie is set by the
# acquisition, not only by the 20-nat MAX_LOG_SCORE_SPREAD cap: on generated
# chemistry GIBBON's log information gain runs from about -65 at a batch's best
# molecule to a median of -225, so widening the cap would release only a few more
# molecules. If every arm here trains poorly, that distribution is the thing to
# attack next, not beta.
#
# What this produces that the September reward arms did not: per-training-step
# figures that separate the two failure modes --
#   sampler/s3gfn/acq/trajectory              is the policy finding better molecules
#   sampler/s3gfn/reward/concentration        how hard the loss reweights the prior
#   sampler/s3gfn/reward/effective_support    how many molecules the reward spreads over
#   sampler/s3gfn/train/batch_health          valid / synthesizable / unique fractions
#
# Read them in that last order first: acq rising while validity or uniqueness
# falls is the policy fleeing the data the GP was fitted on, not learning.
#
# Cost: ~82 min of S3-GFN training at 10,000 steps, then 100k-candidate
# generation, then ~20 min to dock the 2,000 molecules the round selects.
#SBATCH --gres=gpu:a100:1
# 48 CPUs so the Dock3 thread pool, sized from the job's CPUs, matches the
# earlier AmpC runs.
#SBATCH --cpus-per-task=48
# All memory on a 48-core A100 node (RealMemory=510000M). Narval rejects --mem=0
# unless the whole node is requested, and this uses one GPU.
#SBATCH --mem=510000M
# 6h, down from 12h on 2026-10-11, to be backfillable. Measured from the collapsed
# 2026-10-08 arms (5024309, 5024345): 10,000 training steps took 89 min there
# (generated_total=640000 = 10000 x 64, so every step's generation ran), and the
# remaining ~110 min of their 3h20 went into _generate_final_candidates grinding
# 2,000,000 attempts before raising. A healthy arm trades that thrashing for real
# generation but pays for ~9,500 gradient updates those runs skipped
# (online_updates was 429), so budget 3-4.5h. 6h clears that, and if the log_z fix
# does not hold the arm still dies at ~3h20 and the cap costs nothing. Fairshare
# dominates Narval priority (weight 5e6 against 2500 for age), so a shorter job is
# the only lever that moves a queued arm; the window it needs is what shrinks.
#SBATCH --time=6:00:00
#SBATCH --job-name=ampc_reward_shape
# RE-CHECK THIS EVERY TIME; it is a snapshot, not a ranking. Measured
# 2026-10-13 with `sshare -l -A def-bengioy_gpu,def-mailhoto_gpu,def-yvesbrun_gpu`:
#
#   account            acct RawShares   acct LevelFS   my FairShare
#   def-yvesbrun_gpu          16287          2.205          0.369   <- this one
#   def-bengioy_gpu           16287          0.936          0.343
#   def-mailhoto_gpu          16287          0.730          0.329
#
# Two numbers, and both matter. The ACCOUNT row's LevelFS says whether the account
# is under its share (above 1) or over it; only yvesbrun is under today. The
# USER row's FairShare is the real priority factor -- a job's FAIRSHARE term is
# exactly 5e6 x that number -- so it is what to compare once two accounts both
# hold real shares. Here yvesbrun wins on both.
#
# This has now flipped twice in five days, which is why it is re-measured rather
# than remembered. def-yvesbrun_gpu held RawShares 14353 / LevelFS 1.57 on
# 2026-10-08, collapsed to RawShares 1 by 2026-10-10 (arms queued on it sat behind
# ~1079 of 1197 pending GPU jobs), and is restored to 16287 / LevelFS 2.205 as of
# 2026-10-13 -- it has the lowest RawUsage of the three (3.59e9 against bengioy's
# 8.46e9 and mailhoto's 1.08e10) on equal shares. Meanwhile bengioy slipped from
# LevelFS 1.108 to 0.936, so the 2026-10-11 choice is now the wrong one. The three
# arms submitted 2026-10-10 went to def-mailhoto_gpu and waited 35-38 h to start.
#
# def-alexhg_gpu and def-pr61079_gpu still carry RawShares=1 with NormShares
# 0.000000. Their user-level FairShare looks excellent (0.985, 0.680) and their
# LevelFS of inf is zero usage against zero shares, which collapses on first use
# for every member of the account. `sshare -U -u $USER` shows only the user's share
# *within* each account and so hides this entirely; always check the account row
# with `sshare -l -A <acct>`.
#SBATCH --account=def-yvesbrun_gpu
#SBATCH --output=slurm_logs/ampc_reward_shape_%j.out
#SBATCH --error=slurm_logs/ampc_reward_shape_%j.err

set -euo pipefail

TRANSFORM="${1:-}"
BETA="${2:-}"
INITIAL_LOG_Z="${3:-}"
SEED="${4:-}"
if [[ ! "${TRANSFORM}" =~ ^(exponential|power)$ \
      || ! "${BETA}" =~ ^[0-9]*\.?[0-9]+$ \
      || ! "${INITIAL_LOG_Z}" =~ ^-?[0-9]*\.?[0-9]+$ \
      || ! "${SEED}" =~ ^[0-9]*$ ]]; then
  echo "usage: sbatch jobs/reward_shape.sh <exponential|power> <beta> <initial_log_z> [seed]" >&2
  exit 2
fi

# Fixed for the whole study, and stated here rather than left to the YAML so the
# notes and tags below cannot drift from what the run was given.
#   LOG_SCORE_FLOOR  lower bound on log(acq) under power; a constant, so a
#                    molecule's reward no longer depends on its batch.
#   LOG_Z_LR         learning rate of the log_z scalar (the policy's is 1e-4).
LOG_SCORE_FLOOR="-100"
LOG_Z_LR="0.01"

# The arm name carries only the non-defaults, as the exact-DKL arms do:
#   reward_<transform>_beta<beta>_logz<initial_log_z>[_s<seed>]
# The logz part also keeps these arms out of the directories of the 2026-10-08
# ones, which started log_z at 0 under the per-batch floor.
ARM="reward_${TRANSFORM}_beta${BETA}_logz${INITIAL_LOG_Z}"
SEED_OVERRIDE=()
SEED_TAG=""
if [[ -n "${SEED}" ]]; then
  ARM="${ARM}_s${SEED}"
  SEED_OVERRIDE=("runtime.seed=${SEED}")
  SEED_TAG=",seed${SEED}"
fi

cd /home/ethankrz/activelearning
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1   # compute nodes have no internet
# wandb.init() defaults to online and blocks on api.wandb.ai, which this node
# cannot reach. Offline writes to ${WANDB_DIR}; sync from a login node after.
export WANDB_MODE=offline

FITTED_DIR="outputs/ampc/exact_dkl_top_n/balanced_25000_nolayer"
STATE_PATH="${FITTED_DIR}/surrogate_state.pt"
OUTPUT_DIR="outputs/ampc/reward_shape/${ARM}"
# Why validity is lost, not just how much: every discarded generation is counted
# by reason, and a reservoir sample of them is written here. The `{round}`
# placeholder must not start the value -- OmegaConf runs the dotlist through a
# YAML loader, which would read a leading `{` as a flow mapping. Prefixed by
# OUTPUT_DIR it parses as a plain string.
REJECT_LOG_PATH="${OUTPUT_DIR}/rejects_round_{round}.csv"
if [[ ! -f "${STATE_PATH}" ]]; then
  echo "${STATE_PATH} not found: fit the no-layer arm first with" >&2
  echo "  sbatch jobs/exact_dkl_balanced.sh 25000 none" >&2
  exit 2
fi

# Per-arm, so concurrent arms never race on a shared ./wandb/latest-run symlink.
export WANDB_DIR="${OUTPUT_DIR}"
mkdir -p "${WANDB_DIR}" slurm_logs

if [[ "${TRANSFORM}" == power ]]; then
  REWARD_TEXT="R = acq ** ${BETA}, so the reward RATIO between two molecules follows the ratio of their information gains. The transform takes the logarithm itself, which is why the acquisition stays on the value scale. log(acq) is floored at the FIXED value ${LOG_SCORE_FLOOR} nats, the same for every batch, so a molecule's reward depends on that molecule alone; everything below the floor is tied. Earlier arms floored 20 nats below each batch's own best instead, which made the reward level swing from batch to batch"
else
  REWARD_TEXT="R = exp(${BETA} * acq), so the reward ratio follows the DIFFERENCE of the information gains. Generated molecules score ~1e-14 at the 99.9th percentile against a best of ~4e-3, so the bulk of a batch is expected to stay tied at this or any beta; this arm exists to demonstrate that limit"
fi

# A plain-language description, so the run can be understood on its own months
# later. wandb reads WANDB_NOTES at init and shows it on the run's Overview page
# and in the Notes column of the runs table; the same text is kept next to the
# run's outputs.
export WANDB_NOTES="REWARD-SHAPE arm ${ARM}: ONE active-learning round, varying only the reward the S3-GFN policy is trained against.
Surrogate: RESTORED, NOT REFITTED. The exact GP (no inducing points, no minibatching) is rebuilt on the 25000-molecule BALANCED training set data/ampc_balanced_25000.csv -- the library-proportional mix, top band floored at 5% of the budget -- and its fitted parameters are loaded from ${STATE_PATH} in place of training, so every arm of this study shares one surrogate exactly. See ${FITTED_DIR}/arm_description.txt for how that fit was obtained. Features: frozen MiniMol AmpC fingerprints with NO trainable layer, the kernel reading the 512-d fingerprints directly. GP prior mean: learned during that fit, not pinned. float32.
Why this surrogate: it is the first in the study whose uncertainty is not collapsed -- 88% of truths within 2 std on unseen ampc_331k against 6% for the width-256 arm, and a GIBBON effective support of ~21 molecules rather than 2.
Acquisition: GIBBON (QLowerBoundMaxValueEntropy), log_space=true so the information gain never underflows, log_output=false so it is returned on the VALUE scale, which is what the reward transform needs. Max-value support is the 25,000 training molecules.
Reward: transform=${TRANSFORM}, beta=${BETA}. ${REWARD_TEXT}. The loss uses beta * score as log R, so R = exp(beta * score) is always positive even where the logged post-transform score is negative.
Normalizer: log_z STARTS AT ${INITIAL_LOG_Z}, not 0, with learning rate ${LOG_Z_LR} (default 1e-3) and no weight decay. log_z estimates log E[R] under the pretrained prior; ${INITIAL_LOG_Z} is that value as measured for this beta and floor by scripts/reward_shape_preview.py on gp_molformer_set. The 2026-10-08 power arms (jobs 5024309, 5024311, 5024318, 5024345) started it at 0, 9-23 nats too high, which told the policy every molecule it generated was far over-probable; validity fell to 0.4% and they took 60-429 of 10,000 updates. Watch sampler/s3gfn/log_z: it should stay near ${INITIAL_LOG_Z}, and a long drift means the measurement was off.
What to read: the per-training-step figures sampler/s3gfn/acq/trajectory (is the policy finding more informative molecules), reward/effective_support (how many of the 64 molecules in a batch the reward spreads over -- toward 1 means collapse, toward 64 means the policy will not move), reward/concentration, and train/batch_health (valid / synthesizable / unique). acq rising while validity or uniqueness falls is the policy fleeing the data the GP was fitted on. Four more test the log_z diagnosis directly: train/residual splits the RTB residual into its signed mean (log_z at the wrong level -- should sit near 0 if ${INITIAL_LOG_Z} was right) and its standard deviation (the reward shape the policy has yet to match); train/prior_divergence estimates KL(policy || prior), 0 for an inert policy and unbounded for one being torn off its prior; reward/above_floor_fraction is the share of each batch scoring above the ${LOG_SCORE_FLOOR}-nat floor, about 0.03 under the prior and the thing that should climb; train/gradient_norm is the gradient norm before clipping against the threshold of 10.
Then the round itself: 2,000 Dock3 evaluations selected by CostAwareSelector from a 100k generated pool, 10,000 S3-GFN training steps. The question the round answers is whether the true docking scores of those 2,000 beat a random draw from the same pool; expect low single digits, since GIBBON's top-1000 enrichment on generated-like molecules (gp_molformer_set) is 2.35x, not the 6.5x it reaches on the library.
Prior art: exponential beta=100 (job 4012959) was inert, mean log-reward 0.02 nats, validity 0.93; power beta=1 (job 4177912) collapsed, validity 0.0028 and zero yield from 2,000,000 attempts. Both ran on a variational surrogate, not this one.
Arm name: ${ARM}. Arms differ only in what the name carries: reward_<transform>_beta<beta>_logz<initial log_z>, plus _s<n> for a non-default seed.
Submitted with: sbatch jobs/reward_shape.sh $*   (Slurm job ${SLURM_JOB_ID:-unknown})"
printf '%s\n' "${WANDB_NOTES}" > "${OUTPUT_DIR}/arm_description.txt"

RUN_NAME="${ARM//_/-}-${SLURM_JOB_ID:-local}"
TAGS="reward-shape,${TRANSFORM},beta${BETA},logz${INITIAL_LOG_Z},floor${LOG_SCORE_FLOOR},logz-lr${LOG_Z_LR},nolayer,restored-state,fp32,1-round${SEED_TAG}"

# The whole logger is restated as one YAML value rather than overridden field by
# field. OmegaConf.from_dotlist reads every path segment as a dict key, so
# `logger.loggers.0.run_name=...` builds a dict under `loggers` and merging that
# onto the config's list dies with "Cannot merge DictConfig with ListConfig".
# Lists are replaced wholesale on merge, which is why job_10k_eager.sh and the
# reward_* overlays restate the logger too.
LOGGER="logger={type: MultiLogger, loggers: [{type: ConsoleLogger, project_name: ampc-reward-shape, run_name: ${RUN_NAME}}, {type: WandbLogger, project_name: ampc-reward-shape, run_name: ${RUN_NAME}, entity: models-mila5723, group: reward-shape, tags: [${TAGS}]}]}"

# scripts/run_with_sigterm_flush.py wraps activelearning.main so the SIGTERM
# Slurm sends at the wall clock becomes a clean shutdown, flushing buffered
# wandb and run-writer output instead of losing the round.
# --no-sync keeps uv from re-resolving dependencies over a network this node
# does not have.
uv run --no-sync python scripts/run_with_sigterm_flush.py \
  config/ampc/reward_shape.yaml \
  "sampler.reward_transform=${TRANSFORM}" \
  "sampler.beta=${BETA}" \
  "sampler.initial_log_z=${INITIAL_LOG_Z}" \
  "sampler.log_score_floor=${LOG_SCORE_FLOOR}" \
  "sampler.log_z_learning_rate=${LOG_Z_LR}" \
  "${SEED_OVERRIDE[@]}" \
  "run_writer.output_dir=${OUTPUT_DIR}" \
  "sampler.reject_log_path=${REJECT_LOG_PATH}" \
  "${LOGGER}" \
  "${@:5}"

echo
echo "Done. Sync from a login node with:"
echo "  wandb sync ${OUTPUT_DIR}/wandb/offline-run-*"
