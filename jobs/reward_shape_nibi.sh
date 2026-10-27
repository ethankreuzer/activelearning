#!/bin/bash
# Nibi port of jobs/reward_shape.sh: one full reward-shape arm (vargrad, fixed
# floor -150, restored exact-GP surrogate, GIBBON, 10,000 Dock3 evaluations).
# Identical to the Narval script apart from the resources, the account, the
# Nibi oracle overlay (receptor files, dockenv_nibi.sh, 2022 dock64) and the
# output root outputs/ampc/reward_shape_nibi/. See jobs/reward_shape.sh for the
# full rationale of every fixed setting.
#
#   sbatch jobs/reward_shape_nibi.sh <exponential|power> <beta> <initial_log_z> <n_train_steps> [seed]
#
# First use, 2026-10-16: power beta=0.25 at 5000 steps, to compare against
# Narval's power beta=0.1 at 5000 steps.
#
# One H100 bundle: a Nibi GPU node is 8 x H100, 112 cores, 2000G, so one GPU's
# share is 14 cores and 250G; more cores would bill as more than one GPU.
# num_workers: null sizes the Dock3 pool to those 14 CPUs, under the 48-worker
# cap docking needs on Nibi.
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=250G
# Measured on Nibi by the sanity arms: training 0.89 s/step (job 24057277,
# 1500 steps in 1331 s), generation ~4.5 min, selection ~3.5 min, docking
# 1.27 s/molecule wall on 14 workers (job 24059495, 500 in 635 s). At 5000
# steps and 10,000 dockings that is ~74 min + ~8 min + ~3.5 h = ~5 h; 8 h
# leaves margin without asking for 10. SURROGATE=fixedgp_iso has not been
# timed; its GP is the same size as the nolayer arm's, so 8 h should hold.
#SBATCH --time=8:00:00
#SBATCH --job-name=ampc_reward_shape_nibi
# Best on 2026-10-16 by `sshare -l -A def-bengioy_gpu,def-mailhoto_gpu,def-yvesbrun_gpu`:
# account LevelFS 10.4 (bengioy 0.09, mailhoto 0.44), user FairShare 0.347.
# Re-check before every submit.
#SBATCH --account=def-yvesbrun_gpu
#SBATCH --output=slurm_logs/ampc_reward_shape_nibi_%j.out
#SBATCH --error=slurm_logs/ampc_reward_shape_nibi_%j.err

set -euo pipefail

TRANSFORM="${1:-}"
BETA="${2:-}"
INITIAL_LOG_Z="${3:-}"
# Training steps became an experimental variable on 2026-10-14, so it is a
# positional argument and goes in the arm name beside the transform and beta.
# It sits BEFORE the seed, which was argument 4 until that date; the minimum of
# 100 below is what stops an old `... <initial_log_z> <seed>` call from silently
# training for `seed` steps.
N_TRAIN_STEPS="${4:-}"
SEED="${5:-}"
if [[ ! "${TRANSFORM}" =~ ^(exponential|power)$ \
      || ! "${BETA}" =~ ^[0-9]*\.?[0-9]+$ \
      || ! "${INITIAL_LOG_Z}" =~ ^-?[0-9]*\.?[0-9]+$ \
      || ! "${N_TRAIN_STEPS}" =~ ^[0-9]+$ \
      || "${N_TRAIN_STEPS}" -lt 100 \
      || ! "${SEED}" =~ ^[0-9]*$ ]]; then
  echo "usage: sbatch jobs/reward_shape_nibi.sh <exponential|power> <beta> <initial_log_z> <n_train_steps> [seed]" >&2
  echo "n_train_steps is an integer of at least 100; the seed moved to argument 5 on 2026-10-14." >&2
  exit 2
fi

# Fixed for the whole study, and stated here rather than left to the YAML so the
# notes and tags below cannot drift from what the run was given.
#   LOG_SCORE_FLOOR  lower bound on log(acq) under power; a constant, so a
#                    molecule's reward no longer depends on its batch. Lowered
#                    from -100 on 2026-10-13: only ~3% of a prior batch cleared
#                    -100, which left the reward near two-valued.
#   LOSS             vargrad drops log_z and minimises the variance of the RTB
#                    residual, taking the batch's own optimal normalizer. That
#                    removes the failure the 2026-10-12 arms hit, where a
#                    badly placed log_z suppressed the valid molecules and so
#                    starved itself of the updates that would have corrected
#                    it. INITIAL_LOG_Z and LOG_Z_LR are then unused, and kept
#                    only so switching back to rtb needs no edit here.
#   LOG_Z_LR         learning rate of the log_z scalar (the policy's is 1e-4).
#                    Overridable per arm since 2026-10-27: the variational GP's
#                    log scores run one to two orders of magnitude lower than
#                    the exact GPs', so -150 would tie every prior-like molecule.
#                      LOG_SCORE_FLOOR=-500 SURROGATE=svgp300k sbatch ...
LOG_SCORE_FLOOR="${LOG_SCORE_FLOOR:--150}"
if [[ ! "${LOG_SCORE_FLOOR}" =~ ^-[0-9]+(\.[0-9]+)?$ ]]; then
  echo "LOG_SCORE_FLOOR must be a negative number, e.g. -500." >&2
  exit 2
fi
LOSS="vargrad"
LOG_Z_LR="0.01"

# The arm name carries only the non-defaults, as the exact-DKL arms do:
#   reward_<transform>_beta<beta>_<vargrad|logz<initial_log_z>>_floor<floor>_steps<n>[_s<seed>]
# Under vargrad the log_z value is not used, so naming the arm after it would
# claim a setting the run ignored; the loss goes in its place. The floor is in
# the name because it changed on 2026-10-13 and the -100 arms must stay in
# their own directories. The step count is in the name because the 2026-10-14
# grid varies it against beta, so two arms otherwise identical must not share a
# directory -- and so an arm can be identified from its name alone.
if [[ "${LOSS}" == "vargrad" ]]; then
  ARM="reward_${TRANSFORM}_beta${BETA}_vargrad_floor${LOG_SCORE_FLOOR}"
else
  ARM="reward_${TRANSFORM}_beta${BETA}_logz${INITIAL_LOG_Z}_floor${LOG_SCORE_FLOOR}"
fi
ARM="${ARM}_steps${N_TRAIN_STEPS}"
SEED_OVERRIDE=()
SEED_TAG=""
if [[ -n "${SEED}" ]]; then
  ARM="${ARM}_s${SEED}"
  SEED_OVERRIDE=("runtime.seed=${SEED}")
  SEED_TAG=",seed${SEED}"
fi
# Which restored surrogate the arm trains against. nolayer (the default) is the
# DKL no-layer state every earlier arm used; fixedgp_iso is the isotropic
# fixed-feature GP (config/ampc/reward_shape_fixedgp_iso.yaml), identical in
# every other setting. fixedgp_fps_keep05 is the isotropic Matern GP (output
# scale, pinned mean) fitted to the FPS + keep-y>=0.5 training set
# (config/ampc/reward_shape_fixedgp_fps_keep05.yaml). Like ARM_SUFFIX it comes
# from the environment, and it sets the config, the state, the directory, the
# tags and the notes together:
#   SURROGATE=fixedgp_iso sbatch jobs/reward_shape_nibi.sh power 0.1 1000 5000
SURROGATE="${SURROGATE:-nolayer}"
# The W&B project. Arms on the balanced training set share ampc-reward-shape;
# an arm on a different training set gets its own project, so the datasets
# never mix in one runs table.
WANDB_PROJECT="ampc-reward-shape"
if [[ "${SURROGATE}" == nolayer ]]; then
  BASE_CONFIG="config/ampc/reward_shape.yaml"
  FITTED_DIR="outputs/ampc/exact_dkl_top_n/balanced_25000_nolayer"
  SURROGATE_TAGS="nolayer,fp32"
elif [[ "${SURROGATE}" == fixedgp_iso ]]; then
  BASE_CONFIG="config/ampc/reward_shape_fixedgp_iso.yaml"
  FITTED_DIR="outputs/ampc/exact_dkl_top_n/balanced_25000_fixedgp_iso_fp32"
  SURROGATE_TAGS="fixedgp-iso,fp32"
  # Only the non-default surrogate goes in the name, so the nolayer arms keep
  # the directories they already have. _fp32 because a float64 arm of this name
  # was started and cancelled; its directory holds no pool.
  ARM="${ARM}_fixedgp_iso_fp32"
elif [[ "${SURROGATE}" == fixedgp_fps_keep05 ]]; then
  BASE_CONFIG="config/ampc/reward_shape_fixedgp_fps_keep05.yaml"
  FITTED_DIR="outputs/ampc/exact_dkl_top_n/fps_keep05_25000_fixedgp_iso_matern_scaled_pm0.03956_fp32"
  SURROGATE_TAGS="fixedgp-iso-matern,outputscale,pinned-mean,fps-keep05,fp32"
  ARM="${ARM}_fixedgp_fps_keep05_fp32"
  WANDB_PROJECT="ampc-reward-shape-fps-keep-above"
elif [[ "${SURROGATE}" == svgp300k ]]; then
  # The sparse variational GP the scaling study settled on, fitted on Mila and
  # copied here (config/ampc/reward_shape_svgp300k.yaml).
  BASE_CONFIG="config/ampc/reward_shape_svgp300k.yaml"
  FITTED_DIR="outputs/ampc/variational_gp/svgpscale_300000_svgp_M1024_iso_matern_scaled_pm0.03956_elbo_cosine_lr3e-3_steps10064_fp32"
  SURROGATE_TAGS="svgp,M1024,elbo,svgpscale-300k,lr3e-3,steps10064,pinned-mean,fp32"
  ARM="${ARM}_svgp300k_m1024_fp32"
  WANDB_PROJECT="ampc-reward-shape-svgp"
else
  echo "SURROGATE must be nolayer, fixedgp_iso, fixedgp_fps_keep05 or svgp300k." >&2
  exit 2
fi
# Optional suffix for rerunning an arm whose settings are unchanged, so the
# rerun gets its own directory instead of overwriting the earlier run's pool:
#   ARM_SUFFIX=_chunk64 sbatch jobs/reward_shape_nibi.sh power 0.1 1000 10000
# sbatch exports the submitting shell's environment, so this reaches the job.
ARM_SUFFIX="${ARM_SUFFIX:-}"
if [[ -n "${ARM_SUFFIX}" && ! "${ARM_SUFFIX}" =~ ^_[A-Za-z0-9.-]+$ ]]; then
  echo "ARM_SUFFIX must look like _name (letters, digits, '.', '-')." >&2
  exit 2
fi
ARM="${ARM}${ARM_SUFFIX}"
# Built here rather than inline in the notes below: an apostrophe inside a
# ${VAR:+...} expansion makes bash look for a matching quote and the script
# fails to parse.
SUFFIX_TEXT=""
if [[ -n "${ARM_SUFFIX}" ]]; then
  SUFFIX_TEXT=" Arm suffix ${ARM_SUFFIX}: a rerun under unchanged settings, kept in its own directory so the earlier outputs survive."
fi

cd /home/ethankrz/activelearning
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1   # compute nodes have no internet
# wandb.init() defaults to online and blocks on api.wandb.ai, which this node
# cannot reach. Offline writes to ${WANDB_DIR}; sync from a login node after.
export WANDB_MODE=offline

STATE_PATH="${FITTED_DIR}/surrogate_state.pt"
OUTPUT_DIR="outputs/ampc/reward_shape_nibi/${ARM}"
# Why validity is lost, not just how much: every discarded generation is counted
# by reason, and a reservoir sample of them is written here. The `{round}`
# placeholder must not start the value -- OmegaConf runs the dotlist through a
# YAML loader, which would read a leading `{` as a flow mapping. Prefixed by
# OUTPUT_DIR it parses as a plain string.
REJECT_LOG_PATH="${OUTPUT_DIR}/rejects_round_{round}.csv"
# The generated pool, written the moment generation finishes rather than when
# the round completes, so an oracle timeout cannot destroy hours of sampling.
# It is also the input to scoring this pool against the prior's molecules.
POOL_LOG_PATH="${OUTPUT_DIR}/generated_pool_round_{round}.csv"
if [[ ! -f "${STATE_PATH}" ]]; then
  echo "${STATE_PATH} not found: fit that surrogate first, with" >&2
  echo "  sbatch jobs/exact_dkl_balanced.sh 25000 none        (nolayer)" >&2
  echo "  KERNEL=iso sbatch jobs/fixed_feature_gp_balanced_nibi.sh  (fixedgp_iso)" >&2
  echo "  TRAIN_SET=fps_keep05 PRECISION=32 KERNEL=iso FAMILY=matern OUTPUTSCALE=1 \\" >&2
  echo "    PRIOR_MEAN=0.03956042999405787 sbatch jobs/fixed_feature_gp_balanced_nibi.sh  (fixedgp_fps_keep05)" >&2
  echo "  (svgp300k: fitted on Mila by LR=3e-3 STEPS=10000 jobs/variational_gp_scaling_mila.sh elbo 300000; copy its surrogate_state.pt here)" >&2
  exit 2
fi
# Never overwrite an earlier run's generated pool: it is hours of sampling, and
# the input any _scored/ results were computed from. Rerun under ARM_SUFFIX.
if compgen -G "${OUTPUT_DIR}/generated_pool_round_*.csv" > /dev/null; then
  echo "${OUTPUT_DIR} already holds a generated pool from an earlier run." >&2
  echo "Rerun with a suffix, e.g. ARM_SUFFIX=_rerun2 sbatch jobs/reward_shape_nibi.sh $*" >&2
  exit 2
fi

# Per-arm, so concurrent arms never race on a shared ./wandb/latest-run symlink.
export WANDB_DIR="${OUTPUT_DIR}"
mkdir -p "${WANDB_DIR}" slurm_logs

TRAIN_SET_TEXT="the 25000-molecule BALANCED training set data/ampc_balanced_25000.csv -- the library-proportional mix, top band floored at 5% of the budget --"
# Set by the svgp300k branch; the exact-GP wording is filled in after the
# chain, once TRAIN_SET_TEXT has its final value.
RESTORE_TEXT=""
SUPPORT_TEXT=""
if [[ "${SURROGATE}" == svgp300k ]]; then
  TRAIN_SET_TEXT="the 300,000-molecule NESTED scaling set data/ampc_svgpscale_300000.csv -- every 10M row above y 0.2617 (pProp 3.5) kept whole (16,292), the rest split evenly across four bands by farthest-point sampling --"
  RESTORE_TEXT="The sparse variational GP is rebuilt on ${TRAIN_SET_TEXT} and its fitted parameters -- 1024 inducing points and their variational distribution, kernel, noise, input scaling and output standardization -- are loaded from ${STATE_PATH} in place of training."
  SUPPORT_TEXT="Max-value support is capped at 100,000 of the 300,000 training molecules by the deterministic fidelity/target-stratified reservoir, as in the scaling study; candidates are scored 256 per chunk."
  SURROGATE_TEXT="SURROGATE: the SPARSE VARIATIONAL GP (VariationalGPSurrogate) the training-set scaling study settled on (W&B ampc-Variational-GP): 1024 inducing points started at random training rows, Matern-5/2 with ONE lengthscale under BoTorch's dimension-scaled prior, learned output scale, LogNormal(-4, 1) noise prior, mean PINNED at 0.03956, inputs min-max scaled, float32; fitted by Adam on the ELBO, 4096-row minibatches, 10,064 steps cosine-annealed from 3e-3 to 1e-4.
Why this surrogate: 300k rows fixed the low-y overprediction of smaller sets (unseen ampc_331k bias +0.061, RMSE 0.068; gp_molformer_set RMSE 0.023) with GIBBON's ranking unchanged, at the cost of an overconfident tail (gentail 2-std coverage 54%, bias -0.089). It is far more confident than the exact GPs, so log GIBBON runs much lower (gp_molformer_set median -2,688, 99th percentile -1,146): the floor of ${LOG_SCORE_FLOOR} is set for that range, not the exact GPs' -150."
elif [[ "${SURROGATE}" == nolayer ]]; then
  SURROGATE_TEXT="Features: frozen MiniMol AmpC fingerprints with NO trainable layer, the kernel reading the 512-d fingerprints directly. GP prior mean: learned during that fit, not pinned. float32.
Why this surrogate: it is the first in the study whose uncertainty is not collapsed -- 88% of truths within 2 std on unseen ampc_331k against 6% for the width-256 arm, and a GIBBON effective support of ~21 molecules rather than 2."
elif [[ "${SURROGATE}" == fixedgp_fps_keep05 ]]; then
  TRAIN_SET_TEXT="the 25000-molecule training set data/ampc_fps_keep05_25000.csv -- every 10M row at y >= 0.5 kept outright, the rest by the balanced set's stratified farthest-point selection (library quantile bands, top band [0.232, 0.5) at 5% of the budget, lower bands proportional to the library) --"
  SURROGATE_TEXT="SURROGATE: the ISOTROPIC fixed-feature GP (FixedFeatureGPSurrogate, ard=false) with a Matern-5/2 kernel, a learned output scale and the constant mean PINNED at the 10M library mean 0.03956, fitted to the FPS + keep-y>=0.5 training set (config/ampc/reward_shape_fixedgp_fps_keep05.yaml); every other setting is that of the nolayer arm with the same name. BoTorch's SingleTaskGP on the same frozen 512-d MiniMol AmpC fingerprints, min-max scaled to [0, 1], fitted to convergence by L-BFGS-B in float32.
Why this surrogate and training set: stratified farthest-point selection is what keeps this GP calibrated (realistic noise and lengthscale), and forcing in every library row at y >= 0.5 improved ranking on every evaluation set at the cost of a small library-wide upward bias (analyze_archive_diversity/ARCHIVE_DIVERSITY_ANALYSIS.md, step 4a). This arm tests it as the reward before the multi-round loop uses it."
else
  SURROGATE_TEXT="SURROGATE: the ISOTROPIC fixed-feature GP (FixedFeatureGPSurrogate, ard=false), not the DKL no-layer surrogate the earlier arms used; every other setting is that of the nolayer arm with the same name (config/ampc/reward_shape_fixedgp_iso.yaml). BoTorch's SingleTaskGP on the same frozen 512-d MiniMol AmpC fingerprints, min-max scaled to [0, 1], with ONE RBF lengthscale shared by all features under BoTorch's dimension-scaled LogNormal lengthscale prior and the LogNormal(-4, 1) noise prior, fitted to convergence by L-BFGS-B in 80 s in float32 (lengthscale 2.67, noise 0.27 standardized; a float64 fit agrees to four decimals in every hyperparameter and held-out metric). float32, like the nolayer arms.
Why this surrogate: it is the first one fitted to convergence under priors. Offline it matched or beat the DKL no-layer surrogate on the mean (Pearson 0.70 vs 0.68 on unseen ampc_331k, 0.59 vs 0.59 on gp_molformer_set), with a similar latent std level, but its std ranked errors less well on gp_molformer_set (Spearman 0.17 vs 0.22). This arm tests which of the two makes the better reward."
fi

if [[ -z "${RESTORE_TEXT}" ]]; then
  RESTORE_TEXT="The exact GP (no inducing points, no minibatching) is rebuilt on ${TRAIN_SET_TEXT} and its fitted parameters are loaded from ${STATE_PATH} in place of training, so every arm of this study shares one surrogate exactly."
  SUPPORT_TEXT="Max-value support is the 25,000 training molecules."
fi

if [[ "${TRANSFORM}" == power ]]; then
  REWARD_TEXT="R = acq ** ${BETA}, so the reward RATIO between two molecules follows the ratio of their information gains. The transform takes the logarithm itself, which is why the acquisition stays on the value scale. log(acq) is floored at the FIXED value ${LOG_SCORE_FLOOR} nats, the same for every batch, so a molecule's reward depends on that molecule alone; everything below the floor is tied. Earlier arms floored 20 nats below each batch's own best instead, which made the reward level swing from batch to batch"
else
  REWARD_TEXT="R = exp(${BETA} * acq), so the reward ratio follows the DIFFERENCE of the information gains. Generated molecules score ~1e-14 at the 99.9th percentile against a best of ~4e-3, so the bulk of a batch is expected to stay tied at this or any beta; this arm exists to demonstrate that limit"
fi

if [[ "${LOSS}" == "vargrad" ]]; then
  NORMALIZER_TEXT="Loss: VARGRAD, not RTB. There is no log_z parameter. The log_z minimising the mean squared RTB residual is just the negative batch mean of the rest, so substituting it back leaves the VARIANCE of (log p_policy - log p_prior - beta * r). An offset shared by the whole batch therefore contributes no gradient at all, which is the point: the 2026-10-08 and the logz0 arm of 2026-10-12 died because a log_z at the wrong level pushed down exactly the valid, synthesizable molecules -- the only ones the loss can see -- and the mass went to invalid strings, which in turn starved log_z of the updates that would have corrected it (348 of 10,000). VarGrad cannot enter that loop. What it gives up is that the signal is purely within-batch: a batch with one usable molecule has zero variance and no gradient, and a near-constant reward carries little. INITIAL_LOG_Z (${INITIAL_LOG_Z}) and LOG_Z_LR (${LOG_Z_LR}) are passed but UNUSED."
  RESIDUAL_TEXT="train/residual is measured against the normalizer each batch implied, so its mean sits at 0 by construction and only the standard deviation -- the reward shape the policy has yet to match -- carries information; sampler/s3gfn/log_z is that implied normalizer per step, which is also the value INITIAL_LOG_Z should take if this is switched back to rtb"
else
  NORMALIZER_TEXT="Loss: RTB. Normalizer log_z STARTS AT ${INITIAL_LOG_Z}, not 0, with learning rate ${LOG_Z_LR} (default 1e-3) and no weight decay. log_z estimates log E[R] under the pretrained prior; ${INITIAL_LOG_Z} is that value as measured for this beta and floor by scripts/reward_shape_preview.py on gp_molformer_set. The 2026-10-08 power arms (jobs 5024309, 5024311, 5024318, 5024345) started it at 0, 9-23 nats too high, which told the policy every molecule it generated was far over-probable; validity fell to 0.4% and they took 60-429 of 10,000 updates. Watch sampler/s3gfn/log_z: it should stay near ${INITIAL_LOG_Z}, and a long drift means the measurement was off."
  RESIDUAL_TEXT="train/residual splits the RTB residual into its signed mean (log_z at the wrong level -- should sit near 0 if ${INITIAL_LOG_Z} was right) and its standard deviation (the reward shape the policy has yet to match)"
fi

# A plain-language description, so the run can be understood on its own months
# later. wandb reads WANDB_NOTES at init and shows it on the run's Overview page
# and in the Notes column of the runs table; the same text is kept next to the
# run's outputs.
export WANDB_NOTES="REWARD-SHAPE arm ${ARM}: ONE active-learning round, training the S3-GFN policy against a fixed GIBBON acquisition on a frozen surrogate.
THE QUESTION THIS RUN ASKS: does the trained policy generate molecules the acquisition rates higher than the ones the untrained prior produces? Not validity, not docking -- those are read but are not what this arm is for. The answer comes from scoring BOTH pools afterwards with this same restored surrogate and ONE acquisition object, and comparing two distributions on each: the surrogate posterior MEAN, and log(acquisition). The generated pool is written to ${POOL_LOG_PATH} as soon as generation finishes, then prepared and scored alongside gp_molformer_set (83,811 docked prior samples) via exact_dkl_prepare.py --extra-set and exact_dkl_top_n.py --extra-set, which also reports pooled/acquisition/*/top{k}/<set>_share: which of the two pools the acquisition would actually pick from. Both sets must be scored in the SAME invocation, because GIBBON ranks against sampled posterior max-values and a separate run would draw different ones.
WHAT IS NEW HERE, against the 2026-10-12 arms: the loss (vargrad, below) and the log-score floor (-150, was -100). Selection scoring now runs in chunks of 64 (acquisition.score_chunk_size, was 256). Every earlier arm, including the first vargrad floor -150 run (job 5493805), died at selection with 'Tried to allocate 12.21 GiB': gpytorch's exact-GP posterior concatenates the 25,000 training molecules onto every candidate in a chunk, so memory scales with the chunk size, and the earlier kernel-side fix never applied to that path. At 64 the same path already ran every S3-GFN training step, and job 5515579 scored a 100k pool with it at ~17 GiB peak. This is therefore the first arm expected to reach docking.${SUFFIX_TEXT}
Surrogate: RESTORED, NOT REFITTED. ${RESTORE_TEXT} See ${FITTED_DIR}/arm_description.txt for how that fit was obtained. ${SURROGATE_TEXT}
Acquisition: GIBBON (QLowerBoundMaxValueEntropy), log_space=true so the information gain never underflows, log_output=false so it is returned on the VALUE scale, which is what the reward transform needs. ${SUPPORT_TEXT}
Reward: transform=${TRANSFORM}, beta=${BETA}. ${REWARD_TEXT}. The loss uses beta * score as log R, so R = exp(beta * score) is always positive even where the logged post-transform score is negative.
${NORMALIZER_TEXT}
What to read: the per-training-step figures sampler/s3gfn/acq/trajectory (is the policy finding more informative molecules), reward/effective_support (how many of the 64 molecules in a batch the reward spreads over -- toward 1 means collapse, toward 64 means the policy will not move), reward/concentration, and train/batch_health (valid / synthesizable / unique). acq rising while validity or uniqueness falls is the policy fleeing the data the GP was fitted on. Four more: ${RESIDUAL_TEXT}; train/prior_divergence estimates KL(policy || prior), 0 for an inert policy and unbounded for one being torn off its prior; reward/above_floor_fraction is the share of each batch scoring above the ${LOG_SCORE_FLOOR}-nat floor, about 0.03 under the prior and the thing that should climb; train/gradient_norm is the gradient norm before clipping against the threshold of 10.
Then the round itself: ${N_TRAIN_STEPS} S3-GFN training steps, then 10,000 Dock3 evaluations -- the top 10,000 by acquisition, since CostAwareSelector ranks the 100k pool by acquisition over a flat cost of 1.0. Raised from 2,000 on 2026-10-14 so the docking score, the surrogate mean and the acquisition can be related on the same molecules rather than on a thin tail; at ~32 core-seconds each that is ~1.9 h on Narval's 48 CPUs; on this Nibi job's 14 CPUs ~3.5 h, from the 500-docking sanity arm. The question the round answers is whether the true docking scores of those 10,000 beat a random draw from the same pool; expect low single digits, since GIBBON's top-1000 enrichment on generated-like molecules (gp_molformer_set) is 2.35x, not the 6.5x it reaches on the library.
TRAINING LENGTH: this arm runs ${N_TRAIN_STEPS} steps. A shorter arm is not a longer one stopped early -- the warmup/decay schedule spans n_train_steps, so 5,000 steps decays the learning rate twice as fast as 10,000. On job 5493805 (10,000 steps) the above-floor fraction was flat by ~1,500 steps, KL from the prior by ~1,000, and the implied log Z gained ~0.2 nats after 4,000; only the residual standard deviation was still falling at 10,000. The 2026-10-14 grid crosses this against beta to find out whether the shorter schedule reaches the same place.
Per-run outputs for comparing the three quantities on the SAME molecules: artifacts/acquisition/general/round_NNNN/scored_samples.csv carries every pooled molecule with its acquisition score, whether it was selected, and -- for the selected -- the observed probability of binding and the raw DOCK3 score from the observation's metadata (the probability is not monotone in it, so it cannot be recovered afterwards). jobs/reward_shape_score_pool.sh then joins that against the surrogate mean it computes for the whole pool.
Prior art: exponential beta=100 (job 4012959) was inert, mean log-reward 0.02 nats, validity 0.93; power beta=1 (job 4177912) collapsed, validity 0.0028 and zero yield from 2,000,000 attempts. Both ran on a variational surrogate, not this one.
Arm name: ${ARM}. Arms differ only in what the name carries: reward_<transform>_beta<beta>_<vargrad|logz<initial log_z>>_floor<floor>, plus _s<n> for a non-default seed.
Submitted with: sbatch jobs/reward_shape_nibi.sh $*   (Slurm job ${SLURM_JOB_ID:-unknown})"
printf '%s\n' "${WANDB_NOTES}" > "${OUTPUT_DIR}/arm_description.txt"

RUN_NAME="${ARM//_/-}-${SLURM_JOB_ID:-local}"
# Only tag log_z settings when the loss actually reads them, so a tag search
# never turns up an arm that ignored the value it is tagged with.
if [[ "${LOSS}" == "vargrad" ]]; then
  LOSS_TAGS="vargrad,chunk64,acq-vs-prior"
else
  LOSS_TAGS="rtb,logz${INITIAL_LOG_Z},logz-lr${LOG_Z_LR}"
fi
TAGS="reward-shape,${TRANSFORM},beta${BETA},${LOSS_TAGS},floor${LOG_SCORE_FLOOR},steps${N_TRAIN_STEPS},${SURROGATE_TAGS},restored-state,1-round${SEED_TAG}${ARM_SUFFIX:+,suffix${ARM_SUFFIX}}"

# The whole logger is restated as one YAML value rather than overridden field by
# field. OmegaConf.from_dotlist reads every path segment as a dict key, so
# `logger.loggers.0.run_name=...` builds a dict under `loggers` and merging that
# onto the config's list dies with "Cannot merge DictConfig with ListConfig".
# Lists are replaced wholesale on merge, which is why job_10k_eager.sh and the
# reward_* overlays restate the logger too.
LOGGER="logger={type: MultiLogger, loggers: [{type: ConsoleLogger, project_name: ${WANDB_PROJECT}, run_name: ${RUN_NAME}}, {type: WandbLogger, project_name: ${WANDB_PROJECT}, run_name: ${RUN_NAME}, entity: models-mila5723, group: reward-shape, tags: [nibi,${TAGS}]}]}"

# scripts/run_with_sigterm_flush.py wraps activelearning.main so the SIGTERM
# Slurm sends at the wall clock becomes a clean shutdown, flushing buffered
# wandb and run-writer output instead of losing the round.
# --no-sync keeps uv from re-resolving dependencies over a network this node
# does not have.
uv run --no-sync python scripts/run_with_sigterm_flush.py \
  "${BASE_CONFIG}" \
  config/ampc/overrides/nibi_single_fidelity.yaml \
  "sampler.reward_transform=${TRANSFORM}" \
  "sampler.beta=${BETA}" \
  "sampler.n_train_steps=${N_TRAIN_STEPS}" \
  "sampler.initial_log_z=${INITIAL_LOG_Z}" \
  "sampler.loss=${LOSS}" \
  "sampler.log_score_floor=${LOG_SCORE_FLOOR}" \
  "sampler.log_z_learning_rate=${LOG_Z_LR}" \
  "${SEED_OVERRIDE[@]}" \
  "run_writer.output_dir=${OUTPUT_DIR}" \
  "sampler.reject_log_path=${REJECT_LOG_PATH}" \
  "sampler.pool_log_path=${POOL_LOG_PATH}" \
  "${LOGGER}" \
  "${@:5}"

echo
echo "Done. Sync from a login node with:"
echo "  wandb sync ${OUTPUT_DIR}/wandb/offline-run-*"
