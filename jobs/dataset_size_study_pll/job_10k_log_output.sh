#!/bin/bash
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
# Fitting on 10k molecules is small; most of the time goes to live MiniMol
# encoding of the ~333k evaluation SMILES, which runs twice (once for the
# surrogate predictions, once for the acquisition scores).
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --job-name=ampc_size_pll_log_output_10k
#SBATCH --account=def-yvesbrun_gpu
#SBATCH --output=slurm_logs/ampc_size_pll_log_output_10k_%j.out
#SBATCH --error=slurm_logs/ampc_size_pll_log_output_10k_%j.err
cd /home/ethankrz/activelearning

source .venv/bin/activate

# Python block-buffers stdout when it is redirected to a file; flush progress
# lines as they happen instead.
export PYTHONUNBUFFERED=1

# Compute nodes are offline; keep transformers/HF from stalling on hub checks.
export HF_HUB_OFFLINE=1

# Fit the surrogate on the 10k subset, then score the 331k AmpC subset and the
# Olivier in-vitro set. Outputs: outputs/ampc/dataset_size_study_pll_log_output/10k/.
#
# The base config's persistent feature cache is bound to the 10M CSV's row
# order, so run without it (as in config/ampc/overrides/10k_eager_10000steps.yaml)
# to make sure this job can never write a 10k-row cache into that path.
# Compute nodes are offline: sync from a login node first and use --no-sync.
# Trains the variational GP with PredictiveLogLikelihood instead of the ELBO; set
# explicitly so the run doesn't depend on the base config's current value.
# log_space with log_output: scores are log(information gain), negative and
# underflow-free, as in the base config.
uv run --no-sync python scripts/surrogate_dataset_size_study.py \
  config/ampc/s3gfn_minimol_ampc_variational_single_fidelity.yaml \
  dataset.initial_data.path=data/10k_unif_random_subset.csv \
  surrogate.encoder.feature_cache_path=null \
  surrogate.training_params.variational_objective=PredictiveLogLikelihood \
  acquisition.type=QLowerBoundMaxValueEntropy \
  acquisition.log_space=true \
  acquisition.log_output=true \
  --train-label 10k \
  --output-dir outputs/ampc/dataset_size_study_pll_log_output \
  --eval-csv ampc_331k=data/ampc_subset_331k.csv \
  --eval-csv olivier_invitro=data/Olivier_Invitro.csv
