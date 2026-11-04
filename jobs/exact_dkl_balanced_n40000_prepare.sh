#!/bin/bash
# Build a 40,000-row library-proportional ("balanced") set, constructed exactly
# as data/ampc_balanced_25000.csv was, for a single-fidelity AL run that starts
# from a 1.6x archive. Run once, before that run.
#
#   sbatch jobs/exact_dkl_balanced_n40000_prepare.sh
#
# This is jobs/exact_dkl_balanced_prepare.sh at 40,000 instead of 25,000.
# WHY 40,000 AND NOT 50,000: measured, not chosen. Probe job 26162761 ran two
# rounds of the real loop at a 40,000-row GP and passed (fits 251 s and 284 s,
# round-2 peak 72.48 GiB of 79.18, 6.70 GiB spare). The same probe at 45,000
# (job 26162757) and at 50,000 (job 26159147) both died with CUDA OOM in round
# 2's fit. An exact GP's kernel matrix is n*n floats, so the fit peak is
# ~3.01e-8 * n^2 GiB: 48.6 at 40k, 61.3 at 45k, 75.5 at 50k against one H100's
# 79.18, and the fit does not have the device to itself. Everything
# that defines "balanced" is unchanged and is deliberately repeated here rather
# than referenced: --strat-quantiles 50,75,90,95,99,99.75, --strat-top-fraction
# 0.05, --strat-band-fill proportional, --strat-pool-multiple 40,40,40,15,15,15
# ,15 and --strat-seed 42. The band edges are quantiles of the 10M library, not
# of the selection, so they are identical at 25,000 and at 40,000: the top band
# is still floored at 5% of the budget and the lower bands still take their
# library-proportional share, which is what puts ~95% of the rows below the top
# band's cut and makes the set resemble the pool the policy actually samples.
#
# NOTHING FROM THE 25,000 BUILD IS OVERWRITTEN, and that needed a new label
# rather than a new size. scripts/exact_dkl_prepare.py names the pool cache
# `<label>_pool`, with no size in it, so re-running under --strat-label
# balanced would have replaced balanced_pool (910,650 rows) with this build's
# ~1.45M-row pool and the manifest's 'stratified_balanced' block with this one,
# leaving data/ampc_balanced_25000.csv in place but no longer reproducible.
# --strat-label balanced_n40000 namespaces every output instead:
#
#   data/ampc_balanced_n40000_40000.csv
#   cache/ampc/exact_dkl_top_n/balanced_n40000_40000.{npy,csv}
#   cache/ampc/exact_dkl_top_n/balanced_n40000_pool.{npy,csv}
#   the 'stratified_balanced_n40000' block of prep_manifest.json
#
# This follows the precedent of fps_keep05_n45000, which is the same situation.
#
# NOT NESTED IN THE 25,000 SET, and this is unavoidable, not an oversight.
# select_stratified_rows draws one candidate pool and runs one farthest-point
# pass per band, sized for the largest requested size, so a 40,000-only run
# draws ~1.45M pool molecules where the 25,000 run drew 910,650 and greedy FPS
# makes different choices. The resulting set is built the same way but is an
# independent selection: a run seeded from it differs from
# sf-10round-5000dock-5000steps-seed42 in archive size AND membership. Asking
# for 25000,40000 together would nest them, but largest-remainder rounding is
# not monotone in n and can trip the nested-subset guard, and the 25,000 it
# produced would not match the existing file either, so the baseline would have
# to be re-run. Accepted: not nested.
#
# One size only, for that same rounding reason.
#
# Cost. The per-band budgets scale by 1.6, so the pool goes from 910,650 to
# ~1,452,116 molecules -- band 6 excepted, whose nominal 30,000 exceeds the
# band's own 25,066 rows and is capped there, so that band selects from all of
# it. Per band the budgets become 19045, 9520, 5717, 1909, 1523, 286 and 2000.
#
# That pool is ENCODED, by real MiniMol inference -- exact_dkl_prepare builds a
# MiniMolAmpcSmilesFixedEncoder, not the feature store -- so the job is bound by
# GPU compute, not GPU memory. It needs only a few GB (the largest band's pool
# is 761,800 x 512 x 4 bytes = 1.56 GB), so a MIG slice on g[30-37] would FIT;
# it would just be slower, since those profiles are compute fractions of a card
# (3g.40gb is ~3/7, so ~75 min). A whole H100 is the quickest route while
# g[1-29] has idle capacity.
#
# The comparable fps_keep05_n45000 build, at a LARGER 1,610,733 pool, finished
# in 33 minutes on one h100. Its 64 G was NOT transferable, though: job
# 26167170 tried that here and was OOM-killed at 28 minutes with MaxRSS exactly
# at the 64 G cap, so this build's host-memory profile is heavier than that
# one's despite the smaller pool. The true peak is unknown -- a killed job only
# proves it exceeds the cap -- so the request below is the g-node's own share
# per GPU rather than a guess.
#
# WHY GENEROUS HOST MEMORY IS FREE HERE. A g[1-29] node is 112 CPUs, 2000 G and
# 8 H100s, and PriorityFlags=MAX_TRES bills the largest resource fraction, so
# one GPU bills 97600/8 = 12200 whatever else is asked for, up to 14 CPUs and
# 250 G. Trimming below that share buys nothing and risks exactly the OOM above.
# Walltime still matters and stays at 1 h.
#
# How many rows land at y >= 0.5 is NOT predicted here. All 513 of the
# library's hits sit in band 6, where FPS spreads in feature space rather than
# sampling y uniformly; the 25,000 set caught 9 of them from a 1,250 budget and
# a 2,000 budget does not scale that count linearly. Measure it from the
# written set, do not assume it.
# H100, not A100. g[1-29] is 29 nodes of 8 H100s; a100 on this cluster is the
# single node o1, and requesting it left job 26166493 pending behind "Nodes
# required for job are DOWN, DRAINED or reserved for jobs in higher priority
# partitions". Nothing here needs an A100: the GPU serves MiniMol feature
# lookups for the candidate pool, which the memcheck probes did on H100s.
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=1:00:00
#SBATCH --job-name=ampc_balanced_n40000_prep
#SBATCH --account=def-alexhg_gpu
#SBATCH --output=slurm_logs/ampc_balanced_n40000_prep_%j.out
#SBATCH --error=slurm_logs/ampc_balanced_n40000_prep_%j.err

set -euo pipefail

cd /home/ethankrz/activelearning
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1   # compute nodes have no internet

mkdir -p slurm_logs cache/ampc/exact_dkl_top_n

# --sets restricts the encoding loop to the new set, so the job does not
# re-check the caches the earlier prep runs already published. The candidate
# pool is encoded regardless: the selection cannot run without it.
uv run --no-sync python -m scripts.exact_dkl_prepare \
  --cache-dir cache/ampc/exact_dkl_top_n \
  --training-csv data/10M_unif_random_subset.csv \
  --val-csv data/ampc_val_20k.csv \
  --gp-molformer-csv data/gpmolformer_prior_100k_docked.csv \
  --ampc-331k-csv data/ampc_331k_with_y.csv \
  --olivier-csv data/Olivier_Invitro.csv \
  --train-sizes 2000 \
  --n-train-random 100000 \
  --n-train-top 10000 \
  --strat-sizes 40000 \
  --strat-quantiles 50,75,90,95,99,99.75 \
  --strat-top-fraction 0.05 \
  --strat-band-fill proportional \
  --strat-pool-multiple 40,40,40,15,15,15,15 \
  --strat-label balanced_n40000 \
  --strat-seed 42 \
  --eval-seed 42 \
  --sets balanced_n40000_40000 \
  --checkpoint-path minimol_ampc_encoder/model/final.pt \
  --package-path minimol_ampc_encoder \
  --encoder-device cuda \
  --encoder-batch-size 64 \
  "$@"

echo
echo "Band diagnostics, the farthest-point spread against a random draw, and the"
echo "evaluation-set overlap counts are under 'stratified_balanced_n40000' in"
echo "cache/ampc/exact_dkl_top_n/prep_manifest.json."
echo
echo "Check before launching any arm:"
echo "  - the band counts put ~95% of the 40000 below the top band"
echo "  - nn_ratio > 1 per band, i.e. the spreading beat a random draw"
echo "  - data/ampc_balanced_25000.csv is unchanged, and so are the"
echo "    balanced_pool and balanced_25000 caches and the manifest's"
echo "    'stratified_balanced' block"
echo "  - the y distribution of the new set, which is what decides whether the"
echo "    run still starts unenriched:"
echo "      awk -F, 'NR>1{n++; y=\$2+0; s+=y; if(y>=0.5)a++; if(y>=0.6)b++;"
echo "        if(y>=0.66)c++} END{printf \"rows=%d mean=%.5f y>=0.5:%d"
echo "        y>=0.6:%d y>=0.66:%d\\n\", n, s/n, a, b, c}' \\"
echo "        data/ampc_balanced_n40000_40000.csv"
