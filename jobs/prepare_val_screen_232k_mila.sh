#!/bin/bash
# Encode the validation screen (ampc_val_screen_232k) on Mila: 230,887 MiniMol
# fingerprints into cache/ampc/exact_dkl_top_n/ampc_val_screen_232k.{npy,npy.json}
# plus the resolved CSV beside them.
#
#   sbatch jobs/prepare_val_screen_232k_mila.sh
#
# This is the ONLY encode the exact-GP sweep needs. The other two held-out sets
# are already cached and were verified on 2026-11-09 against the loader's own
# length-framed hash: olivier_invitro (1,521 x 512) and, if it is ever wanted
# back, train_random (100,000 x 512). gp_molformer_set is NOT prepared here: its
# source, data/gpmolformer_prior_100k_docked.csv, is not on this cluster yet.
#
# WHY THERE IS NO SHORTCUT. Every other set in this study was published out of
# the 20 GB fingerprint store without inference, because its molecules are in
# the 10M library. This one is a DISJOINT sample of the screen: 0 of its 230,887
# SMILES appear in cache/ampc/s3gfn_minimol_ampc_fingerprints.smiles.txt, which
# is exactly what makes it a genuine held-out set, and exactly why its features
# have to be computed. Measured, not assumed: the overlap check was run on
# 2026-11-09 and returned 0 (0.000%).
#
# THE LABELS come from data/ampc_val_screen_232k_labelled.csv, written by
# applying the fitted hit-rate curves to the ground-truth pprop column of
# data/ampc_val_screen_232k.csv:
#
#   y = HitRateModel.hit_rate_from_pprop(pprop, 6.5)
#
# Two provenance facts verified before that file was written, both on
# 2026-11-09. (1) Its pprop came from ampc_full_scores_1p7b.df: mapping its
# score column through that table reproduces the stored pprop to 1.8e-06, while
# the broken full_scores.df is off by up to 1.535. (2) The label route is the
# library's own: y = hit_rate_from_pprop(pprop_1p7b, 6.5) reproduces the 10M
# library's y with worst error 0.0e+00 over 3,000 rows. And as an independent
# check, the ipw-weighted mean y of the written file is 0.0042859221 against the
# library's 0.0042864189 -- 1.16e-04 relative, from two disjoint samples of the
# same screen.
#
# The `weight` column is ipw and is carried through by the set's SetSpec. It is
# not decoration: unweighted mean y is 0.1255 against a weighted 0.0042859, a
# ~29x tail enrichment, so EVERY population-level metric on this set must use it.
#
# RESOURCES. One A100, as requested. The encode is MiniMol inference over 230,887
# molecules; the features themselves are small (230,887 x 512 float32 = 473 MB).
# `--sets ampc_val_screen_232k` now also skips the 10M library read, which exists
# only to derive the train_* sets, so this job's resident set is the screen plus
# the model rather than 10M SMILES.
#
# MEASURED, job 11618802 on 2026-11-09: 7 min 46 s wall, 12.34 GB peak RSS of
# the 32 G requested, for all 230,887 molecules -- about 495 molecules/s. An
# estimate of 1-2 h, extrapolated from the multi-round loop's pool encode
# (MULTI_ROUND_AL_PLAN.md:150), was an order of magnitude too pessimistic.
#
# WHERE THE TIME GOES: not the GPU. Through the whole run the A100 sat at 0%
# utilization with a flat 479 MiB, while AveCPU reached 22 min over 7:46 wall,
# i.e. ~3 of the 4 cores busy. The work is CPU-side graph featurization. So the
# lever for a bigger encode is --cpus-per-task, not a better card; the GPU is
# needed only because the model is loaded onto it.
#
# The 6 h limit is left as it is because a timeout loses everything (below), and
# 32 G because the measured peak is 12.34 GB. RSS climbs roughly linearly while
# the run proceeds -- do not mistake that for a leak heading to OOM, as a reading
# of this job's first 8 minutes did.
#
# A TIMEOUT LOSES EVERYTHING, so do not trim that 6 h. The cache is NOT written
# incrementally: _publish_feature_cache builds the whole (230887, 512) tensor in
# memory, writes it to a temporary file and commits it atomically at the end
# (minimol_encoder.py:819-841), and the manifest is only ever written with
# "complete": true. There is no resumable partial state and no partial cache for
# a later run to extend.
#
# There is also NO PROGRESS OUTPUT: the encode logs one line, "<set>: encoded N
# molecules", AFTER it finishes (exact_dkl_prepare.py:600), and nothing in the
# encoder or its backend reports batch progress. To judge progress while a job
# runs, sample the batch step's RSS -- the features accumulate in memory, 473 MB
# over the full set -- and the GPU with:
#
#   sstat -j <jobid>.batch --format=MaxRSS -n
#   srun --jobid=<jobid> --overlap nvidia-smi \
#     --query-gpu=utilization.gpu,memory.used --format=csv,noheader
#
# `unkillable` is chosen over `main`: it is not preemptible, and its per-user
# QOS caps (1 GPU, 32 G) are not binding here. If it will not schedule, `main`
# with the same directives is the fallback and accepts preemption.
#SBATCH --partition=unkillable
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --job-name=val_screen_232k_prep
#SBATCH --output=slurm_logs/val_screen_232k_prep_%j.out
#SBATCH --error=slurm_logs/val_screen_232k_prep_%j.err

set -euo pipefail

cd "${HOME}/activelearning"

LABELLED="data/ampc_val_screen_232k_labelled.csv"

# Checked up front so a missing input fails in seconds rather than after the
# encoder has loaded.
for required in \
  "${LABELLED}" \
  minimol_ampc_encoder/model/final.pt \
  ampc_hitrate_fits/fitted_params.json
do
  if [[ ! -e "${required}" ]]; then
    echo "Missing ${required}." >&2
    exit 2
  fi
done

# The labelled file must carry both the target and the weights: without `weight`
# every population number on this set would silently describe the enriched
# sample only.
#
# The trailing CR is stripped before comparing. csv.writer defaults to \r\n and
# the prepared sets in cache/ are CRLF too, so the file is right and a guard
# that compared the raw line was not: it failed with "header is 'SMILES,y,weight',
# expected 'SMILES,y,weight'", the difference being invisible. (Job 11618775.)
HEADER="$(head -1 "${LABELLED}" | tr -d '\r')"
if [[ "${HEADER}" != "SMILES,y,weight" ]]; then
  echo "${LABELLED} header is '${HEADER}', expected 'SMILES,y,weight'." >&2
  exit 3
fi

# Refuse to silently rebuild an existing cache; the encode is hours.
for guarded in \
  cache/ampc/exact_dkl_top_n/ampc_val_screen_232k.npy \
  cache/ampc/exact_dkl_top_n/ampc_val_screen_232k.csv
do
  if [[ -e "${guarded}" ]]; then
    echo "${guarded} already exists. Pass --overwrite deliberately, or move it:" >&2
    echo "  mv ${guarded} ${guarded}.OLD" >&2
    exit 4
  fi
done

source .venv/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1   # compute nodes have no internet

mkdir -p slurm_logs cache/ampc/exact_dkl_top_n

nvidia-smi --query-gpu=name,memory.total --format=csv

# --skip-subsets: the training sets are not this job's business, and the
# exact-GP sweep's training set comes from build_svgp_scaling_sets.py instead.
uv run --no-sync python -m scripts.exact_dkl_prepare \
  --sets ampc_val_screen_232k \
  --val-screen-csv "${LABELLED}" \
  --cache-dir cache/ampc/exact_dkl_top_n \
  --encoder-device cuda \
  --skip-subsets \
  "$@"

echo
echo "Done. Verify the cache matches the CSV under the loader's own hash:"
echo "  python3 -c \"import csv,json,hashlib;"
echo "  d=hashlib.sha256();"
echo "  rows=list(csv.DictReader(open('cache/ampc/exact_dkl_top_n/ampc_val_screen_232k.csv')));"
echo "  [ (d.update(len(r['SMILES'].encode()).to_bytes(8,'big')), d.update(r['SMILES'].encode())) for r in rows ];"
echo "  j=json.load(open('cache/ampc/exact_dkl_top_n/ampc_val_screen_232k.npy.json'));"
echo "  print(d.hexdigest()==j['input_sha256'], j['row_count'], j['shape'])\""
