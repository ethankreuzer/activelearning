#!/bin/bash
# Mila copy of jobs/exact_dkl_balanced_n40000_prepare.sh: build the 40,000-row
# library-proportional ("balanced") set, constructed exactly as
# data/ampc_balanced_25000.csv was. No docking, so nothing ties this to Nibi.
#
#   sbatch jobs/exact_dkl_balanced_n40000_prepare_mila.sh
#
# Read the Nibi script's header for WHY: why 40,000 and not 50,000 (measured --
# probe jobs 26162761 passed, 26162757 and 26159147 died of CUDA OOM in round
# 2's fit), why the label must change so nothing from the 25,000 build is
# overwritten, why the set is not nested in the 25,000 one, and the per-band
# budgets. None of that differs by cluster; only the resources below do.
#
# WHAT MUST BE ON MILA FIRST. The selection picks points by distance in MiniMol
# feature space, so it encodes a ~1.45M-molecule candidate pool: it needs the
# 10M library and the encoder, and writes into the study cache. Copy from Nibi:
#
#   data/10M_unif_random_subset.csv        662M
#   minimol_ampc_encoder/                   34M  (package + model/final.pt)
#   .venv                                        or rebuild it there
#
# The four eval CSVs (--val-csv, --gp-molformer-csv, --ampc-331k-csv,
# --olivier-csv) ARE required: main() resolves them unconditionally, after the
# selection, to count the new set's overlap with each. Mila job 11535584 died
# there after 22 min (FileNotFoundError: data/ampc_val_20k.csv). The Nibi job
# 26167170 cited as proof otherwise was OOM-killed before reaching that line.
# The resolved copies in cache/ampc/exact_dkl_top_n/{val_set,ampc_331k,
# olivier_invitro}.csv work as stand-ins; gp_molformer_set.csv needs a
# passes_sa column appended. Extra args after the script name are forwarded
# and override the flags below.
#
# RESOURCES ARE THE UNVERIFIED PART OF THIS SCRIPT. They could not be checked
# from Nibi, so treat them as a starting point, not a measurement:
#
#   MEMORY. The binding constraint, and its true peak is unknown. On Nibi job
#   26167170 was OOM-killed with MaxRSS exactly at a 64 G cap, so the peak is
#   ABOVE 64 G and no run has yet established how far above; the Nibi script
#   now asks for 200 G because a g-node bills by its GPU share and host memory
#   there is free up to 250 G per GPU. Mila bills differently and its partitions
#   cap memory, so 200 G may simply not be grantable. If this job is rejected
#   or OOM-killed, raise --mem and re-submit rather than assuming the build is
#   broken: an OOM kill here means the cap, not the code.
#
#   PARTITION. Measured on Mila 2026-11-04 (sacctmgr show qos): the per-user
#   QOS caps are unkillable cpu=6/mem=32G/gpu=1, main cpu=8/mem=48G/gpu=2, and
#   short-unkillable requires 4 GPUs. None fits 12 CPUs + 200G on one GPU, and
#   asking for it fails at submit with QOSMaxCpuPerUserLimit. `long` has no
#   partition QOS, so it grants this; it is preemptible, but at ~2 h a
#   preemption costs one re-submit. Check before changing anything:
#       sacctmgr -n -P show qos format=name,maxtrespu
#       sbatch --test-only jobs/exact_dkl_balanced_n40000_prepare_mila.sh
#
#   GPU. Any modern card: the job needs a few GB (the largest band's pool is
#   761,800 x 512 x 4 bytes = 1.56 GB) and is bound by MiniMol inference
#   throughput, not memory. l40s to match the other Mila jobs here.
#
#   TIME. 33 minutes on one H100 for a LARGER 1,610,733-molecule pool
#   (fps_keep05_n45000). An l40s is slower than an H100, so 1 h is thin; 2 h
#   here, which is still short enough not to hurt scheduling.
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=2:00:00
#SBATCH --job-name=ampc_balanced_n40000_prep
#SBATCH --output=slurm_logs/ampc_balanced_n40000_prep_%j.out
#SBATCH --error=slurm_logs/ampc_balanced_n40000_prep_%j.err

set -euo pipefail

cd "${HOME}/activelearning"

# Checked here rather than after the encoder has loaded: a missing input is a
# copy that did not happen, and it should say so in seconds.
for required in \
  data/10M_unif_random_subset.csv \
  minimol_ampc_encoder/model/final.pt \
  minimol_ampc_encoder/pyproject.toml
do
  if [[ ! -e "${required}" ]]; then
    echo "Missing ${required}. Copy it from Nibi:" >&2
    echo "  rsync -avP nibi:activelearning/${required} ${required}" >&2
    exit 2
  fi
done

# Refuse to clobber the 25,000 build's outputs, which share this cache dir. The
# label namespaces them, so a collision here means the label was changed.
for guarded in \
  cache/ampc/exact_dkl_top_n/balanced_pool.npy \
  cache/ampc/exact_dkl_top_n/balanced_25000.npy
do
  if [[ -e "${guarded}" ]]; then
    echo "note: ${guarded} is present and will NOT be touched (label is balanced_n40000)."
  fi
done

source .venv/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1   # compute nodes have no internet

mkdir -p slurm_logs cache/ampc/exact_dkl_top_n

nvidia-smi --query-gpu=name,memory.total --format=csv

# EVERY selection parameter below is identical to
# jobs/exact_dkl_balanced_prepare.sh, which built data/ampc_balanced_25000.csv.
# Only --strat-sizes, --strat-label and --sets differ, and only so this build
# is 40,000 rows and overwrites nothing of that one's.
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
echo "  - data/ampc_balanced_25000.csv and the balanced_* caches are unchanged"
echo
echo "Then the y distribution, which decides whether the run still starts"
echo "unenriched. The 25,000 set had 9 rows at y >= 0.5, 2 at >= 0.6, 0 at"
echo ">= 0.66; this one is NOT predicted, measure it:"
echo
echo "  awk -F, 'NR>1{n++; y=\$2+0; s+=y; if(y>=0.5)a++; if(y>=0.6)b++;"
echo "    if(y>=0.66)c++} END{printf \"rows=%d mean=%.5f y>=0.5:%d y>=0.6:%d\\"
echo "    y>=0.66:%d\\n\", n, s/n, a, b, c}' data/ampc_balanced_n40000_40000.csv"
echo
echo "Copy the set and its cache back to Nibi for the AL run:"
echo "  data/ampc_balanced_n40000_40000.csv"
echo "  cache/ampc/exact_dkl_top_n/balanced_n40000_40000.npy{,.json}"
