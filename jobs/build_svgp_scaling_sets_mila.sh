#!/bin/bash
# Build the SVGP scaling sets (data/ampc_svgpscale_<n>.csv and their feature
# caches) on Mila from the CORRECTED 10M library.
#
#   SIZES=25000 sbatch jobs/build_svgp_scaling_sets_mila.sh          # the exact GP's set
#   SIZES="50000 100000 300000" sbatch jobs/build_svgp_scaling_sets_mila.sh
#   sbatch jobs/build_svgp_scaling_sets_mila.sh                      # all five, nested
#
# 25,000 is the exact GP's training size: its fit is an exact Cholesky, which at
# n=25,000 peaked at 38.8 GB of an L40S's 48 GB, and n=40,000 would need ~48 GiB.
#
# ADDING AN ARM SMALLER THAN THE LARGEST IS FREE, and 20,000 was added this way
# on 2026-11-10 (job 11627959, 4 min 37 s for all five). The pools are sized by
# `budgets[requested[-1]]`, the LARGEST arm, so re-running with a smaller size
# appended leaves every other arm's pool, greedy order and prefix untouched: the
# four arms that already existed came back byte-for-byte identical, verified by
# md5. A LARGER arm is a different matter -- it redraws every pool and changes
# every existing set.
#
# What you cannot do is carve the new arm out of an existing CSV. The written
# file is sorted by descending pProp (`rows[np.lexsort((rows, -values[rows]))]`),
# so it records WHICH molecules were selected and not the greedy order a prefix
# would need. Re-run the build instead; back the old arms up first, because the
# builder refuses to overwrite, and see the warning on the suffix below.
#
# NESTING, and what splitting this job costs. Within ONE job the arms are
# nested -- a single farthest-point pass per band, sized for the largest arm,
# gives every smaller arm a per-band prefix. ACROSS jobs they are not: each
# band's candidate pool is drawn as min(band_size, max_budget * multiple), so a
# job whose largest arm is 25,000 draws smaller pools than one whose largest is
# 300,000, and greedy FPS then picks different molecules from them. So
# SIZES=25000 followed by SIZES="50000 100000 300000" yields a 25k set that is
# NOT a subset of the 50k set, and the three larger arms nested only among
# themselves. That is the price of getting the exact GP's set first, and it is
# fine as long as nothing claims otherwise: the scaling study's comparison is
# between the three larger arms.
#
# The bands are cut on pProp in uniform 0.5-wide steps (--pprop-csv supplies it),
# not on the target y. The script's docstring has the reasoning; the short version
# is that y is a hit probability, peaks at pProp 6.32 and falls away above it, so
# a carve on y is a band closed at both ends and drops the best-docking molecules.
# This is what makes these sets differ from the ones built before 2026-11-09.
#
# --skip-gentail is deliberate: the generated tail set is fixed separately (its y is
# remapped from the run's dock scores), and the builder would otherwise rebuild it
# from the run's stored y, which came from the old, wrong score -> pProp table.
#
# Nothing is encoded for the library arms: the pool's features are a slice of the
# 10M fingerprint store, whose SMILES order the builder checks against the CSV by
# hash. The GPU is only for farthest-point sampling (one band's pool as an (N, 512)
# float32 matrix), so any card works; l40s to match the other Mila jobs here.
#
# Memory and time, measured rather than guessed. The directives below are sized
# for the DEFAULT build, all four arms at once.
#
# Job 11569178 built the old three-arm ladder in 14 min 40 s and peaked at
# 30.7 GB with a pool of 1,585,502 rows. The seven-band design's pool is
# 1,140,138 rows for the same largest arm, 72% of that, so expect ~10-12 min and
# ~22-25 GB. Nearly all of it is gathering the pool out of the 20 GB fingerprint
# store, at roughly 1,900 rows/s, and that gather is sized by the LARGEST arm
# alone -- which is why adding the 25,000 arm to a ladder that already builds
# 300,000 costs almost nothing, and why building it on its own took about a
# minute.
#
# So: `unkillable` at 32 G, which is its per-user QOS cap and the most that can
# be asked there (more is rejected with QOSMaxMemoryPerUser). Not a compromise
# but the best of the three -- measured with `sbatch --test-only` on 2026-11-09
# at 20:09, the same job would start at:
#
#   unkillable, 32 G   immediately      not preemptible
#   long,       32 G   +1 h 15 min      preemptible
#   main,    24-48 G   +11 h 35 min     not preemptible
#
# 32 G against a projected 22-25 GB peak is ~30% headroom. Modelling the old
# build's 30.7 GB as ~5 GB of fixed cost (the library frame and the two 10M
# SMILES lists) plus ~16 GB per million pool rows puts this build at ~24 GB.
# If it is OOM-killed, that model is wrong rather than the code: move to `long`
# at 48 G and accept preemption. The job writes nothing until the whole
# selection is complete, so an interrupted build leaves no half-written arm.
#SBATCH --partition=unkillable
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --job-name=svgp_scale_prep
#SBATCH --output=slurm_logs/svgp_scale_prep_%j.out
#SBATCH --error=slurm_logs/svgp_scale_prep_%j.err

set -euo pipefail

cd "${HOME}/activelearning"

# Which arms to build. Space-separated, passed through to --sizes. See the
# nesting note above before splitting a build across jobs.
SIZES="${SIZES:-20000 25000 50000 100000 300000}"
for size in ${SIZES}; do
  if [[ ! "${size}" =~ ^[1-9][0-9]*$ ]]; then
    echo "SIZES must be positive integers; got '${SIZES}'." >&2
    exit 2
  fi
done
echo "Building arms: ${SIZES}"

# Checked up front so a missing or stale input fails in seconds, not after the
# library has loaded.
for required in \
  data/10M_unif_random_subset.csv \
  data/ampc_unif_random_10M.csv.gz \
  cache/ampc/s3gfn_minimol_ampc_fingerprints.npy \
  cache/ampc/s3gfn_minimol_ampc_fingerprints.npy.json \
  cache/ampc/s3gfn_minimol_ampc_fingerprints.smiles.txt
do
  if [[ ! -e "${required}" ]]; then
    echo "Missing ${required}. Copy it from Nibi (login node):" >&2
    echo "  rsync -avP nibi:activelearning/${required} ${required}" >&2
    exit 2
  fi
done

# The library must be the corrected one. Its first data row's y is
# 0.013251174689703469; the old, wrong file has a different value there.
if ! sed -n 2p data/10M_unif_random_subset.csv | grep -q ",0.013251174689703469,1$"; then
  echo "data/10M_unif_random_subset.csv does not start with the corrected y" >&2
  echo "(expected 0.013251174689703469 on the first row). Is this the OLD file?" >&2
  exit 3
fi

# The ground-truth pProp table is what makes the uniform bands possible; the old
# one pinned 97.48% of the reference screen at a single score. A pprop_1p7b
# column is the marker that this file came through the right table.
# Read the header into a variable rather than piping zcat into grep: under
# `set -o pipefail` the `head` closes the pipe, zcat dies of SIGPIPE, and the
# pipeline reports failure even though the column is there.
PPROP_HEADER="$(gzip -cd data/ampc_unif_random_10M.csv.gz | head -1 || true)"
if [[ "${PPROP_HEADER}" != *pprop_1p7b* ]]; then
  echo "data/ampc_unif_random_10M.csv.gz has no pprop_1p7b column." >&2
  echo "Only pprop_OLD would be available, which comes from the broken" >&2
  echo "full_scores.df table and must not be used for the bands." >&2
  exit 3
fi

# The builder refuses to overwrite a cache; fail here with a clear message
# instead, and only for the arms this job actually builds. The pre-2026-11-09
# files are NOT stale labels -- they carry correct y. What changed is the
# SELECTION: bands on pProp in uniform 0.5 steps instead of on y at four edges.
# So the backup suffix is _OLD_YBANDS, distinct from the _OLD_WRONG_Y files,
# which really were mislabelled.
for size in ${SIZES}; do
  for guarded in \
    "cache/ampc/exact_dkl_top_n/svgpscale_${size}.npy" \
    "data/ampc_svgpscale_${size}.csv"
  do
    if [[ -e "${guarded}" ]]; then
      echo "${guarded} already exists. Back the old arm up first." >&2
      echo >&2
      echo "WARNING: _OLD_YBANDS files ALREADY EXIST for 50k/100k/300k and are" >&2
      echo "the genuinely superseded y-banded sets. Use that suffix only if you" >&2
      echo "are replacing a y-banded arm; to re-run the CURRENT recipe (e.g. to" >&2
      echo "append a smaller size), move the files to a dated directory instead," >&2
      echo "as backup/svgpscale_prerebuild_20261110/ does, or you will overwrite" >&2
      echo "real history with identical copies of the current files." >&2
      echo >&2
      echo "Replacing a y-banded arm:" >&2
      echo "  mv data/ampc_svgpscale_${size}.csv data/ampc_svgpscale_${size}_OLD_YBANDS.csv" >&2
      echo "  for ext in npy npy.json; do" >&2
      echo "    mv cache/ampc/exact_dkl_top_n/svgpscale_${size}.\${ext} \\" >&2
      echo "       cache/ampc/exact_dkl_top_n/svgpscale_${size}_OLD_YBANDS.\${ext}" >&2
      echo "  done" >&2
      exit 4
    fi
  done
done

source .venv/bin/activate

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1   # compute nodes have no internet

mkdir -p slurm_logs cache/ampc/exact_dkl_top_n

nvidia-smi --query-gpu=name,memory.total --format=csv

uv run --no-sync python -m scripts.build_svgp_scaling_sets \
  --skip-gentail --sizes ${SIZES} "$@"

# One selection record per build, rather than letting a later build overwrite an
# earlier one's provenance: the script always writes <label>_selection.json.
SELECTION="cache/ampc/exact_dkl_top_n/svgpscale_selection.json"
PER_BUILD="cache/ampc/exact_dkl_top_n/svgpscale_selection_n${SIZES// /_}.json"
cp "${SELECTION}" "${PER_BUILD}"

echo
echo "Selection recorded in:"
echo "  ${PER_BUILD}"
echo "  ${SELECTION}  (overwritten by the next build)"
