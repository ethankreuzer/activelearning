"""Which K-ladder arms exist on disk, for the figure scripts in this directory.

The ladder is RAGGED and the figure scripts must not assume otherwise:

  ExactGP   seeds 42, 43 at every K, plus seeds 44 and 45 at K = 200, 250 and 300
            only (Slurm job 11911887, 2026-11-26, added to resolve whether the
            duplicate-rate spread at those three K values was real).
  ELBO      seeds 42 and 43 only.
  PLL       seeds 42 and 43 only, and K = 200 and 250 were never run for it.

One arm, exactgp K=300 seed 45, hit the 500,000-attempt cap after 57,399 of the
requested 100,000 molecules, so the sampler raised and never wrote a pool. It
HAS training_steps_round_0001.csv and generation_summary_round_0001.json and so
appears in the training and generation figures (as a cap marker where the figure
draws one); it has no generated_pool_round_0001.csv and so is absent from every
pool figure. Its generation numbers describe a run that never filled the pool
and are not comparable with the arms that did.

The pool-derived .npz files (pool_diversity, pool_train_similarity) are written
by scripts/pool_diversity.py and scripts/pool_train_similarity.py as separate
jobs, which have NOT been run on seeds 44 and 45. Pool figures therefore still
show two seeds until those jobs are rerun.

Rather than hard-code any of this, a script names the file it reads and asks
seeds_for() which seeds actually have it. Adding seeds or K values later means
running the arms, not editing the figures.
"""

import re
from pathlib import Path

REPO = Path("/home/mila/k/kreuzere/activelearning")
ROOT = REPO / "outputs/ampc/k_ladder"
LOGS = REPO / "slurm_logs"

# The fitted surrogate each arm restored, for its scored prior sample. That CSV is the
# step-0 reference distribution: the policy is rebuilt from the stock prior every round
# (copy.deepcopy of the pretrained policy, sampler.py), so it is the right baseline to
# compare a pool against.
EVAL_CSV = {
    "exactgp": REPO / "outputs/ampc/exact_gp_sweep/n20000_on_pinned/eval/gp_molformer_set.csv",
    "elbo": REPO / "outputs/ampc/variational_gp_sweep/n20000_elbo_on_s10000/eval/gp_molformer_set.csv",
    "pll": REPO / "outputs/ampc/variational_gp_sweep/n20000_pll_on_s10000/eval/gp_molformer_set.csv",
    "pll-desalt": REPO / "outputs/ampc/variational_gp_sweep/n20000_pll_on_s10000/eval/gp_molformer_set.csv",
}

# Every seed the ladder has ever used, ascending. Membership is decided per arm
# by what is on disk, so listing a seed here that was never run costs nothing.
ALL_SEEDS = [42, 43, 44, 45]
KS = [10, 30, 100, 200, 250, 300, 1000]
PLL_KS = [10, 30, 100, 300, 1000]  # K = 200 and 250 were run for ExactGP and ELBO only

# The per-arm files a figure might key off, by the artifact it needs.
GENERATION = "generation_summary_round_0001.json"
TRAINING = "training_steps_round_0001.csv"
POOL = "generated_pool_round_0001.csv"
DIVERSITY = "pool_diversity_round_0001.npz"
TRAIN_SIMILARITY = "pool_train_similarity_round_0001.npz"
SCORED = "artifacts/acquisition/general/round_0001/scored_samples.csv"

_LOG_INDEX = None


def log_for(surrogate, k, seed):
    """The Slurm log holding this arm's generation progress, or None.

    Arms were submitted by several different job scripts, so the file name does not
    identify the arm; each log names it on its own 'arm  kladder_...' header line.
    The directory is scanned once and cached.
    """
    global _LOG_INDEX
    if _LOG_INDEX is None:
        _LOG_INDEX = {}
        header = re.compile(r"^arm\s+(kladder_\S+)", re.MULTILINE)
        for path in sorted(LOGS.glob("*.log")) + sorted(LOGS.glob("*.out")):
            try:
                text = path.read_text(errors="ignore")
            except OSError:
                continue
            if "Final candidate batch" not in text:
                continue
            found = header.search(text)
            if found:
                _LOG_INDEX.setdefault(found.group(1), path)
    return _LOG_INDEX.get(f"kladder_{surrogate}_K{k}_s{seed}_steps5000")


def generation_progress(surrogate, k, seed):
    """[(molecules_accepted, attempts), ...] through this arm's final generation.

    Parsed from the sampler's progress lines, which are logged at widening intervals
    (every batch for the first three, then every fifth), so the points are not evenly
    spaced in attempts. Empty when no log was found.
    """
    path = log_for(surrogate, k, seed)
    if path is None:
        return []
    pattern = re.compile(
        r"Final candidate batch \d+: (\d+)/\d+ unique candidate\(s\), attempts=(\d+)"
    )
    return [(int(m.group(1)), int(m.group(2)))
            for m in pattern.finditer(path.read_text(errors="ignore"))]


def arm(surrogate, k, seed):
    """Path to one arm's output directory."""
    return ROOT / f"kladder_{surrogate}_K{k}_s{seed}_steps5000"


def ks_for(surrogate):
    """The K values run for this surrogate."""
    return PLL_KS if surrogate.startswith("pll") else KS


def seeds_for(surrogate, k, artifact):
    """Seeds of this arm whose `artifact` is on disk, ascending.

    Returns an empty list when the arm was never run, or ran but did not write
    that file -- a caller iterating K values should skip those rather than fail.
    """
    return [sd for sd in ALL_SEEDS if (arm(surrogate, k, sd) / artifact).exists()]


def series(surrogate, artifact, value):
    """Per-seed points and the per-K seed mean, over a ragged seed grid.

    Parameters
    ----------
    surrogate : str
        Arm surrogate label, e.g. ``"exactgp"`` or ``"pll-desalt"``.
    artifact : str
        The per-arm file ``value`` reads; decides which seeds count as run.
    value : callable
        ``value(surrogate, k, seed) -> float``.

    Returns
    -------
    tuple
        ``(xs, ys, ks, means)``: ``xs``/``ys`` are one point per seed for a
        scatter, ``ks``/``means`` the seed mean for a line. ``ks`` omits any K
        with no data, so the line never interpolates across a missing rung.
        The number of seeds behind each mean varies by K -- see this module's
        docstring -- so a figure using ``means`` should let the reader count
        the markers.
    """
    xs, ys, ks, means = [], [], [], []
    for k in ks_for(surrogate):
        seeds = seeds_for(surrogate, k, artifact)
        if not seeds:
            continue
        values = [value(surrogate, k, sd) for sd in seeds]
        xs += [k] * len(values)
        ys += values
        ks.append(k)
        means.append(sum(values) / len(values))
    return xs, ys, ks, means


def seed_caption(surrogate, artifact):
    """Seeds behind one surrogate's figure, as a title fragment.

    ``"seeds 42, 43"`` when the grid is square, otherwise it names the K values
    each seed covers, so a title never claims seeds an arm does not have.
    """
    by_seed = {}
    for k in ks_for(surrogate):
        for sd in seeds_for(surrogate, k, artifact):
            by_seed.setdefault(sd, []).append(k)
    if not by_seed:
        return "no seeds"
    everywhere = [sd for sd, ks in by_seed.items() if len(ks) == len(ks_for(surrogate))]
    parts = []
    if everywhere:
        parts.append("seeds " + ", ".join(str(sd) for sd in sorted(everywhere)))
    # Seeds covering the same K values are named once, so four seeds at three
    # shared K values read as "44, 45 at K=200/250/300" and not as two clauses.
    shared = {}
    for sd, ks in by_seed.items():
        if sd not in everywhere:
            shared.setdefault(tuple(ks), []).append(sd)
    for ks in sorted(shared):
        seeds = ", ".join(str(sd) for sd in sorted(shared[ks]))
        parts.append(f"{seeds} at K={'/'.join(str(k) for k in ks)}")
    return "; ".join(parts)
