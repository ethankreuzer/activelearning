"""Derive the reward scale (beta and the log-score floor) from a constant K.

With the ``power`` reward transform the policy is trained toward ``log R = beta * r``
where ``r`` is the log of the acquisition score. A fixed prior sample, the policy's
step-0 sampling distribution, is scored by the surrogate, and K is defined as the
reward ratio between its 99th-percentile and its median molecule:

    K = R(p99) / R(median) = exp(beta * (r_p99 - r_median))
    beta = ln(K) / (r_p99 - r_median)

so one K means the same thing for every surrogate even though beta, which has units
of 1/nats, does not. The floor is a low percentile (5th by default) of the same
distribution. Both depend on the surrogate, and the surrogate is refit every round,
so they are re-derived each round from the same fixed sample rather than set once.

This is the in-loop counterpart of ``scripts/k_ladder_params.py``; the math is
identical.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

#: Fewer finite scores than this cannot support a 99th percentile worth trusting.
MIN_SCORES = 10_000


@dataclass(frozen=True)
class RewardScale:
    """The reward scale derived from one scoring of the prior sample.

    Attributes
    ----------
    beta : float
        ``ln(K) / spread``, the exponent in ``log R = beta * r``.
    floor : float
        The ``floor_percentile`` of the log scores; the ``log_score_floor``.
    median, p99 : float
        The two points whose reward ratio is K.
    spread : float
        ``p99 - median``, in nats.
    n : int
        Number of scores the percentiles were taken over.
    k : float
        The K the beta was derived for.
    floor_percentile : float
        Which percentile ``floor`` is.
    """

    beta: float
    floor: float
    median: float
    p99: float
    spread: float
    n: int
    k: float
    floor_percentile: float


def percentile(sorted_values: Sequence[float], q: float) -> float:
    """Return the q-th percentile by linear interpolation, as numpy's default does."""
    position = (len(sorted_values) - 1) * q / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    low, high = sorted_values[lower], sorted_values[upper]
    if lower == upper:
        return low
    if not (math.isfinite(low) and math.isfinite(high)):
        # Interpolating toward -inf would give nan; report the unbounded side.
        return low if fraction == 0.0 else (high if fraction == 1.0 else -math.inf)
    return low + (high - low) * fraction


def derive_reward_scale(
    log_scores: Sequence[float],
    k: float,
    floor_percentile: float = 5.0,
) -> RewardScale:
    """Derive beta and the log-score floor from scored prior molecules.

    Parameters
    ----------
    log_scores : Sequence[float]
        Log acquisition score of every molecule in the prior sample. A score
        that underflowed to zero must be passed as ``-inf``, never dropped: it
        is a real molecule in the lower tail, and dropping it would move every
        percentile.
    k : float
        Reward ratio between the 99th-percentile and the median molecule.
        Must be a finite number above 1.
    floor_percentile : float, optional
        Percentile of the scores used as the floor. Must lie in ``[0, 50)``, so
        that it is below the median.

    Returns
    -------
    RewardScale

    Raises
    ------
    ValueError
        If ``k`` is not a finite number above 1; if there are fewer than
        ``MIN_SCORES`` scores or any is ``nan`` or ``+inf``; if the median is
        not finite; if the floor is not finite (more than ``floor_percentile``
        percent of the scores underflowed); or if the 99th percentile is not
        above the median, which is what an unfitted acquisition returning the
        same score everywhere produces.
    """
    if not (math.isfinite(k) and k > 1.0):
        raise ValueError(f"K must be a finite number above 1, got {k!r}.")
    if not (0.0 <= floor_percentile < 50.0):
        raise ValueError(
            f"floor_percentile must be in [0, 50), got {floor_percentile!r}."
        )
    values = [float(score) for score in log_scores]
    if any(math.isnan(score) or score == math.inf for score in values):
        raise ValueError("Log scores must not contain nan or +inf.")
    if len(values) < MIN_SCORES:
        raise ValueError(
            f"Only {len(values)} scores; at least {MIN_SCORES} are needed for a "
            "stable 99th percentile."
        )
    values.sort()
    median = percentile(values, 50.0)
    p99 = percentile(values, 99.0)
    floor = percentile(values, floor_percentile)
    if not math.isfinite(median):
        raise ValueError(
            "The median log score is not finite: over half the prior sample "
            "underflowed to zero."
        )
    if not math.isfinite(floor):
        raise ValueError(
            f"The {floor_percentile:g}th-percentile log score is not finite: more "
            f"than {floor_percentile:g}% of the prior sample underflowed to zero, "
            "so there is no floor to derive."
        )
    spread = p99 - median
    if not spread > 0.0:
        raise ValueError(
            f"The 99th percentile ({p99:.6g}) is not above the median "
            f"({median:.6g}); the acquisition does not rank the prior sample "
            "(is the surrogate fitted and the acquisition updated?)."
        )
    return RewardScale(
        beta=math.log(k) / spread,
        floor=floor,
        median=median,
        p99=p99,
        spread=spread,
        n=len(values),
        k=float(k),
        floor_percentile=float(floor_percentile),
    )


def log_scores_from_values(values: Sequence[float]) -> list[float]:
    """Return the log of acquisition values, with ``-inf`` for a zero score.

    Parameters
    ----------
    values : Sequence[float]
        Acquisition scores on the value scale.

    Raises
    ------
    ValueError
        If any value is negative or not finite, which has no logarithm.
    """
    logs: list[float] = []
    for value in values:
        score = float(value)
        if not math.isfinite(score) or score < 0.0:
            raise ValueError(
                f"Acquisition returned {score!r}; the power reward needs a finite, "
                "non-negative score."
            )
        logs.append(math.log(score) if score > 0.0 else -math.inf)
    return logs


@dataclass(frozen=True)
class RewardCalibrationSettings:
    """How a sampler re-derives its reward scale every round.

    Attributes
    ----------
    k : float
        Reward ratio between the 99th-percentile and median prior molecule,
        held constant over rounds. Must be a finite number above 1.
    prior_set_path : str
        The fixed prior sample to rescore each round: a ``.csv`` with a
        ``smiles_column`` header, or any other file with one SMILES per line.
    smiles_column : str
        Header of the SMILES column when ``prior_set_path`` is a CSV.
    floor_percentile : float
        Percentile of the rescored log scores used as the log-score floor.
    """

    k: float
    prior_set_path: str
    smiles_column: str = "SMILES"
    floor_percentile: float = 5.0

    def __post_init__(self) -> None:
        if not (math.isfinite(self.k) and self.k > 1.0):
            raise ValueError(f"K must be a finite number above 1, got {self.k!r}.")
        if not (0.0 <= self.floor_percentile < 50.0):
            raise ValueError(
                f"floor_percentile must be in [0, 50), got {self.floor_percentile!r}."
            )
        if not self.prior_set_path:
            raise ValueError("prior_set_path must be set.")


def load_prior_smiles(path: str | Path, smiles_column: str = "SMILES") -> list[str]:
    """Read the fixed prior sample's SMILES, in file order.

    Parameters
    ----------
    path : str or Path
        A ``.csv`` with a ``smiles_column`` header, or a file with one SMILES
        per line.
    smiles_column : str
        Header of the SMILES column for a CSV.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If a CSV has no such column, or the file holds no SMILES.
    """
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise FileNotFoundError(f"Prior sample not found: {resolved}")
    if resolved.suffix.lower() == ".csv":
        with resolved.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or smiles_column not in reader.fieldnames:
                raise ValueError(
                    f"{resolved} has no column {smiles_column!r}; "
                    f"found {reader.fieldnames}."
                )
            smiles = [row[smiles_column] for row in reader]
    else:
        smiles = resolved.read_text(encoding="utf-8").splitlines()
    smiles = [value.strip() for value in smiles if value.strip()]
    if not smiles:
        raise ValueError(f"{resolved} holds no SMILES.")
    return smiles
