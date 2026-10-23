"""A fixed evaluation set the surrogate is scored on every round.

The loop's own ``surrogate/general/held_out/*`` panels score each round's newly
docked molecules, which is accuracy on the frontier the policy proposed but a
moving target: it cannot say whether the surrogate itself got worse. A set that
does not change can, which is what makes "it worked early and degraded at round
N" a statement about the surrogate rather than about the policy.

Loaded once per run and held in memory: the molecules as candidates and the
targets as a NumPy array. The features are NOT cached here. They come from the
surrogate's own encoder, so they are by construction the ones the GP was fitted
with -- a separate feature file would be a second source of truth that a
changed encoder or checkpoint would silently invalidate.
"""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from activelearning.utils.types import Candidate

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HeldOutSet:
    """Molecules and observed targets for one fixed evaluation set.

    Attributes
    ----------
    name : str
        Namespace the metrics are logged under.
    candidates : tuple[Candidate, ...]
        The molecules, in file order.
    targets : np.ndarray
        Their observed targets, aligned with ``candidates``.
    """

    name: str
    candidates: tuple[Candidate, ...]
    targets: np.ndarray


class HeldOutSetConfig(BaseModel):
    """Configuration for one fixed evaluation set."""

    model_config = ConfigDict(extra="forbid")

    #: Metrics are logged under ``surrogate/eval_set/<name>/``. Lower case,
    #: digits and underscores, because it becomes part of a metric key.
    name: str = Field(pattern=r"^[a-z0-9_]+$")
    #: CSV holding one molecule and its observed target per row.
    path: str
    x_column: str = "SMILES"
    y_column: str = "y"
    fidelity: Optional[int] = 1
    #: Rows to keep, evenly spaced through the file, or None for all of them.
    #: The cost is one encode-and-predict pass over the set per round, which for
    #: a store-backed encoder is lookups rather than inference.
    max_rows: Optional[int] = Field(default=None, ge=1)

    def load(self) -> HeldOutSet:
        """Read the set from disk.

        Returns
        -------
        HeldOutSet
            The molecules and their finite targets.

        Raises
        ------
        FileNotFoundError
            If the CSV does not exist.
        ValueError
            If a configured column is missing, or no row has a finite target.
        """
        return load_held_out_set(
            name=self.name,
            path=Path(self.path),
            x_column=self.x_column,
            y_column=self.y_column,
            fidelity=self.fidelity,
            max_rows=self.max_rows,
        )


def load_held_out_set(
    *,
    name: str,
    path: Path,
    x_column: str = "SMILES",
    y_column: str = "y",
    fidelity: int | None = 1,
    max_rows: int | None = None,
) -> HeldOutSet:
    """Read a fixed evaluation set from a CSV.

    Rows whose target is missing or not finite are dropped, because a metric
    over them is undefined and a figure cannot place them.

    Parameters
    ----------
    name : str
        Namespace for the set's metrics.
    path : Path
        The CSV to read.
    x_column, y_column : str
        Columns holding the molecule and its observed target.
    fidelity : int or None, optional
        Fidelity recorded on every candidate.
    max_rows : int, optional
        Keep at most this many rows, evenly spaced through the file rather than
        taken from the front: these files are often sorted by target, so a
        prefix would be a biased sample of one end.

    Returns
    -------
    HeldOutSet
        The loaded set.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Held-out set {name!r} is not a file: {path}")
    molecules: list[str] = []
    targets: list[float] = []
    dropped = 0
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        for column in (x_column, y_column):
            if column not in fieldnames:
                raise ValueError(
                    f"Held-out set {name!r} has no {column!r} column; {path} holds "
                    f"{fieldnames}."
                )
        for row in reader:
            target = _finite(row.get(y_column))
            if target is None:
                dropped += 1
                continue
            molecules.append(row[x_column])
            targets.append(target)
    if not molecules:
        raise ValueError(f"Held-out set {name!r} has no row with a finite target.")
    if max_rows is not None and len(molecules) > max_rows:
        keep = np.linspace(0, len(molecules) - 1, num=max_rows).astype(np.int64)
        keep = np.unique(keep)
        molecules = [molecules[index] for index in keep]
        targets = [targets[index] for index in keep]
    _logger.info(
        "Held-out set %s: %d rows from %s%s.",
        name,
        len(molecules),
        path,
        f", {dropped} dropped for a non-finite target" if dropped else "",
    )
    return HeldOutSet(
        name=name,
        candidates=tuple(
            Candidate(x=molecule, fidelity=fidelity) for molecule in molecules
        ),
        targets=np.asarray(targets, dtype=np.float64),
    )


def _finite(value: object) -> float | None:
    """Return a finite float, or ``None`` when the cell is not one."""
    if value is None or value == "":
        return None
    try:
        target = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return target if math.isfinite(target) else None


#: Accepted literal for a set that the loop should score every round.
HeldOutSetKind = Literal["csv"]
