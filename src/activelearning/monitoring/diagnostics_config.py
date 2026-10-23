"""Configuration values for active-learning diagnostics."""

from collections.abc import Sequence
from dataclasses import dataclass, field

from activelearning.monitoring.held_out import HeldOutSetConfig


@dataclass(frozen=True)
class DiagnosticsConfig:
    """Control optional diagnostic enrichment for configured monitoring sinks.

    Core round metrics and profiling are collected independently of this
    configuration. Diagnostic enrichment is collected only when enabled and
    when the active-learning loop has a logger or run writer to receive it.
    When diagnostics are disabled, temporary component data is still cleared
    after each round so it cannot appear in a later round.

    Parameters
    ----------
    enabled : bool, default=True
        Collect reusable and component-specific diagnostic metrics and figures.
        Does not disable core metrics or profiling.
    figure_interval : int, default=1
        Render diagnostic figures every this many completed rounds. Scalar
        diagnostic metrics are still collected for every completed round.
    max_points : int, default=1000
        Bound diagnostic prediction batches, retained rolling history, and
        rendered or histogram points. Scalar summaries use all finite values.
    held_out_sets : Sequence[HeldOutSetConfig], default=()
        Fixed evaluation sets the surrogate is scored on every round. Specs,
        not loaded sets: the loop reads each one once before the first round.
        These are the constant scale on which the surrogate's own health can be
        read across rounds, unlike the per-round panels, which score whatever
        that round proposed. Each costs one encode-and-predict pass per round.
    """

    enabled: bool = True
    figure_interval: int = 1
    max_points: int = 1000
    held_out_sets: Sequence[HeldOutSetConfig] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate diagnostic rendering limits."""
        if self.figure_interval < 1:
            raise ValueError("figure_interval must be at least 1.")
        if self.max_points < 1:
            raise ValueError("max_points must be at least 1.")
