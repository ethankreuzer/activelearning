"""Per-stage wall-clock and peak-memory measurement.

The exact-DKL study exists to extrapolate what a larger training set would
cost, so the timings and peaks are the primary output rather than a diagnostic.
That puts two awkward details front and centre:

**CUDA peaks are resettable; host RSS is not.**
``torch.cuda.reset_peak_memory_stats()`` resets the peak to the *current*
allocation, so a stage's reported CUDA peak is ``max(live_at_start,
peak_during_stage)`` -- the live footprint, which is the quantity worth
extrapolating. ``cuda_allocated_at_start_mib`` recovers the stage-attributable
increment. ``resource.getrusage(...).ru_maxrss`` has no reset at all: it is a
monotone high-water mark for the whole process. So host memory is reported
three ways -- the global high-water mark at stage end, the current RSS at both
ends, and the start-to-end difference, which is the only attributable number.

**Stages must be flat.** A nested reset would destroy the enclosing stage's
peak, so :meth:`StageProfiler.stage` refuses to nest. Aggregate stages use
:meth:`StageProfiler.timing_only`, which reports seconds and host RSS but emits
no CUDA columns at all -- no number rather than a wrong one.

Every reader is injectable, so the whole module is testable on a CPU with fakes.
It lives in ``scripts/`` because it has one consumer; promote it to
``activelearning.monitoring`` if a second one appears.
"""

from __future__ import annotations

import resource
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Callable

#: Bytes per MiB. Memory is reported in MiB: bytes are unreadable in a runs
#: table and GiB loses the resolution that distinguishes the stages.
MIB = 1024**2


@dataclass(frozen=True)
class StageRecord:
    """Measurements for one completed stage.

    Attributes
    ----------
    name : str
        Stage name, used as a single key segment.
    seconds : float
        Wall-clock duration.
    cuda_peak_allocated_mib : float or None
        Peak tensor memory during the stage, or ``None`` for a timing-only
        stage or a CPU run.
    cuda_peak_reserved_mib : float or None
        Peak caching-allocator reservation, which is what has to fit in VRAM.
    cuda_allocated_at_start_mib : float or None
        Live tensor memory when the stage opened, so the increment attributable
        to the stage is ``cuda_peak_allocated_mib`` minus this.
    host_rss_start_mib : float
        Resident set size when the stage opened.
    host_rss_end_mib : float
        Resident set size when it closed.
    host_peak_rss_mib : float
        Process high-water RSS at stage end. Monotone across the whole run, so
        it is *not* attributable to this stage on its own.
    host_rss_delta_mib : float
        ``host_rss_end_mib - host_rss_start_mib``. Negative when the stage
        released more than it took.
    """

    name: str
    seconds: float
    cuda_peak_allocated_mib: float | None
    cuda_peak_reserved_mib: float | None
    cuda_allocated_at_start_mib: float | None
    host_rss_start_mib: float
    host_rss_end_mib: float
    host_peak_rss_mib: float
    host_rss_delta_mib: float


def _current_rss_bytes() -> int:
    """Return the process's current resident set size in bytes."""
    import os

    with open("/proc/self/statm") as handle:
        pages = int(handle.read().split()[1])
    return pages * os.sysconf("SC_PAGE_SIZE")


def _maxrss_bytes() -> int:
    """Return the process's high-water resident set size in bytes."""
    # ru_maxrss is in kilobytes on Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


class CudaReader:
    """Reads and resets CUDA peak-memory statistics for one device."""

    def __init__(self, device: Any = None) -> None:
        """Initialize the reader.

        Parameters
        ----------
        device : Any, optional
            Device passed through to the torch calls. ``None`` uses the current
            device.
        """
        self._device = device

    def synchronize(self) -> None:
        """Block until queued work finishes, so timings include it."""
        import torch

        torch.cuda.synchronize(self._device)

    def reset_peaks(self) -> None:
        """Reset the peak statistics to the current allocation."""
        import torch

        torch.cuda.reset_peak_memory_stats(self._device)

    def allocated_bytes(self) -> int:
        """Return currently allocated tensor bytes."""
        import torch

        return int(torch.cuda.memory_allocated(self._device))

    def peak_allocated_bytes(self) -> int:
        """Return peak allocated tensor bytes since the last reset."""
        import torch

        return int(torch.cuda.max_memory_allocated(self._device))

    def peak_reserved_bytes(self) -> int:
        """Return peak reserved bytes since the last reset."""
        import torch

        return int(torch.cuda.max_memory_reserved(self._device))


class StageProfiler:
    """Measures a sequence of flat, non-overlapping stages."""

    def __init__(
        self,
        *,
        cuda_reader: Any | None = None,
        clock: Callable[[], float] = time.perf_counter,
        rss_reader: Callable[[], int] = _current_rss_bytes,
        maxrss_reader: Callable[[], int] = _maxrss_bytes,
    ) -> None:
        """Initialize the profiler.

        Parameters
        ----------
        cuda_reader : Any, optional
            Object exposing ``synchronize``, ``reset_peaks``,
            ``allocated_bytes``, ``peak_allocated_bytes`` and
            ``peak_reserved_bytes``. ``None`` disables CUDA measurement, which
            is what a CPU run wants.
        clock : callable, default=time.perf_counter
            Monotonic clock.
        rss_reader : callable
            Returns the current RSS in bytes.
        maxrss_reader : callable
            Returns the high-water RSS in bytes.
        """
        self._cuda = cuda_reader
        self._clock = clock
        self._rss = rss_reader
        self._maxrss = maxrss_reader
        self._records: list[StageRecord] = []
        self._open: str | None = None

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Measure one stage, including its CUDA peaks.

        Parameters
        ----------
        name : str
            Stage name. Must be a single key segment: lower case, digits,
            underscores, dots or hyphens.

        Yields
        ------
        None
            The body of the stage.

        Raises
        ------
        RuntimeError
            If another stage is already open. Resetting the CUDA peak inside a
            nested stage would silently destroy the enclosing stage's peak.
        """
        with self._measure(name, measure_cuda=self._cuda is not None):
            yield

    @contextmanager
    def timing_only(self, name: str) -> Iterator[None]:
        """Measure one stage's duration and host memory, but no CUDA peaks.

        For aggregate stages that enclose others. Emitting no CUDA columns is
        deliberate: a peak measured across nested resets would be wrong, and a
        wrong number is worse than a missing one.

        Parameters
        ----------
        name : str
            Stage name.

        Yields
        ------
        None
            The body of the stage.
        """
        with self._measure(name, measure_cuda=False, allow_nesting=True):
            yield

    @contextmanager
    def _measure(
        self, name: str, *, measure_cuda: bool, allow_nesting: bool = False
    ) -> Iterator[None]:
        """Shared stage body; see :meth:`stage` and :meth:`timing_only`."""
        if not allow_nesting and self._open is not None:
            raise RuntimeError(
                f"Cannot open stage {name!r} while {self._open!r} is open: a nested "
                "CUDA peak reset would destroy the enclosing stage's peak. Use "
                "timing_only() for an aggregate stage."
            )
        previous_open = self._open
        if not allow_nesting:
            self._open = name

        cuda_start: int | None = None
        if measure_cuda and self._cuda is not None:
            self._cuda.synchronize()
            cuda_start = self._cuda.allocated_bytes()
            self._cuda.reset_peaks()
        rss_start = self._rss()
        started = self._clock()
        try:
            yield
        finally:
            if measure_cuda and self._cuda is not None:
                self._cuda.synchronize()
            seconds = self._clock() - started
            rss_end = self._rss()
            peak_allocated = (
                self._cuda.peak_allocated_bytes()
                if measure_cuda and self._cuda is not None
                else None
            )
            peak_reserved = (
                self._cuda.peak_reserved_bytes()
                if measure_cuda and self._cuda is not None
                else None
            )
            self._records.append(
                StageRecord(
                    name=name,
                    seconds=seconds,
                    cuda_peak_allocated_mib=_to_mib(peak_allocated),
                    cuda_peak_reserved_mib=_to_mib(peak_reserved),
                    cuda_allocated_at_start_mib=_to_mib(cuda_start),
                    host_rss_start_mib=rss_start / MIB,
                    host_rss_end_mib=rss_end / MIB,
                    host_peak_rss_mib=self._maxrss() / MIB,
                    host_rss_delta_mib=(rss_end - rss_start) / MIB,
                )
            )
            self._open = previous_open

    def records(self) -> tuple[StageRecord, ...]:
        """Return the completed stages, in the order they finished."""
        return tuple(self._records)

    def metrics(self, prefix: str = "run/stage") -> dict[str, float]:
        """Flatten the records into one-off scalars for the run summary.

        Parameters
        ----------
        prefix : str, default="run/stage"
            Key prefix. The result is ``<prefix>/<stage>/<field>``.

        Returns
        -------
        dict[str, float]
            Every finite measurement. ``None`` columns are omitted rather than
            reported as zero.
        """
        metrics: dict[str, float] = {}
        for record in self._records:
            for field, value in asdict(record).items():
                if field == "name" or value is None:
                    continue
                metrics[f"{prefix}/{record.name}/{field}"] = float(value)
        return metrics

    def run_peaks(self, prefix: str = "run/peak") -> dict[str, float]:
        """Return the run-level maxima across every measured stage.

        Parameters
        ----------
        prefix : str, default="run/peak"
            Key prefix.

        Returns
        -------
        dict[str, float]
            The largest CUDA peaks seen in any stage, and the process
            high-water RSS.
        """
        peaks: dict[str, float] = {}
        allocated = [
            record.cuda_peak_allocated_mib
            for record in self._records
            if record.cuda_peak_allocated_mib is not None
        ]
        reserved = [
            record.cuda_peak_reserved_mib
            for record in self._records
            if record.cuda_peak_reserved_mib is not None
        ]
        if allocated:
            peaks[f"{prefix}/cuda_allocated_mib"] = max(allocated)
        if reserved:
            peaks[f"{prefix}/cuda_reserved_mib"] = max(reserved)
        peaks[f"{prefix}/host_rss_mib"] = self._maxrss() / MIB
        return peaks

    def rows(self) -> list[dict[str, Any]]:
        """Return the records as CSV rows with a stable column order.

        The column order is part of the artifact: this CSV is the input to the
        cost extrapolation.

        Returns
        -------
        list[dict[str, Any]]
            One row per stage.
        """
        return [asdict(record) for record in self._records]

    @staticmethod
    def column_names() -> tuple[str, ...]:
        """Return the CSV column order."""
        return (
            "name",
            "seconds",
            "cuda_peak_allocated_mib",
            "cuda_peak_reserved_mib",
            "cuda_allocated_at_start_mib",
            "host_rss_start_mib",
            "host_rss_end_mib",
            "host_peak_rss_mib",
            "host_rss_delta_mib",
        )


def _to_mib(value: int | None) -> float | None:
    """Convert bytes to MiB, passing ``None`` through."""
    return None if value is None else value / MIB


def write_stage_csv(path: Any, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write the stage records to a CSV, atomically.

    Called after every stage, so a run killed by the wall clock still leaves
    the measurements it already took.

    Parameters
    ----------
    path : Any
        Destination path.
    rows : Sequence[Mapping[str, Any]]
        Rows from :meth:`StageProfiler.rows`.
    """
    import csv
    import os
    from pathlib import Path

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_name(destination.name + ".tmp")
    with tmp_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(StageProfiler.column_names()))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in writer.fieldnames})
    os.replace(tmp_path, destination)
