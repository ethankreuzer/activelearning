"""Tests for the per-stage time and memory profiler.

The profiler's output is the exact-DKL study's primary artifact, so these
tests pin the measurement semantics -- especially the two that are easy to get
wrong: the CUDA peak is reset per stage while host RSS is a monotone high-water
mark, and a nested stage would destroy the enclosing stage's peak.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from activelearning.monitoring.keys import validate_log_key
from scripts.stage_profiler import MIB, StageProfiler, write_stage_csv


class _FakeCuda:
    """Scriptable CUDA reader that records the order of its calls.

    Each queue yields one value per stage and then repeats its last value, so a
    test only has to script the stages it actually cares about.
    """

    def __init__(
        self,
        *,
        allocated: list[int],
        peak_allocated: list[int],
        peak_reserved: list[int] | None = None,
    ) -> None:
        self.calls: list[str] = []
        self._allocated = list(allocated)
        self._peak_allocated = list(peak_allocated)
        self._peak_reserved = list(
            peak_reserved if peak_reserved is not None else peak_allocated
        )

    @staticmethod
    def _next(queue: list[int]) -> int:
        """Pop the next value, repeating the last one once exhausted."""
        return queue.pop(0) if len(queue) > 1 else queue[0]

    def synchronize(self) -> None:
        self.calls.append("synchronize")

    def reset_peaks(self) -> None:
        self.calls.append("reset_peaks")

    def allocated_bytes(self) -> int:
        self.calls.append("allocated")
        return self._next(self._allocated)

    def peak_allocated_bytes(self) -> int:
        self.calls.append("peak_allocated")
        return self._next(self._peak_allocated)

    def peak_reserved_bytes(self) -> int:
        self.calls.append("peak_reserved")
        return self._next(self._peak_reserved)


class _Clock:
    """Monotonic clock advancing one second per read."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        self.now += 1.0
        return self.now


def _profiler(**kwargs: object) -> StageProfiler:
    """Build a profiler with deterministic fakes."""
    defaults = {
        "clock": _Clock(),
        "rss_reader": lambda: 100 * MIB,
        "maxrss_reader": lambda: 200 * MIB,
    }
    defaults.update(kwargs)
    return StageProfiler(**defaults)  # type: ignore[arg-type]


def test_stage_records_time_and_every_memory_column() -> None:
    """A CUDA stage reports seconds plus all four memory quantities."""
    cuda = _FakeCuda(allocated=[10 * MIB], peak_allocated=[50 * MIB])
    profiler = _profiler(cuda_reader=cuda)
    with profiler.stage("gp_fit"):
        pass
    (record,) = profiler.records()
    assert record.name == "gp_fit"
    assert record.seconds == pytest.approx(1.0)
    assert record.cuda_peak_allocated_mib == pytest.approx(50.0)
    assert record.cuda_allocated_at_start_mib == pytest.approx(10.0)
    assert record.host_rss_start_mib == pytest.approx(100.0)
    assert record.host_peak_rss_mib == pytest.approx(200.0)


def test_cuda_peaks_are_reset_at_start_and_read_at_end() -> None:
    """The reset must precede the body and the peak read must follow it."""
    cuda = _FakeCuda(allocated=[0], peak_allocated=[1])
    profiler = _profiler(cuda_reader=cuda)
    with profiler.stage("gp_fit"):
        cuda.calls.append("body")
    assert cuda.calls.index("reset_peaks") < cuda.calls.index("body")
    assert cuda.calls.index("body") < cuda.calls.index("peak_allocated")
    # Synchronised before reading the clock as well as the peaks, or async
    # kernels land in the next stage.
    assert cuda.calls.count("synchronize") == 2


def test_nested_stage_is_refused() -> None:
    """A nested reset would destroy the enclosing stage's peak."""
    profiler = _profiler(cuda_reader=_FakeCuda(allocated=[0], peak_allocated=[1]))
    with pytest.raises(RuntimeError, match="while 'outer' is open"):
        with profiler.stage("outer"):
            with profiler.stage("inner"):
                pass


def test_timing_only_emits_no_cuda_columns() -> None:
    """An aggregate stage reports no CUDA peak rather than a wrong one."""
    cuda = _FakeCuda(allocated=[0], peak_allocated=[1])
    profiler = _profiler(cuda_reader=cuda)
    with profiler.timing_only("run_total"):
        with profiler.stage("gp_fit"):
            pass
    aggregate = next(r for r in profiler.records() if r.name == "run_total")
    assert aggregate.cuda_peak_allocated_mib is None
    assert aggregate.cuda_peak_reserved_mib is None
    assert aggregate.seconds > 0.0
    assert "run/stage/run_total/cuda_peak_allocated_mib" not in profiler.metrics()
    assert "run/stage/run_total/seconds" in profiler.metrics()


def test_timing_only_may_enclose_a_measured_stage() -> None:
    """The aggregate wrapper does not trip the no-nesting rule."""
    profiler = _profiler(cuda_reader=_FakeCuda(allocated=[0], peak_allocated=[1]))
    with profiler.timing_only("run_total"):
        with profiler.stage("gp_fit"):
            pass
    assert [record.name for record in profiler.records()] == ["gp_fit", "run_total"]


def test_host_rss_delta_is_the_attributable_number() -> None:
    """``ru_maxrss`` cannot be reset, so the start-to-end delta is what counts."""
    readings = iter([100 * MIB, 180 * MIB, 180 * MIB, 180 * MIB])
    profiler = _profiler(rss_reader=lambda: next(readings))
    with profiler.stage("features_ampc_331k"):
        pass
    with profiler.stage("quiet"):
        pass
    grew, quiet = profiler.records()
    assert grew.host_rss_delta_mib == pytest.approx(80.0)
    assert quiet.host_rss_delta_mib == pytest.approx(0.0)
    # The high-water mark is the same in both, which is exactly why it is not
    # attributable to either stage on its own.
    assert grew.host_peak_rss_mib == quiet.host_peak_rss_mib


def test_metric_and_peak_keys_are_valid_log_keys() -> None:
    """Every emitted key must survive the logger's key validation."""
    cuda = _FakeCuda(allocated=[0], peak_allocated=[1])
    profiler = _profiler(cuda_reader=cuda)
    with profiler.stage("score_gibbon_value_ampc_331k"):
        pass
    with profiler.timing_only("run_total"):
        pass
    for key in {**profiler.metrics(), **profiler.run_peaks()}:
        validate_log_key(key)


def test_run_peaks_take_the_maximum_across_stages() -> None:
    """The run-level peak is the largest any single stage reached."""
    cuda = _FakeCuda(allocated=[0], peak_allocated=[30 * MIB, 70 * MIB])
    profiler = _profiler(cuda_reader=cuda)
    with profiler.stage("small"):
        pass
    with profiler.stage("large"):
        pass
    assert profiler.run_peaks()["run/peak/cuda_allocated_mib"] == pytest.approx(70.0)


def test_measurements_survive_an_exception_in_the_body() -> None:
    """A stage that raises still records, so a failed run keeps its numbers."""
    profiler = _profiler(cuda_reader=_FakeCuda(allocated=[0], peak_allocated=[1]))
    with pytest.raises(ValueError):
        with profiler.stage("gp_fit"):
            raise ValueError("boom")
    assert [record.name for record in profiler.records()] == ["gp_fit"]


def test_csv_column_order_is_stable(tmp_path: Path) -> None:
    """The CSV is the extrapolation input, so its columns are part of the contract."""
    profiler = _profiler(cuda_reader=_FakeCuda(allocated=[0], peak_allocated=[1]))
    with profiler.stage("gp_fit"):
        pass
    path = tmp_path / "stage_profile.csv"
    write_stage_csv(path, profiler.rows())
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    assert tuple(rows[0]) == StageProfiler.column_names()
    assert rows[1][0] == "gp_fit"
    assert not list(tmp_path.glob("*.tmp"))


def test_cpu_run_reports_no_cuda_columns() -> None:
    """Without a CUDA reader every stage is timing plus host memory only."""
    profiler = _profiler(cuda_reader=None)
    with profiler.stage("gp_fit"):
        pass
    (record,) = profiler.records()
    assert record.cuda_peak_allocated_mib is None
    assert record.seconds > 0.0
