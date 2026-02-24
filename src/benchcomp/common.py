from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Self


@dataclass
class Device:
    brand: str = ""
    name: str = ""
    model: str = ""
    cpu_cores: int = 0
    cpu_freq: int = 0
    mem_size_mb: int = 0
    emulated: bool = True


@dataclass
class MetricMetadata:
    name: str
    name_short: str
    unit: str


@dataclass
class Metric:
    # Easier printing
    metadata: MetricMetadata

    _runs: list[float]

    # Cached values (not in __init__)
    _min: float | None = field(init=False, default=None)
    _max: float | None = field(init=False, default=None)
    _median: float | None = field(init=False, default=None)
    _mean: float | None = field(init=False, default=None)
    _stdev: float | None = field(init=False, default=None)

    @property
    def runs(self) -> list[float]:
        return self._runs

    @runs.setter
    def runs(self, value: list[float]):
        self._runs = value
        self._min = None
        self._max = None
        self._median = None
        self._stdev = None

    def min(self) -> float:
        if len(self._runs) < 1:
            return float("nan")
        if self._min is None:
            self._min = min(self._runs)
        return self._min

    def max(self) -> float:
        if len(self._runs) < 1:
            return float("nan")
        if self._max is None:
            self._max = max(self._runs)
        return self._max

    def median(self) -> float:
        if len(self._runs) < 1:
            return float("nan")
        if self._median is None:
            self._median = median(self._runs)
        return self._median

    def mean(self) -> float:
        if len(self._runs) < 1:
            return float("nan")
        if self._mean is None:
            self._mean = mean(self._runs)
        return self._mean

    def stdev(self) -> float:
        if len(self._runs) < 2:
            return float("nan")
        if self._stdev is None:
            self._stdev = stdev(self._runs)
        return self._stdev

    def cv(self) -> float:
        """Calculates the Coefficient of Variation (StDev / Mean)."""
        if len(self._runs) < 2:
            return float("nan")
        if self.mean() == 0:
            return float("nan")
        return self.stdev() / self.mean()


@dataclass
class SampledMetric:
    p50: float
    p90: float
    p95: float
    p99: float
    runs: list[list[float]]


@dataclass
class StartupTimingMetric:
    time_to_initial_display_ms: Metric
    """ timeToInitialDisplayMs - Time from the system receiving a launch intent to rendering the first frame of the destination Activity. """

    time_to_full_display_ms: Metric | None
    """
    timeToFullDisplayMs - Time from the system receiving a launch intent until
    the application reports fully drawn via android.app.Activity.reportFullyDrawn.

    The measurement stops at the completion of rendering the first frame after (or containing) the reportFullyDrawn() call.
    This measurement may not be available prior to API 29.
    """


@dataclass
class FrameTimingMetric:
    frame_count: Metric
    """
    frameCount - How many total frames were produced. This is a secondary metric which
    can be used to understand why the above metrics changed. For example,
    when removing unneeded frames that were incorrectly invalidated to save power,
    frameOverrunMs and frameDurationCpuMs will often get worse, as the removed frames were trivial.
    Checking frameCount can be a useful indicator in such cases.
    """

    frame_duration_ms: SampledMetric
    """
    frameDurationCpuMs - How much time the frame took to be produced on the CPU - on both the UI Thread, and RenderThread.
    Note that this doesn't account for time before the frame started (before Choreographer#doFrame), as that data isn't available in traces prior to API 31.
    """

    frame_overrun_ms: SampledMetric | None
    """
    frameOverrunMs (Requires API 31) - How much time a given frame missed its deadline by.
    Positive numbers indicate a dropped frame and visible jank / stutter,
    negative numbers indicate how much faster than the deadline a frame was.
    """

    def calc_freeze_frame_duration_ms(self, target: float) -> list[float]:
        """
        Calculates total 'freeze' time per run (time exceeding the target deadline).
        """
        result: list[float] = []
        for run in self.frame_duration_ms.runs:
            freeze_ms: float = 0.0
            for ft in run:
                if ft > target:
                    freeze_ms += ft - target
            result.append(freeze_ms)
        return result


class MemorySamplingMode(Enum):
    UNKNOWN = 0
    LAST = 1
    MAX = 2


@dataclass
class MemoryUsageMetric:
    sampling_mode: MemorySamplingMode
    """
    There are two modes for measurement - Last, which represents the last observed
    value during an iteration, and Max, which represents the largest sample observed per measurement.
    """

    memory_rss_anon_kb: Metric
    """
    memoryRssAnonKb - Anonymous resident/allocated memory owned by the process,
    not including memory mapped files or shared memory.
    """

    memory_rss_file_kb: Metric
    """ memoryRssFileKb - Memory allocated by the process to map files. """

    memory_heap_size_kb: Metric
    """ memoryHeapSizeKb - Heap memory allocations from the Android Runtime, sampled after each GC. """

    memory_gpu_kb: Metric | None
    """ memoryGpuKb - GPU Memory allocated for the process. """


@dataclass
class Benchmark:
    name: str
    class_name: str
    total_runtime_ns: int
    warmup_iterations: int
    repeat_iterations: int
    data: StartupTimingMetric | FrameTimingMetric | MemoryUsageMetric | None

    def is_same_benchmark(self, other: Self) -> bool:
        return (
            (self.name == other.name)
            and (self.class_name == other.class_name)
            and (type(self.data) is type(other.data))
        )


@dataclass
class BenchmarkReport:
    filepath: Path = field(default_factory=Path)
    device: Device = field(default_factory=Device)
    benchmarks: dict[str, Benchmark] = field(default_factory=dict)


@dataclass
class AnalysisReport:
    title: str = ""
    baseline_reports: list[BenchmarkReport] = field(default_factory=list)
    candidate_reports: list[BenchmarkReport] = field(default_factory=list)
    comparisons: list[BenchmarkComparisonResult] = field(default_factory=list)


class Verdict(Enum):
    NOT_SIGNIFICANT = 0
    IMPROVEMENT = 1
    REGRESSION = 2


@dataclass
class BenchmarkComparisonResult:
    a_bench_ref: list[Benchmark]
    b_bench_ref: list[Benchmark]
    a_metric: Metric
    b_metric: Metric
    comparison_method: str
    comparison_result: Any
    verdict: Verdict

    @property
    def benchmark_name(self) -> str:
        return self.a_bench_ref[0].name

    @property
    def benchmark_class(self) -> str:
        return self.a_bench_ref[0].class_name

    @property
    def metric_metadata(self) -> MetricMetadata:
        return self.a_metric.metadata

    def is_compatible_with(self, other: Self) -> bool:
        is_same_benchmark: bool = True
        ref_bench = self.a_bench_ref[0]
        for bench in self.a_bench_ref + self.b_bench_ref:
            is_same_benchmark &= ref_bench.is_same_benchmark(bench)

        return (
            is_same_benchmark
            and (self.a_metric.metadata == other.a_metric.metadata)
            and (self.b_metric.metadata == other.b_metric.metadata)
            and (self.comparison_method == other.comparison_method)
            and (type(self.comparison_result) is type(other.comparison_result))
        )

    @staticmethod
    def _calc_total_run_time_ms(benchmarks: list[Benchmark]) -> float:
        def to_seconds(time_ns: float):
            return time_ns / (1000 * 1000 * 1000)

        time:float = 0.0
        for b in benchmarks:
            time += to_seconds(b.total_runtime_ns)
        return time



def calc_total_runtime(benchmarks: list[Benchmark], target_unit: str = "s") -> float:
    """
    Calculates total runtime across all benchmarks in the desired unit.
    Supported units: 'ns', 'ms', 's'.
    """
    def humanize_time(time: float, unit: str) -> float:
        unit_divisors: dict[str, int] = {
            "ns": 1,
            "ms": 1_000_000,
            "s": 1_000_000_000,
        }
        return time / unit_divisors[unit]

    time: float = 0.0
    for b in benchmarks:
        time += humanize_time(b.total_runtime_ns, target_unit)
    return time


def calc_total_iterations(benchmarks: list[Benchmark], iteration_type: str = "repeat") -> int:
    """
    Sums iterations across benchmarks.

    Args:
        iteration_type: Either 'warm' for warmup or 'repeat' for main iterations.

    Raises:
        ValueError: If 'type' is neither 'warm' or 'repeat'.
    """
    if iteration_type == "warm":
        def it_access_func(b: Benchmark):
            return b.warmup_iterations
    elif iteration_type == "repeat":
        def it_access_func(b: Benchmark):
            return b.repeat_iterations
    else:
        raise ValueError(f"Unknown iteration type '{iteration_type}'")

    iters: int = 0
    for b in benchmarks:
        iters += it_access_func(b)
    return iters
