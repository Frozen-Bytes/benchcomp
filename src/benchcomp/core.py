from __future__ import annotations

import logging
import math
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass, field, fields
from enum import Enum
from functools import cached_property
from itertools import chain
from pathlib import Path
from statistics import mean, median, stdev
from typing import Callable, Self, cast

from scipy.stats import mannwhitneyu

logger = logging.getLogger(__name__)

MEASURES: list[str] = [
    # StartupTimingMetric
    "TID",                # Time To Initial Display
    "TFD",                # Time To Full Display

    # FrameFreezeMetric
    "FFD",                # Frame Freeze Duration
    "FFO",                # Frame Freeze Overrun

    # MemoryUsageMetric
    "MEM_RSS_MAX",        # Total RSS Memory Usage Max
    "MEM_RSS_ANON_MAX",   # Memory Resident Set Size Anonymous
    "MEM_RSS_FILE_MAX",   # Memory Resident Set Size File
    "MEM_HEAP_SIZE_MAX",  # Memory Heap Size
    "MEM_GPU_MAX",        # Memory GPU

    "MEM_RSS_LAST",       # Total RSS Memory Usage Last
    "MEM_RSS_ANON_LAST",  # Memory Resident Set Size Anonymous
    "MEM_RSS_FILE_LAST",  # Memory Resident Set Size File
    "MEM_HEAP_SIZE_LAST", # Memory Heap Size
    "MEM_GPU_LAST",       # Memory GPU
]

DEFAULT_THRESHOLDS: dict[str, float] = {
    "stepfit": 25.0,
    "mannwhitneyu": 0.01,
}

DEFAULT_FRAME_TIME_TARGET_MS: float = 1000 / 60  # ~16.667 ms (60 FPS)
_g_frame_time_target_ms = DEFAULT_FRAME_TIME_TARGET_MS

def get_frame_time_target_ms() -> float:
    return _g_frame_time_target_ms

def set_frame_time_target_ms(target: float) -> None:
    global _g_frame_time_target_ms
    _g_frame_time_target_ms = target


@dataclass(frozen=True)
class CompareFunctionMetadata:
    id: str
    name: str
    state_label: str


COMPARE_METHODS: dict[str, CompareFunctionMetadata] = {
    "stepfit": CompareFunctionMetadata(
        id="stepfit",
        name="Step Fit",
        state_label="fit",
    ),
    "mannwhitneyu": CompareFunctionMetadata(
        id="mannwhitneyu",
        name="Mann-Whitney U-Test",
        state_label="p",
    ),
}


AggregateFunc = Callable[[list[list[float]]], list[float]]
AGGREGATE_FUNCTIONS: dict[str, AggregateFunc] = {
    "concat": lambda runs: list(chain.from_iterable(runs)),
    "min": lambda runs: [min(run) for run in runs],
    "max": lambda runs: [max(run) for run in runs],
    "median": lambda runs: [median(run) for run in runs],
    "mean": lambda runs: [mean(run) for run in runs],
}


@dataclass
class MeasurementMetadata:
    name: str
    name_short: str
    unit: str


@dataclass(frozen=True)
class Measurement:
    metadata: MeasurementMetadata
    runs: list[float]

    @staticmethod
    def aggregate(measurements: list[Measurement], function: str) -> Measurement:
        if not measurements:
            raise ValueError(f"Cannot aggregate measurements using '{function}': no measurements were provided.")

        if function not in AGGREGATE_FUNCTIONS.keys():
            raise ValueError(f"Unknown aggregation method: {function}")

        aggregator = AGGREGATE_FUNCTIONS[function]

        return Measurement(
            metadata=measurements[0].metadata,
            runs=aggregator([m.runs for m in measurements]),
        )

    @cached_property
    def min(self) -> float:
        return min(self.runs) if len(self.runs) >= 1 else float("nan")

    @cached_property
    def max(self) -> float:
        return max(self.runs) if len(self.runs) >= 1 else float("nan")

    @cached_property
    def median(self) -> float:
        return median(self.runs) if len(self.runs) >= 1 else float("nan")

    @cached_property
    def mean(self) -> float:
        return mean(self.runs) if len(self.runs) >= 1 else float("nan")

    @cached_property
    def stdev(self) -> float:
        return stdev(self.runs) if len(self.runs) >= 2 else float("nan")

    @cached_property
    def cv(self) -> float:
        """Calculates the Coefficient of Variation (StDev / Mean)."""
        if math.isnan(self.stdev) or self.mean == 0:
            return float("nan")
        return self.stdev / self.mean


@dataclass
class SampledMeasurement:
    p50: float
    p90: float
    p95: float
    p99: float
    runs: list[list[float]]


@dataclass
class MetricBase(ABC):
    @staticmethod
    def aggregate(metrics: list[MetricBase], function: str) -> MetricBase:
        if not metrics:
            raise ValueError(f"Cannot aggregate measurements using '{function}': no metrics were provided.")

        return type(metrics[0])._aggregate(metrics, function)

    def get_measurements(self) -> list[Measurement]:
        result: list[Measurement] = []

        for field_def in fields(self):
            value = getattr(self, field_def.name)

            if isinstance(value, Measurement):
                result.append(value)
            elif value is not None and isinstance(value, Measurement):
                result.append(value)

        result.extend(self._get_derived_measurements())

        return result

    @classmethod
    def _aggregate(cls, metrics: list["MetricBase"], function: str) -> "MetricBase":
        if not all(isinstance(m, cls) for m in metrics):
            raise ValueError("Cannot aggregate different metric types")

        kwargs = {}
        for field_def in fields(cls):
            field_name = field_def.name
            values = [getattr(m, field_name) for m in metrics]

            # Aggregate Measurement
            if isinstance(values[0], Measurement):
                kwargs[field_name] = Measurement.aggregate(values, function)
            # Aggregate optional Measurement
            elif values[0] is None or isinstance(values[0], Measurement):
                non_null = [v for v in values if v is not None]
                kwargs[field_name] = (
                    Measurement.aggregate(non_null, function) if non_null else None
                )
            # Otherwise just take first (e.g. enums)
            else:
                kwargs[field_name] = values[0]

        return cls(**kwargs)

    def _get_derived_measurements(self) -> list[Measurement]:
        return []


@dataclass
class StartupTimingMetric(MetricBase):
    time_to_initial_display_ms: Measurement
    """ timeToInitialDisplayMs - Time from the system receiving a launch intent to rendering the first frame of the destination Activity. """

    time_to_full_display_ms: Measurement | None
    """
    timeToFullDisplayMs - Time from the system receiving a launch intent until
    the application reports fully drawn via android.app.Activity.reportFullyDrawn.

    The measurement stops at the completion of rendering the first frame after (or containing) the reportFullyDrawn() call.
    This measurement may not be available prior to API 29.
    """


@dataclass
class FrameFreezeMetric(MetricBase):
    frame_freeze_duration_ms: Measurement
    frame_freeze_overrun_ms: Measurement | None


@dataclass
class FrameTimingMetric(MetricBase):
    frame_count: Measurement
    """
    frameCount - How many total frames were produced. This is a secondary metric which
    can be used to understand why the above metrics changed. For example,
    when removing unneeded frames that were incorrectly invalidated to save power,
    frameOverrunMs and frameDurationCpuMs will often get worse, as the removed frames were trivial.
    Checking frameCount can be a useful indicator in such cases.
    """

    frame_duration_ms: SampledMeasurement
    """
    frameDurationCpuMs - How much time the frame took to be produced on the CPU - on both the UI Thread, and RenderThread.
    Note that this doesn't account for time before the frame started (before Choreographer#doFrame), as that data isn't available in traces prior to API 31.
    """

    frame_overrun_ms: SampledMeasurement | None
    """
    frameOverrunMs (Requires API 31) - How much time a given frame missed its deadline by.
    Positive numbers indicate a dropped frame and visible jank / stutter,
    negative numbers indicate how much faster than the deadline a frame was.
    """

    def _get_derived_measurements(self) -> list[Measurement]:
        return self.to_freeze_metric().get_measurements()

    def to_freeze_metric(self, frame_time_target_ms: float | None = None) -> FrameFreezeMetric:
        if not frame_time_target_ms:
            frame_time_target_ms = get_frame_time_target_ms()

        return FrameFreezeMetric(
            frame_freeze_duration_ms=Measurement(
                MeasurementMetadata(
                    name="Frame Freeze Duration",
                    name_short="FFD",
                    unit="ms",
                ),
                runs=self._calc_freeze_frame_duration_ms(frame_time_target_ms),
            ),
            frame_freeze_overrun_ms=Measurement(
                MeasurementMetadata(
                    name="Frame Freeze Overrun",
                    name_short="FFO",
                    unit="ms",
                ),
                runs=self._calc_freeze_frame_overrun_ms(),
            )
            if self.frame_overrun_ms
            else None,
        )

    # This is a hack, but works for now
    @classmethod
    def _aggregate(cls, metrics: list[MetricBase], function: str) -> MetricBase:
        freeze_metrics: list[MetricBase] = [ cast(FrameTimingMetric, m).to_freeze_metric() for m in metrics ]
        return FrameFreezeMetric.aggregate(freeze_metrics, function)

    def _calc_freeze_frame_overrun_ms(self) -> list[float]:
        """
        Calculates total 'freeze' time per run form frame overruns.
        """
        if not self.frame_overrun_ms:
            return []

        result: list[float] = []
        for run in self.frame_overrun_ms.runs:
            freeze_ms: float = 0.0
            for ft in run:
                if ft > 0.0:
                    freeze_ms += ft
            result.append(freeze_ms)

        return result

    def _calc_freeze_frame_duration_ms(self, target: float) -> list[float]:
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
class MemoryUsageMetric(MetricBase):
    sampling_mode: MemorySamplingMode
    """
    There are two modes for measurement - Last, which represents the last observed
    value during an iteration, and Max, which represents the largest sample observed per measurement.
    """

    memory_rss_anon_kb: Measurement
    """
    memoryRssAnonKb - Anonymous resident/allocated memory owned by the process,
    not including memory mapped files or shared memory.
    """

    memory_rss_file_kb: Measurement
    """ memoryRssFileKb - Memory allocated by the process to map files. """

    memory_heap_size_kb: Measurement
    """ memoryHeapSizeKb - Heap memory allocations from the Android Runtime, sampled after each GC. """

    memory_gpu_kb: Measurement | None
    """ memoryGpuKb - GPU Memory allocated for the process. """

    @property
    def memory_rss_total_kb(self) -> Measurement:
        """memoryRSSTotalKb - Total RSS memory usage."""
        return Measurement(
            metadata=MeasurementMetadata(
                name=f"Total RSS Memory Usage {self.sampling_mode.name.title()}",
                name_short=f"MEM_RSS_{self.sampling_mode.name.upper()}",
                unit="Kb",
            ),
            runs=[
                anon + file
                for anon, file in zip(self.memory_rss_anon_kb.runs, self.memory_rss_file_kb.runs)
            ],
        )

    def _get_derived_measurements(self) -> list[Measurement]:
        return [self.memory_rss_total_kb]


@dataclass
class Benchmark:
    name: str
    class_name: str
    total_runtime_ns: int
    warmup_iterations: int
    repeat_iterations: int
    metrics: list[MetricBase]

    @property
    def id(self) -> str:
        return f"{self.class_name}#{self.name}"

    @staticmethod
    def aggregate(benchmarks: list[Benchmark], function: str) -> Benchmark:
        if not benchmarks:
            raise ValueError(f"Cannot aggregate benchmarks using '{function}': no benchmarks were provided.")

        ref_bench = benchmarks[0]

        total_runtime_ns = 0
        warmup_iterations = 0
        repeat_iterations = 0
        metrics_by_type: dict[type, list[MetricBase]] = defaultdict(list)

        for bench in benchmarks:
            if not ref_bench.is_same_benchmark(bench):
                raise ValueError("Cannot aggregate different types of benchmarks")

            total_runtime_ns += bench.total_runtime_ns
            warmup_iterations += bench.warmup_iterations
            repeat_iterations += bench.repeat_iterations

            for metric_type, metric in bench.get_metrics_by_type().items():
                metrics_by_type[metric_type].append(metric)

        metrics_agg: list[MetricBase] = []
        for metrics in metrics_by_type.values():
            metrics_agg.append(MetricBase.aggregate(metrics, function))

        return Benchmark(
            name=ref_bench.name,
            class_name=ref_bench.class_name,
            total_runtime_ns=total_runtime_ns,
            warmup_iterations=warmup_iterations,
            repeat_iterations=repeat_iterations,
            metrics=metrics_agg,
        )

    def get_metrics_by_type(self) -> dict[type, MetricBase]:
        return {type(metric): metric for metric in self.metrics}

    def is_same_benchmark(self, other: Self) -> bool:
        return self.id == other.id


@dataclass
class Device:
    brand: str = ""
    name: str = ""
    alias: str = ""
    model: str = ""
    sdk: int = 0
    sdk_codename: str = ""
    cpu_cores: int = 0
    cpu_freq: int = 0
    cpu_locked: bool = True
    mem_size_bytes: int = 0
    emulated: bool = True

    @property
    def mem_size_mb(self) -> int:
        return self.mem_size_bytes // (1000 * 1000)


@dataclass
class BenchmarkReport:
    device: Device
    benchmarks: list[Benchmark]
    filepath: Path | None = None


class Verdict(Enum):
    NOT_SIGNIFICANT = 0
    IMPROVEMENT = 1
    REGRESSION = 2


@dataclass
class CompareResult:
    metadata: CompareFunctionMetadata
    statistic: float
    verdict: Verdict


@dataclass
class MeasurementComparisonResult:
    a: Measurement
    b: Measurement
    result: list[CompareResult]

    @property
    def metadata(self) -> MeasurementMetadata:
        return self.a.metadata


@dataclass
class BenchmarkComparisonResult:
    a: Benchmark
    b: Benchmark
    results: list[MeasurementComparisonResult]
    thresholds: dict[str, float]

    @property
    def benchmark_id(self) -> str:
        assert self.a.id == self.b.id
        return self.a.id

    @property
    def benchmark_name(self) -> str:
        assert self.a.name == self.b.name
        return self.a.name

    @property
    def benchmark_class(self) -> str:
        assert self.a.class_name == self.b.class_name
        return self.a.class_name

    def has_regressed(self, compare_method: str | None = None) -> bool:
        for mesurement_result in self.results:
            for compare_result in mesurement_result.result:
                if compare_method and compare_result.metadata.id != compare_method:
                    continue

                if compare_result.verdict == Verdict.REGRESSION:
                    return True

        return False


@dataclass
class AnalysisReport:
    title: str = ""
    methods: list[CompareFunctionMetadata] = field(default_factory=list)
    baseline_reports: list[BenchmarkReport] = field(default_factory=list)
    candidate_reports: list[BenchmarkReport] = field(default_factory=list)
    comparisons: list[BenchmarkComparisonResult] = field(default_factory=list)

    def get_devices(self) -> list[Device]:
        unique_devices: list[Device] = []

        devices = [r.device for r in self.baseline_reports + self.candidate_reports]
        for device in devices:
            if device not in unique_devices:
                unique_devices.append(device)

        return unique_devices


def calculate_step_fit_score(a: list[float], b: list[float]) -> float:
    """
    Calculates the step fit between two distributions.

    Args:
        a: The reference (baseline) data points.
        b: The candidate (new) data points.

    Returns:
        The step fit score. A positive value indicates 'b' is lower than 'a'
        potential improvement, while a negative value
        indicates 'b' is higher than 'a' (potential regression).
        Returns 0.0 if either list is empty or the pooled error is zero.
    """

    def sum_squared_error(values: list[float]) -> float:
        avg = sum(values) / len(values)
        return sum((v - avg) ** 2 for v in values)

    if not a or not b:
        return 0.0

    total_squared_error = sum_squared_error(a) + sum_squared_error(b)
    step_error = math.sqrt(total_squared_error) / (len(a) + len(b))
    if step_error == 0.0:
        return 0.0

    return (sum(a) / len(a) - sum(b) / len(b)) / step_error


def compare_measurement(
    a: Measurement,
    b: Measurement,
    methods: set[str],
    thresholds: dict[str, float] = {},
) -> MeasurementComparisonResult:
    """
    Executes a statistical significance test between two samples.

    Args:
        method: "stepfit" or "mannwhitneyu".
        a_runs: Baseline data sample.
        b_runs: Candidate data sample.
        threshold: The significance cutoff (e.g., alpha for p-value).

    Returns:
        A tuple of (Verdict, result_value), where result_value is
        either the step fit score or the p-value.
    """
    results: list[CompareResult] = []
    for method in methods:
        if method not in thresholds:
            raise ValueError(f"Unknown method '{method}'. Available methods: {', '.join(COMPARE_METHODS.keys())}.")

        threshold = thresholds[method] if thresholds else DEFAULT_THRESHOLDS[method]
        verdict: Verdict = Verdict.NOT_SIGNIFICANT
        statistic = 0.0
        match method:
            case "stepfit":
                statistic = calculate_step_fit_score(a.runs, b.runs)
                if abs(statistic) < threshold:
                    verdict = Verdict.NOT_SIGNIFICANT
                elif statistic < 0:
                    verdict = Verdict.REGRESSION
                else:
                    verdict = Verdict.IMPROVEMENT
            case "mannwhitneyu":
                left_tail_test = mannwhitneyu(
                    a.runs,
                    b.runs,
                    alternative="less",
                )
                right_tail_test = mannwhitneyu(
                    a.runs,
                    b.runs,
                    alternative="greater",
                )
                if left_tail_test.pvalue < threshold:
                    statistic = left_tail_test.pvalue
                    verdict = Verdict.REGRESSION
                elif right_tail_test.pvalue < threshold:
                    statistic = right_tail_test.pvalue
                    verdict = Verdict.IMPROVEMENT
                else:
                    verdict = Verdict.NOT_SIGNIFICANT
                    statistic = min(left_tail_test.pvalue, right_tail_test.pvalue)
            case _:
                raise ValueError(f"Unknown comparison method: {method}")

        results.append(
            CompareResult(
                metadata=COMPARE_METHODS[method],
                statistic=statistic,
                verdict=verdict,
            )
        )

    return MeasurementComparisonResult(a=a, b=b, result=results)


def compare_metric(
    a: MetricBase,
    b: MetricBase,
    methods: set[str],
    thresholds: dict[str, float],
    measures: set[str] = set(),
) -> list[MeasurementComparisonResult]:
    a_measurements = a.get_measurements()
    b_measurements = b.get_measurements()

    results = []
    for a_measure, b_measure in zip(a_measurements, b_measurements):
        measure_id = a_measure.metadata.name_short
        if measures and measure_id not in measures:
            continue

        results.append(
            compare_measurement(
                a_measure,
                b_measure,
                methods=methods,
                thresholds=thresholds,
            )
        )

    return results


def compare_benchmark(
    a: Benchmark,
    b: Benchmark,
    methods: set[str],
    thresholds: dict[str, float],
    measures: set[str] = set()
) -> BenchmarkComparisonResult:
    a_metrics_by_type = a.get_metrics_by_type()
    b_metrics_by_type = b.get_metrics_by_type()
    common_metrics = a_metrics_by_type.keys() & b_metrics_by_type.keys()

    results = [ ]
    for metric_type in common_metrics:
        a_metric = a_metrics_by_type[metric_type]
        b_metric = b_metrics_by_type[metric_type]

        result = compare_metric(
            a_metric,
            b_metric,
            methods=methods,
            thresholds=thresholds,
            measures=measures,
        )

        results += result

    return BenchmarkComparisonResult(a=a, b=b, results=results, thresholds=thresholds)


def compare_benchmarks(
    a: list[Benchmark],
    b: list[Benchmark],
    methods: set[str],
    thresholds: dict[str, float],
    measures: set[str] = set(),
) -> list[BenchmarkComparisonResult]:
    a_benchmarks_by_id = { benchmark.id: benchmark for benchmark in a }
    b_benchmarks_by_id = { benchmark.id: benchmark for benchmark in b }

    a_only_benchmarks = a_benchmarks_by_id.keys() - b_benchmarks_by_id.keys()
    if a_only_benchmarks:
        logger.warning(f"found {len(a_only_benchmarks)} benchmarks in A, but not in B, skipping {a_only_benchmarks}")

    b_only_benchmarks = b_benchmarks_by_id.keys() - a_benchmarks_by_id.keys()
    if b_only_benchmarks:
        logger.warning(f"found {len(b_only_benchmarks)} benchmarks in B, but not in A, skipping {b_only_benchmarks}")

    common_benchmarks = a_benchmarks_by_id.keys() & b_benchmarks_by_id.keys()
    results: list[BenchmarkComparisonResult] = []
    for benchmark_id in common_benchmarks:
        results.append(
            compare_benchmark(
                a=a_benchmarks_by_id[benchmark_id],
                b=b_benchmarks_by_id[benchmark_id],
                methods=methods,
                thresholds=thresholds,
                measures=measures,
            )
        )

    return results
