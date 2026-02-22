import logging
import math
from dataclasses import dataclass
from enum import Enum
from itertools import chain
from statistics import mean, median
from typing import Any, Self

from scipy.stats import mannwhitneyu

from benchcomp.parser_common import (
    Benchmark,
    FrameTimingMetric,
    MemoryUsageMetric,
    Metric,
    MetricMetadata,
    StartupTimingMetric,
)

DEFAULT_STEP_FIT_THRESHOLD: float = 25.0
DEFAULT_P_VALUE_THRESHOLD: float = 0.01
DEFAULT_FRAME_TIME_TARGET_MS: float = 1000 / 60

AGGREGATE_METHODS: list[str] = [
    "none",
    "min",
    "max",
    "median",
    "concat",
]

logger = logging.getLogger(__name__)


class Verdict(Enum):
    NOT_SIGNIFICANT = 0
    IMPROVEMENT = 1
    REGRESSION = 2


@dataclass
class BenchmarkCompareResult:
    a_bench_ref: list[Benchmark]
    b_bench_ref: list[Benchmark]
    a_metric: Metric
    b_metric: Metric
    method: str
    verdict: Verdict
    result: Any

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
            and (self.method == other.method)
            and (type(self.result) is type(other.result))
        )

    @staticmethod
    def _calc_total_run_time_ms(benchmarks: list[Benchmark]) -> float:
        def to_seconds(time_ns: float):
            return time_ns / (1000 * 1000 * 1000)

        time:float = 0.0
        for b in benchmarks:
            time += to_seconds(b.total_run_time_ns)
        return time


def step_fit(a: list[float], b: list[float]) -> float:
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


def _extract_benchmark_runs(
    bench: Benchmark,
    frametime_target: float,
) -> tuple[list[float], MetricMetadata]:
    """Extracts runs and metadata based on the specific metric type."""
    data = bench.data

    if isinstance(data, StartupTimingMetric):
        return data.time_to_initial_display_ms.runs, data.time_to_initial_display_ms.metadata

    if isinstance(data, FrameTimingMetric):
        metadata = MetricMetadata(name="Freeze Frame Duration", name_short="FFD", unit="ms")
        return data.calc_freeze_frame_duration_ms(frametime_target), metadata

    if isinstance(data, MemoryUsageMetric):
        metadata = MetricMetadata(name="Total Memory Usage", name_short="MEMU", unit="Kb")
        runs: list[float] = [
            anon + file
            for anon, file in zip(data.memory_rss_anon_kb.runs, data.memory_rss_file_kb.runs)
        ]
        return runs, metadata

    return [], MetricMetadata(name="Unknown", name_short="Unknown", unit="")


def _apply_aggregation(runs: list[list[float]], method: str) -> list[float]:
    """Aggregates multiple runs into a single representative list."""
    if method == "none":
        if len(runs) != 1:
            raise ValueError("Aggregation 'none' requires exactly one benchmark run")
        return runs[0]

    if method == "concat":
        return list(chain.from_iterable(runs))

    dispatch = {
        "min": min,
        "max": max,
        "median": median,
        "mean": mean,
    }

    if method not in dispatch:
        raise ValueError(f"Unknown aggregation method: {method}")

    return [dispatch[method](run) for run in runs]


def _compare_runs(
    method: str,
    a_runs: list[float],
    b_runs: list[float],
    threshold: float,
    *args,
    **kwargs,
) -> tuple[Verdict, Any]:
    verdict: Verdict = Verdict.NOT_SIGNIFICANT
    compare_result = None
    match method:
        case "stepfit":
            compare_result = step_fit(a_runs, b_runs, *args, **kwargs)
            if abs(compare_result) < threshold:
                verdict = Verdict.NOT_SIGNIFICANT
            elif compare_result < 0:
                verdict = Verdict.REGRESSION
            else:
                verdict = Verdict.IMPROVEMENT
        case "mannwhitneyu":
            res_less = mannwhitneyu(
                a_runs,
                b_runs,
                alternative="less",
                method="exact",
                *args,
                **kwargs,
            )
            res_greater = mannwhitneyu(
                a_runs,
                b_runs,
                alternative="greater",
                method="exact",
                *args,
                **kwargs,
            )
            if (res_less.pvalue < threshold) and (res_greater.pvalue < threshold):
                # Note: not sure this is correct interpretation of the p-value
                compare_result = min(res_less.pvalue, res_greater.pvalue)
                verdict = Verdict.NOT_SIGNIFICANT
            elif res_less.pvalue < threshold:
                compare_result = res_less.pvalue
                verdict = Verdict.REGRESSION
            elif res_greater.pvalue < threshold:
                compare_result = res_greater.pvalue
                verdict = Verdict.IMPROVEMENT
            else:
                verdict = Verdict.NOT_SIGNIFICANT
                compare_result = min(res_less.pvalue, res_greater.pvalue)
        case _:
            raise ValueError(f"Unknown comparison method: {method}")

    return (verdict, compare_result)


def compare_benchmark(
    a: Benchmark | list[Benchmark],
    b: Benchmark | list[Benchmark],
    method: str,
    threshold: float,
    frametime_target: float = DEFAULT_FRAME_TIME_TARGET_MS,
    aggregate: str = "none",
    *args,
    **kwargs,
) -> BenchmarkCompareResult | None:
    def collect(benchmarks: list[Benchmark]) -> tuple[list[list[float]], MetricMetadata]:
        raw, meta = [], MetricMetadata("Unknown", "Unknown", "")
        for b in benchmarks:
            runs, meta = _extract_benchmark_runs(b, frametime_target)
            if runs:
                raw.append(runs)
            else:
                logger.warning(f"unknown benchmark '{bench.name}' type, skipping. (type: {type(bench.data)})")
        return raw, meta

    a_list = [a] if isinstance(a, Benchmark) else a
    b_list = [b] if isinstance(b, Benchmark) else b

    ref_bench = a_list[0]
    for bench in a_list + b_list:
        if not ref_bench.is_same_benchmark(bench):
            raise ValueError(f"benchmark mismatch expected: {ref_bench.name}, found: {bench.name}")

    a_raw, metadata = collect(a_list)
    b_raw, _ = collect(b_list)
    if not a_raw or not b_raw:
        logger.error(f"no valid runs found for {ref_bench.name}")
        return None

    try:
        a_final = _apply_aggregation(a_raw, aggregate)
        b_final = _apply_aggregation(b_raw, aggregate)

        verdict, compare_result = _compare_runs(
            method=method,
            a_runs=a_final,
            b_runs=b_final,
            threshold=threshold,
            *args,
            **kwargs,
        )
    except Exception as e:
        logger.warning(f"failed to compare benchmark '{ref_bench.name}' metric '{metadata.name}', skipping. ({e})'")
        return None

    return BenchmarkCompareResult(
        a_bench_ref=a_list,
        b_bench_ref=b_list,
        a_metric=Metric(metadata=metadata, _runs=a_final),
        b_metric=Metric(metadata=metadata, _runs=b_final),
        verdict=verdict,
        method=method,
        result=compare_result,
    )
