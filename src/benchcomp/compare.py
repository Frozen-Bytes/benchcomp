import logging
import math
from dataclasses import dataclass
from enum import Enum
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


class Verdict(Enum):
    NOT_SIGNIFICANT = 0
    IMPROVEMENT = 1
    REGRESSION = 2


@dataclass
class BenchmarkCompareResult:
    a_bench_ref: Benchmark
    b_bench_ref: Benchmark
    a_metric: Metric
    b_metric: Metric
    method: str
    verdict: Verdict
    result: Any

    def get_benchmark_name(self) -> str:
        return self.a_bench_ref.name

    def get_benchmark_class(self) -> str:
        return self.a_bench_ref.class_name

    def get_metric_metadata(self) -> MetricMetadata:
        return self.a_metric.metadata

    def is_compatible_with(self, other: Self) -> bool:
        return (
            (self.a_bench_ref.is_same_benchmark(other.a_bench_ref))
            and (self.b_bench_ref.is_same_benchmark(other.b_bench_ref))
            and (self.a_metric.metadata == other.a_metric.metadata)
            and (self.b_metric.metadata == other.b_metric.metadata)
            and (self.method == other.method)
            and (type(self.result) is type(other.result))
        )

DEFAULT_STEP_FIT_THRESHOLD: float = 25.0
DEFAULT_P_VALUE_THRESHOLD: float = 0.01
DEFAULT_FRAME_TIME_TARGET_MS: float = 1000 / 60

logger = logging.getLogger(__name__)


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


def compare_benchmark(
    a: Benchmark,
    b: Benchmark,
    method: str,
    threshold: float,
    frametime_target: float = DEFAULT_FRAME_TIME_TARGET_MS,
    *args,
    **kwargs,
) -> BenchmarkCompareResult | None:
    assert a.is_same_benchmark(b)

    if a.repeat_iterations != b.repeat_iterations:
        logger.warning(f"benchmark '{a.name}' iteration mismatch, (a: {a.repeat_iterations}, b: {b.repeat_iterations})")

    a_runs: list[float]
    b_runs: list[float]
    if isinstance(a.data, StartupTimingMetric) and isinstance(b.data, StartupTimingMetric):
        metadata = a.data.time_to_initial_display_ms.metadata
        a_runs = a.data.time_to_initial_display_ms.runs
        b_runs = b.data.time_to_initial_display_ms.runs
    elif isinstance(a.data, FrameTimingMetric) and isinstance(b.data, FrameTimingMetric):
        metadata = MetricMetadata(
            name="Freeze Frame Duration",
            name_short="FFD",
            unit="ms",
        )
        a_runs=a.data.calc_freeze_frame_duration_ms(frametime_target)
        b_runs=b.data.calc_freeze_frame_duration_ms(frametime_target)
    elif isinstance(a.data, MemoryUsageMetric) and isinstance(b.data, MemoryUsageMetric):
        metadata = MetricMetadata(
            name="Total Memory Usage",
            name_short="MEMU",
            unit="Kb",
        )
        a_runs=(a.data.memory_rss_anon_kb.runs + a.data.memory_rss_file_kb.runs)
        b_runs=(b.data.memory_rss_anon_kb.runs + b.data.memory_rss_file_kb.runs)
    else:
        logger.warning(f"benchmark '{a.name}' type mismatch or unknown, skipping. (a_type: {type(a.data)}, b_type: {type(b.data)})")
        return None

    a_metric = Metric(metadata=metadata, _runs=a_runs)
    b_metric = Metric(metadata=metadata, _runs=b_runs)

    verdict: Verdict = Verdict.NOT_SIGNIFICANT
    compare_result = None
    try:
        match method:
            case "stepfit":
                compare_result = step_fit(a_metric.runs, b_metric.runs, *args, **kwargs)
                if abs(compare_result) < threshold:
                    verdict = Verdict.NOT_SIGNIFICANT
                elif compare_result < 0:
                    verdict = Verdict.REGRESSION
                else:
                    verdict = Verdict.IMPROVEMENT
            case "mannwhitneyu":
                res_less = mannwhitneyu(
                    a_metric.runs,
                    b_metric.runs,
                    alternative="less",
                    method="exact",
                    *args,
                    **kwargs,
                )
                res_greater = mannwhitneyu(
                    a_metric.runs,
                    b_metric.runs,
                    alternative="greater",
                    method="exact",
                    *args,
                    **kwargs,
                )
                if (res_less.pvalue < threshold) and (res_greater.pvalue < threshold):
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
                assert False, "Unknown compare function"
    except Exception as e:
        logger.warning(f"failed to compare benchmark '{a.name}' metric '{metadata.name}', skipping. ({e})'")
        return None

    return BenchmarkCompareResult(
        a_bench_ref=a,
        b_bench_ref=b,
        a_metric=a_metric,
        b_metric=b_metric,
        verdict=verdict,
        method=method,
        result=compare_result,
    )
