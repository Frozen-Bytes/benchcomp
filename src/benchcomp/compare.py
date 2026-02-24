from __future__ import annotations

import logging
import math
from itertools import chain
from statistics import mean, median
from typing import Any, Callable

from scipy.stats import mannwhitneyu

from benchcomp.common import (
    Benchmark,
    BenchmarkComparisonResult,
    FrameTimingMetric,
    MemoryUsageMetric,
    Metric,
    MetricMetadata,
    StartupTimingMetric,
    Verdict,
)

logger = logging.getLogger(__name__)

AggregationFunction = Callable[[list[list[float]]], list[float]]

DEFAULT_STEP_FIT_THRESHOLD: float = 25.0
DEFAULT_P_VALUE_THRESHOLD: float = 0.01
DEFAULT_FRAME_TIME_TARGET_MS: float = 1000 / 60  # ~16.667 ms (60 FPS)
DEFAULT_AGGREGATE_FUNCTION = "none"

COMPARE_METHODS: dict[str, dict[str, str]] = {
    "stepfit": {
        "header": "Step Fit",
        "state": "fit",
    },
    "mannwhitneyu": {
        "header": "Mann-Whitney U-Test",
        "state": "pval",
    },
}


def _aggregate_none(runs: list[list[float]]) -> list[float]:
    """Return the single run as-is; fail if multiple runs exist."""
    if len(runs) != 1:
        raise ValueError("Aggregation 'none' requires exactly one benchmark run")
    return runs[0]

AGGREGATE_FUNCTIONS: dict[str, AggregationFunction] = {
    "none": _aggregate_none,
    "concat": lambda runs: list(chain.from_iterable(runs)),
    "min": lambda runs: [min(run) for run in runs],
    "max": lambda runs: [max(run) for run in runs],
    "median": lambda runs: [median(run) for run in runs],
    "mean": lambda runs: [mean(run) for run in runs],
}


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


def _extract_benchmark_runs(
    bench: Benchmark,
    frame_time_target: float,
) -> tuple[list[float], MetricMetadata]:
    """
    Extracts numerical samples and metadata from a raw Benchmark object.

    This function acts as an adapter, converting various benchmark metric types
    into a uniform list of floats.

    Logic per Metric:
    * StartupTimingMetric: Uses 'time_to_initial_display_ms'.
    * FrameTimingMetric: Calculates Freeze Frame Duration (FFD) based on
        the provided 'frame_time_target'.
    * MemoryUsageMetric: Sums 'RSS_Anon' and 'RSS_File' for each run to
        derive total resident memory usage.

    Args:
        bench: The Benchmark instance to parse.
        frame_time_target: The frame budget target (ms) used for FFD calculation.

    Returns:
        A tuple containing (list of numerical values, MetricMetadata).
    """
    data = bench.data

    if isinstance(data, StartupTimingMetric):
        return data.time_to_initial_display_ms.runs, data.time_to_initial_display_ms.metadata

    if isinstance(data, FrameTimingMetric):
        metadata = MetricMetadata(name="Freeze Frame Duration", name_short="FFD", unit="ms")
        return data.calc_freeze_frame_duration_ms(frame_time_target), metadata

    if isinstance(data, MemoryUsageMetric):
        metadata = MetricMetadata(name="Total Memory Usage", name_short="MEMU", unit="Kb")
        runs: list[float] = [
            anon + file
            for anon, file in zip(data.memory_rss_anon_kb.runs, data.memory_rss_file_kb.runs)
        ]
        return runs, metadata

    return [], MetricMetadata(name="Unknown", name_short="Unknown", unit="")


def _aggregate_runs(runs: list[list[float]], function: str) -> list[float]:
    """
    Reduces multiple experimental runs into a single representative sample.

    Aggregation Strategies:
    * 'none': Validates that only one run exists and returns it.
    * 'concat': Merges all run data into one large distribution.
    * 'mean'/'median'/'min'/'max': Reduces each individual run to a
        single scalar, returning a list of scalars.

    Args:
        runs: A list of lists, where each inner list represents one benchmark run.
        function: The strategy string (e.g., "median", "concat").

    Raises:
        ValueError: If 'none' is chosen with >1 run or if the method is unknown.
    """
    if function not in AGGREGATE_FUNCTIONS.keys():
        raise ValueError(f"Unknown aggregation method: {function}")

    aggregator = AGGREGATE_FUNCTIONS[function]
    return aggregator(runs)


def _evaluate_statistical_significance(
    method: str,
    a_runs: list[float],
    b_runs: list[float],
    threshold: float,
    *args,
    **kwargs,
) -> tuple[Verdict, Any]:
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
    verdict: Verdict = Verdict.NOT_SIGNIFICANT
    test_result = None
    match method:
        case "stepfit":
            test_result = calculate_step_fit_score(a_runs, b_runs, *args, **kwargs)
            if abs(test_result) < threshold:
                verdict = Verdict.NOT_SIGNIFICANT
            elif test_result < 0:
                verdict = Verdict.REGRESSION
            else:
                verdict = Verdict.IMPROVEMENT
        case "mannwhitneyu":
            left_tail_test = mannwhitneyu(
                a_runs,
                b_runs,
                alternative="less",
                method="exact",
                *args,
                **kwargs,
            )
            right_tail_test = mannwhitneyu(
                a_runs,
                b_runs,
                alternative="greater",
                method="exact",
                *args,
                **kwargs,
            )
            if (left_tail_test.pvalue < threshold) and (right_tail_test.pvalue < threshold):
                # Note: not sure this is correct interpretation of the p-value
                test_result = min(left_tail_test.pvalue, right_tail_test.pvalue)
                verdict = Verdict.NOT_SIGNIFICANT
            elif left_tail_test.pvalue < threshold:
                test_result = left_tail_test.pvalue
                verdict = Verdict.REGRESSION
            elif right_tail_test.pvalue < threshold:
                test_result = right_tail_test.pvalue
                verdict = Verdict.IMPROVEMENT
            else:
                verdict = Verdict.NOT_SIGNIFICANT
                test_result = min(left_tail_test.pvalue, right_tail_test.pvalue)
        case _:
            raise ValueError(f"Unknown comparison method: {method}")

    return (verdict, test_result)


def compare_benchmarks(
    a: Benchmark | list[Benchmark],
    b: Benchmark | list[Benchmark],
    comparison_method: str,
    threshold: float,
    frame_time_target: float = DEFAULT_FRAME_TIME_TARGET_MS,
    aggregation_function: str = "none",
    *args,
    **kwargs,
) -> BenchmarkComparisonResult | None:
    """
    Args:
        a: Baseline benchmark(s).
        b: Candidate benchmark(s).
        method: Statistical test to use ("stepfit" or "mannwhitneyu").
        threshold: The significance threshold.
        frame_time_target: The MS target for frame-based metrics.
        aggregation_method: The method to combine multiple runs.

    Returns:
        A BenchmarkComparisonResult if successful, or None if data collection failed.
    """
    def collect_benchmark_runs(benchmarks: list[Benchmark]) -> tuple[list[list[float]], MetricMetadata]:
        concatenated_runs = []
        metric_metadata = MetricMetadata("Unknown", "Unknown", "")
        for b in benchmarks:
            runs, metric_metadata = _extract_benchmark_runs(b, frame_time_target)
            if runs:
                concatenated_runs.append(runs)
            else:
                logger.warning(f"unknown benchmark '{bench.name}' type, skipping. (type: {type(bench.data)})")
        return concatenated_runs, metric_metadata

    a_benchmarks = [a] if isinstance(a, Benchmark) else a
    b_benchmarks = [b] if isinstance(b, Benchmark) else b

    ref_bench = a_benchmarks[0]
    for bench in a_benchmarks + b_benchmarks:
        if not ref_bench.is_same_benchmark(bench):
            raise ValueError(f"benchmark mismatch expected: {ref_bench.name}, found: {bench.name}")

    a_runs, metric_metadata = collect_benchmark_runs(a_benchmarks)
    b_runs, _ = collect_benchmark_runs(b_benchmarks)
    if not a_runs or not b_runs:
        logger.error(f"no valid runs found for {ref_bench.name}")
        return None

    try:
        a_aggregated_runs = _aggregate_runs(a_runs, function=aggregation_function)
        b_aggregated_runs = _aggregate_runs(b_runs, function=aggregation_function)

        verdict, comparison_result = _evaluate_statistical_significance(
            method=comparison_method,
            a_runs=a_aggregated_runs,
            b_runs=b_aggregated_runs,
            threshold=threshold,
            *args,
            **kwargs,
        )
    except Exception:
        logger.exception(f"failed to compare benchmark '{ref_bench.name}' metric '{metric_metadata.name}', skipping")
        return None

    return BenchmarkComparisonResult(
        a_bench_ref=a_benchmarks,
        b_bench_ref=b_benchmarks,
        a_metric=Metric(metadata=metric_metadata, _runs=a_aggregated_runs),
        b_metric=Metric(metadata=metric_metadata, _runs=b_aggregated_runs),
        comparison_method=comparison_method,
        comparison_result=comparison_result,
        verdict=verdict,
    )
