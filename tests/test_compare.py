from dataclasses import dataclass

import pytest

from benchcomp.core import (
    Benchmark,
    Measurement,
    MeasurementMetadata,
    MetricBase,
    Verdict,
    compare_benchmark,
    compare_benchmarks,
    compare_measurement,
    compare_metric,
)


@dataclass
class DummyMetric(MetricBase):
    m1: Measurement
    m2: Measurement | None = None


def make_benchmark(
    name: str = "DummyBenchmark",
    class_name: str = "",
    metrics: list[MetricBase] = [],
) -> Benchmark:
    return Benchmark(
        name=name,
        class_name=class_name,
        total_runtime_ns=1,
        warmup_iterations=0,
        repeat_iterations=1,
        metrics=metrics,
    )


def make_measurement(name: str, runs: list[float]) -> Measurement:
    return Measurement(
        metadata=MeasurementMetadata(
            name=name,
            name_short=name,
            unit="u",
        ),
        runs=runs,
    )


def test_compare_measurement_stepfit_improvement():
    a = make_measurement("m", [100, 110, 120])
    b = make_measurement("m", [10, 20, 30])

    result = compare_measurement(
        a,
        b,
        methods={"stepfit"},
        thresholds={"stepfit": 1.0},
    )

    assert len(result.result) == 1

    compare_result = result.result[0]
    assert compare_result.verdict == Verdict.IMPROVEMENT


def test_compare_measurement_stepfit_regression():
    a = make_measurement("m", [10, 20, 30])
    b = make_measurement("m", [100, 110, 120])

    result = compare_measurement(
        a,
        b,
        methods={"stepfit"},
        thresholds={"stepfit": 1.0},
    )

    assert len(result.result) == 1

    compare_result = result.result[0]
    assert compare_result.verdict == Verdict.REGRESSION


def test_compare_measurement_stepfit_not_significant():
    a = make_measurement("m", [1, 2, 3])
    b = make_measurement("m", [1, 2, 3])

    result = compare_measurement(
        a,
        b,
        methods={"stepfit"},
        thresholds={"stepfit": 1.0},
    )

    assert len(result.result) == 1

    compare_result = result.result[0]
    assert compare_result.verdict == Verdict.NOT_SIGNIFICANT


def test_compare_measurement_mannwhitneyu_improvement():
    a = make_measurement("m", [100, 110, 120, 130])
    b = make_measurement("m", [10, 20, 30, 40])

    result = compare_measurement(
        a,
        b,
        methods={"mannwhitneyu"},
        thresholds={"mannwhitneyu": 0.05},
    )

    assert len(result.result) == 1

    compare_result = result.result[0]
    assert compare_result.verdict == Verdict.IMPROVEMENT


def test_compare_measurement_mannwhitneyu_regression():
    a = make_measurement("m", [10, 20, 30, 40])
    b = make_measurement("m", [100, 110, 120, 130])

    result = compare_measurement(
        a,
        b,
        methods={"mannwhitneyu"},
        thresholds={"mannwhitneyu": 0.05},
    )

    assert len(result.result) == 1

    compare_result = result.result[0]
    assert compare_result.verdict == Verdict.REGRESSION


def test_compare_measurement_mannwhitneyu_not_significant():
    a = make_measurement("m", [1, 2, 3])
    b = make_measurement("m", [1, 2, 3])

    result = compare_measurement(
        a,
        b,
        methods={"mannwhitneyu"},
        thresholds={"mannwhitneyu": 0.01},
    )

    assert len(result.result) == 1

    compare_result = result.result[0]
    assert compare_result.verdict == Verdict.NOT_SIGNIFICANT


def test_compare_metric_multiple_measurements():
    m1_a = make_measurement("A", [100, 110, 120])
    m1_b = make_measurement("A", [10, 20, 30])

    m2_a = make_measurement("B", [5, 6, 7])
    m2_b = make_measurement("B", [5, 6, 7])

    metric_a = DummyMetric(m1=m1_a, m2=m2_a)
    metric_b = DummyMetric(m1=m1_b, m2=m2_b)

    results = compare_metric(
        metric_a,
        metric_b,
        methods={"stepfit"},
        thresholds={"stepfit": 1.0},
    )

    assert len(results) == 2
    assert results[0].result[0].verdict == Verdict.IMPROVEMENT
    assert results[1].result[0].verdict == Verdict.NOT_SIGNIFICANT


def test_compare_metric_measure_filter():
    m1_a = make_measurement("A", [100, 110, 120])
    m1_b = make_measurement("A", [10, 20, 30])

    m2_a = make_measurement("B", [5, 6, 7])
    m2_b = make_measurement("B", [5, 6, 7])

    metric_a = DummyMetric(m1=m1_a, m2=m2_a)
    metric_b = DummyMetric(m1=m1_b, m2=m2_b)

    results = compare_metric(
        metric_a,
        metric_b,
        methods={"stepfit"},
        thresholds={"stepfit": 1.0},
        measures={"A"},
    )

    assert len(results) == 1
    assert results[0].a.metadata.name_short == "A"


def test_compare_benchmark():
    m_a = make_measurement("A", [100, 110, 120])
    m_b = make_measurement("A", [10, 20, 30])

    metric_a = DummyMetric(m1=m_a)
    metric_b = DummyMetric(m1=m_b)

    bench_a = make_benchmark(metrics=[metric_a])
    bench_b = make_benchmark(metrics=[metric_b])

    result = compare_benchmark(
        bench_a,
        bench_b,
        methods={"stepfit"},
        thresholds={"stepfit": 1.0},
    )

    assert len(result.results) == 1
    measurement_comp_result = result.results[0]
    assert len(measurement_comp_result.result) == 1
    comp_result = measurement_comp_result.result[0]
    assert comp_result.verdict == Verdict.IMPROVEMENT


def test_compare_benchmarks_multiple():
    m_a = make_measurement("A", [100, 110, 120])
    m_b = make_measurement("A", [10, 20, 30])

    metric_a = DummyMetric(m1=m_a)
    metric_b = DummyMetric(m1=m_b)

    bench_a1 = make_benchmark("bench1", metrics=[metric_a, metric_b])
    bench_a2 = make_benchmark("bench2", metrics=[metric_a])

    bench_b1 = make_benchmark("bench1", metrics=[metric_a, metric_b])

    results = compare_benchmarks(
        a=[bench_a1, bench_a2],
        b=[bench_b1],
        methods={"stepfit"},
        thresholds={"stepfit": 1.0},
    )

    # bench2 exists only in A -> skipped
    assert len(results) == 1
    assert results[0].benchmark_id == bench_a1.id


def test_aggregate_sums_scalar_fields_and_groups_metrics(monkeypatch):
    m_a = make_measurement("A", [10, 20, 30])
    m_b = make_measurement("A", [100, 200, 300])

    metric_a = DummyMetric(m1=m_a)
    metric_b = DummyMetric(m1=m_b)

    bench1 = make_benchmark(metrics=[metric_a])
    bench2 = make_benchmark(metrics=[metric_b])

    result = Benchmark.aggregate([bench1, bench2], function="max")

    assert result.id == bench1.id
    assert result.id == bench2.id
    assert result.name == bench1.name
    assert result.name == bench2.name
    assert result.class_name == bench1.class_name
    assert result.class_name == bench2.class_name
    assert result.warmup_iterations == bench1.warmup_iterations + bench2.warmup_iterations
    assert result.repeat_iterations == bench1.repeat_iterations + bench2.repeat_iterations
    assert result.total_runtime_ns == bench1.total_runtime_ns + bench2.total_runtime_ns

    assert len(result.metrics) == 1

    measurements = result.metrics[0].get_measurements()
    assert len(measurements) == 1
    assert measurements[0].metadata == m_a.metadata
    assert measurements[0].runs == [30, 300]


def test_aggregate_empty_list():
    with pytest.raises(ValueError, match="no benchmarks"):
        Benchmark.aggregate([], function="sum")


def test_aggregate_mismatched_benchmarks():
    bench1 = make_benchmark(name="BenchA")
    bench2 = make_benchmark(name="BenchB")  # different id

    with pytest.raises(ValueError, match="different types"):
        Benchmark.aggregate([bench1, bench2], function="sum")
