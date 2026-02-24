import json
import logging
from pathlib import Path
from typing import Any

from benchcomp.common import (
    Benchmark,
    BenchmarkReport,
    Device,
    FrameTimingMetric,
    MemoryMetricMode,
    MemoryUsageMetric,
    Metric,
    MetricMetadata,
    SampledMetric,
    StartupTimingMetric,
)

logger = logging.getLogger(__name__)

def _parse_device(data: dict[str, Any]) -> Device:
    build = data.get("build", {})

    return Device(
        brand = build.get("brand", ""),
        name = build.get("device", ""),
        model = build.get("model", ""),
        cpu_cores = data.get("cpuCoreCount", 0),
        cpu_freq = data.get("cpuMaxFreqHz", 0),
        mem_size = data.get("memTotalBytes", 0) // (1024 * 1024),
        emulated=True
    )

def _parse_metric(data: dict[str, Any], name: str, name_short: str, unit: str) -> Metric:
    return Metric(
        MetricMetadata(
            name=name,
            name_short=name_short,
            unit=unit,
        ),
        _runs=data.get("runs", []),
    )

def _parse_sampled_metric(data: dict[str, Any]) -> SampledMetric:
    return SampledMetric(
        p50=data.get("P50", 0.0),
        p90=data.get("P90", 0.0),
        p95=data.get("P95", 0.0),
        p99=data.get("P99", 0.0),
        runs=data.get("runs", []),
    )

def _parse_startup_timing_metric(data: dict[str, Any]) -> StartupTimingMetric | None:
    metrics = data.get("metrics", {})
    time_to_full_display_ms: Metric | None = None

    if "timeToFullDisplayMs" in metrics:
        time_to_full_display_ms = _parse_metric(
            metrics.get("timeToFullDisplayMs"),
            name="Time to Full Display",
            name_short="TFD",
            unit="ms"
        )

    return StartupTimingMetric(
        time_to_initial_display_ms=_parse_metric(
            metrics["timeToInitialDisplayMs"],
            name="Time to Initial Display",
            name_short="TID",
            unit="ms"
        ),
        time_to_full_display_ms=time_to_full_display_ms,
    )

def _parse_frame_timing_metric(data: dict[str, Any]) -> FrameTimingMetric | None:
    metrics = data.get("metrics", {})
    sampled_metrics = data.get("sampledMetrics", {})
    frame_overrun_ms: SampledMetric | None = None

    if "frameOverrunMs" in sampled_metrics:
        frame_overrun_ms = _parse_sampled_metric(sampled_metrics.get("frameOverrunMs"))

    return FrameTimingMetric(
        frame_count=_parse_metric(
            metrics.get("frameCount", {}),
            name="Frame Count",
            name_short="FC",
            unit=""
        ),
        frame_duration_ms=_parse_sampled_metric(sampled_metrics.get("frameDurationCpuMs", {})),
        frame_overrun_ms=frame_overrun_ms,
    )

def _parse_memory_usage_metric(data: dict[str, Any]) -> MemoryUsageMetric | None:
    metrics = data.get("metrics", {})

    METRIC_CONFIGS = [
        ("memoryRssAnon", "Memory Resident Set Size Anonymous", "MEM_RSS_ANON"),
        ("memoryRssFile", "Memory Resident Set Size File", "MEM_RSS_FILE"),
        ("memoryHeapSize", "Memory Heap Size", "MEM_HEAP_SIZE"),
        ("memoryGpu", "Memory GPU", "MEM_GPU"),
    ]
    mode = MemoryMetricMode.MAX if "memoryRssAnonMaxKb" in metrics else MemoryMetricMode.LAST
    suffix = "MaxKb" if mode == MemoryMetricMode.MAX else "LastKb"

    results: dict[str, Metric] = {}
    for key, full_name, short_name, in METRIC_CONFIGS:
        metric_key = f"{key}{suffix}"
        results[key] = _parse_metric(
            metrics.get(metric_key),
            name=f"{full_name} {str(mode.name).title()}",
            name_short=f"{short_name}_{str(mode.name).upper()}",
            unit="Kb",
        )

    if any(results[k] is None for k in ["memoryRssAnon", "memoryRssFile", "memoryHeapSize"]):
            return None

    return MemoryUsageMetric(
        mode=mode,
        memory_rss_anon_kb=results["memoryRssAnon"],
        memory_rss_file_kb=results["memoryRssFile"],
        memory_heap_size_kb=results["memoryHeapSize"],
        memory_gpu_kb=results["memoryGpu"],
    )

def _parse_benchmark(data: dict[str, Any]) -> Benchmark | None:
    name: str = data.get("name", "")
    metrics_json = data.get("metrics", {})

    METRIC_PARSER_MAP = {
        "timeToInitialDisplayMs": _parse_startup_timing_metric,
        "frameCount": _parse_frame_timing_metric,
        "memoryRssAnonMaxKb": _parse_memory_usage_metric,
        "memoryRssAnonLastKb": _parse_memory_usage_metric,
    }

    bench_data: StartupTimingMetric | FrameTimingMetric | MemoryUsageMetric | None = None
    for key, parser in METRIC_PARSER_MAP.items():
        if key in metrics_json:
            bench_data = parser(data)
            break

    if bench_data is None:
        logger.warning(f"unable to detect benchmark '{name}' metric type, skipping")
        return None

    return Benchmark(
        name=name,
        class_name=data.get("className", ""),
        total_run_time_ns=data.get("totalRunTimeNs", 0),
        warmup_iterations=data.get("warmupIterations", 0),
        repeat_iterations=data.get("repeatIterations", 0),
        data=bench_data,
    )


def parse_macrobechmark_report(path: Path | str) -> BenchmarkReport | None:
    report: BenchmarkReport = BenchmarkReport(filepath=Path(path))
    with open(path, "r") as file:
        try:
            root = json.load(file)
            report.device = _parse_device(root.get("context", {}))
            benchmarks = root.get("benchmarks", [])
            for bench_obj in benchmarks:
                try:
                    benchmark = _parse_benchmark(bench_obj)
                    if benchmark is not None:
                        report.benchmarks[benchmark.name] = benchmark
                except Exception as e:
                    name = bench_obj.get("name", "")
                    logger.warning(f"failed to parse benchmark '{name}', skipping. ({e})")

        except json.JSONDecodeError | UnicodeDecodeError:
            logger.error(f"failed to parse json file '{path}', invalid JSON document")
            return None

    return report
