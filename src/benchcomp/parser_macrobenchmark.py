import json
import logging
from pathlib import Path
from typing import Any

from benchcomp.parser_common import (
    Benchmark,
    BenchmarkReport,
    Device,
    FrameTimingMetric,
    MemoryMetricMode,
    MemoryUsageMetric,
    Metric,
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
        _runs=data.get("runs", []),
        name=name,
        name_short=name_short,
        unit=unit,
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

    memory_mode: MemoryMetricMode = MemoryMetricMode.UNKNOWN
    memory_rss_anon_kb: Metric | None = None
    memory_rss_file_kb: Metric | None = None
    memory_heap_size_kb: Metric | None = None
    memory_gpu_kb: Metric | None = None

    if "memoryRssAnonMaxKb" in metrics:
        memory_mode = MemoryMetricMode.MAX
        memory_rss_anon_kb = _parse_metric(
            metrics.get("memoryRssAnonMaxKb"),
            name="Memory Resident Set Size Anonymous Max",
            name_short="MEM_RSS_ANON_MAX",
            unit="Kb"
        )
    elif "memoryRssAnonLastKb" in metrics:
        memory_mode = MemoryMetricMode.LAST
        memory_rss_anon_kb = _parse_metric(
            metrics.get("memoryRssAnonLastKb"),
            name="Memory Resident Set Size Anonymous Last",
            name_short="MEM_RSS_ANON_LAST",
            unit="Kb"
        )

    if "memoryRssFileMaxKb" in metrics:
        memory_rss_file_kb = _parse_metric(
            metrics.get("memoryRssFileMaxKb"),
            name="Memory Resident Set Size File Max",
            name_short="MEM_RSS_FILE_MAX",
            unit="Kb"
        )
    elif "memoryRssFileLastKb" in metrics:
        memory_rss_file_kb = _parse_metric(
            metrics.get("memoryRssFileLastKb"),
            name="Memory Resident Set Size File Last",
            name_short="MEM_RSS_FILE_Last",
            unit="Kb"
        )

    if "memoryHeapSizeMaxKb" in metrics:
        memory_heap_size_kb = _parse_metric(
            metrics.get("memoryHeapSizeMaxKb"),
            name="Memory Heap Size Max",
            name_short="MEM_HEAP_SIZE_MAX",
            unit="Kb"
        )
    elif "memoryHeapSizeLastKb" in metrics:
        memory_heap_size_kb = _parse_metric(
            metrics.get("memoryHeapSizeLastKb"),
            name="Memory Heap Size LAST",
            name_short="MEM_HEAP_SIZE_LAST",
            unit="Kb"
        )

    if "memoryGpuMaxKb" in metrics:
        memory_gpu_kb = _parse_metric(
            metrics.get("memoryGpuMaxKb"),
            name="Memory GPU Max",
            name_short="MEM_GPU_MAX",
            unit="Kb"
        )
    elif "memoryGpuLastKb" in metrics:
        memory_gpu_kb = _parse_metric(
            metrics.get("memoryGpuLastKb"),
            name="Memory GPU Last",
            name_short="MEM_GPU_LAST",
            unit="Kb"
        )

    if (
        memory_rss_anon_kb is None
        or memory_rss_file_kb is None
        or memory_heap_size_kb is None
    ):
        return None

    return MemoryUsageMetric(
        mode=memory_mode,
        memory_rss_anon_kb=memory_rss_anon_kb,
        memory_rss_file_kb=memory_rss_file_kb,
        memory_heap_size_kb=memory_heap_size_kb,
        memory_gpu_kb=memory_gpu_kb,
    )

def _parse_benchmark(data: dict[str, Any]) -> Benchmark | None:
    name: str = data.get("name", "")
    metrics_json = data.get("metrics", {})
    bench_data: StartupTimingMetric | FrameTimingMetric | MemoryUsageMetric | None = None

    # Detect if benchmark is of type StartupTimingMetric
    if "timeToInitialDisplayMs" in metrics_json:
        bench_data = _parse_startup_timing_metric(data)
    # Detect if benchmark is of type FrameTimingMetric
    elif "frameCount" in metrics_json:
        bench_data = _parse_frame_timing_metric(data)
    # Detect if benchmark is of type MemoryUsageMetric
    elif "memoryRssAnonMaxKb" in metrics_json:
        bench_data = _parse_memory_usage_metric(data)
    # Detect if benchmark is of type MemoryUsageMetric
    elif "memoryRssAnonLastKb" in metrics_json:
        bench_data = _parse_memory_usage_metric(data)
    else:
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
    report: BenchmarkReport = BenchmarkReport()
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
