from statistics import mean, median, stdev

import pytest

from benchcomp.common import (
    Benchmark,
    FrameTimingMetric,
    MemorySamplingMode,
    MemoryUsageMetric,
    Metric,
    StartupTimingMetric,
)
from benchcomp.parser_macrobenchmark import (
    _parse_benchmark,
    _parse_device,
    _parse_metric,
)


def test_parse_device_basic():
    data = {
        "build": {
            "brand": "google",
            "device": "generic_x86_64_arm64",
            "fingerprint": "google/sdk_gphone_x86_64/generic_x86_64_arm64:11/RSR1.240422.006/12134477:userdebug/dev-keys",
            "id": "RSR1.240422.006",
            "model": "sdk_gphone_x86_64",
            "type": "userdebug",
            "version": {
                "codename": "REL",
                "sdk": 30
            }
        },
        "cpuCoreCount": 2,
        "cpuLocked": True,
        "cpuMaxFreqHz": 2000,
        "memTotalBytes": 4_120_588_288,
        "sustainedPerformanceModeEnabled": False,
        "artMainlineVersion": 1,
        "osCodenameAbbreviated": "R",
        "compilationMode": "verify",
        "payload": {}
    }

    device = _parse_device(data)

    assert device.brand == "google"
    assert device.name == "generic_x86_64_arm64"
    assert device.model == "sdk_gphone_x86_64"
    assert device.cpu_cores == 2
    assert device.mem_size_mb == 4_120_588_288 // (1024 * 1024)
    assert device.emulated is True


def test_metric():
    runs: list[float] = [1500, 2000, 2500]
    data = {
        "minimum": min(runs),
        "maximum": max(runs),
        "median": median(runs),
        "coefficientOfVariation": stdev(runs) / mean(runs),
        "runs": runs
    }

    metric: Metric = _parse_metric(data, name="Test", name_short="T", unit="t")

    # Metadata
    assert metric.metadata.name == "Test"
    assert metric.metadata.name_short == "T"
    assert metric.metadata.unit == "t"

    # Data
    assert metric.runs == runs

    # Statistical Functions
    assert metric.min() == min(runs)
    assert metric.median() == median(runs)
    assert metric.max() == max(runs)
    assert metric.mean() == mean(runs)
    assert metric.stdev() == pytest.approx(stdev(runs))
    assert metric.cv() == pytest.approx(stdev(runs) / mean(runs))


def test_parse_startup_benchmark():
    tfd_runs: list[float] = [1000, 2000, 3000]
    tid_runs: list[float] = [4000, 5000, 6000]
    data = {
        "name": "StartupBenchmark",
        "className": "com.test.app.startup.StartupBenchmark",
        "totalRunTimeNs": 1_000_000_000,
        "warmupIterations": 2,
        "repeatIterations": 3,
        "metrics": {
            "timeToFullDisplayMs": {
                "minimum": min(tfd_runs),
                "maximum": max(tfd_runs),
                "median": median(tfd_runs),
                "coefficientOfVariation": stdev(tfd_runs) / mean(tfd_runs),
                "runs": tfd_runs
            },
            "timeToInitialDisplayMs": {
                "minimum": min(tid_runs),
                "maximum": max(tid_runs),
                "median": median(tid_runs),
                "coefficientOfVariation": stdev(tid_runs) / mean(tid_runs),
                "runs": tid_runs
            }
        },
    }

    benchmark: Benchmark | None = _parse_benchmark(data)

    # Metadata
    assert benchmark is not None
    assert benchmark.name == "StartupBenchmark"
    assert benchmark.class_name == "com.test.app.startup.StartupBenchmark"
    assert benchmark.total_runtime_ns == 1_000_000_000
    assert benchmark.warmup_iterations == 2
    assert benchmark.repeat_iterations == 3

    # Data
    assert isinstance(benchmark.data, StartupTimingMetric)

    assert benchmark.data.time_to_full_display_ms
    assert benchmark.data.time_to_full_display_ms.runs == tfd_runs
    assert benchmark.data.time_to_full_display_ms.metadata.name == "Time to Full Display"
    assert benchmark.data.time_to_full_display_ms.metadata.name_short == "TFD"
    assert benchmark.data.time_to_full_display_ms.metadata.unit == "ms"

    assert benchmark.data.time_to_initial_display_ms.runs == tid_runs
    assert benchmark.data.time_to_initial_display_ms.metadata.name == "Time to Initial Display"
    assert benchmark.data.time_to_initial_display_ms.metadata.name_short == "TID"
    assert benchmark.data.time_to_initial_display_ms.metadata.unit == "ms"

    # Helpers
    assert benchmark.is_same_benchmark(benchmark)


def test_parse_frame_timing_benchmark():
    frame_count: list[float] = [3, 3, 3]
    frame_duration: list[float] = [1, 2, 3]
    frame_overrun: list[float] = [-1, 0, 1]

    data = {
        "name": "FrameTimingBenchmark",
        "className": "com.test.app.startup.FrameTimingBenchmark",
        "totalRunTimeNs": 1_000_000_000,
        "warmupIterations": 2,
        "repeatIterations": 3,
        "metrics": {
            "frameCount": {
                "runs": frame_count,
            },
        },
        "sampledMetrics": {
            "frameDurationCpuMs": {
                "P50": 1,
                "P90": 2,
                "P95": 3,
                "P99": 4,
                "runs": [
                    frame_duration,
                    frame_duration,
                    frame_duration,
                ],
            },
            "frameOverrunMs": {
                "P50": 5,
                "P90": 6,
                "P95": 7,
                "P99": 8,
                "runs": [
                    frame_overrun,
                    frame_overrun,
                    frame_overrun,
                ],
            },
        },
    }

    benchmark: Benchmark | None = _parse_benchmark(data)

    # Metadata
    assert benchmark is not None
    assert benchmark.name == "FrameTimingBenchmark"
    assert benchmark.class_name == "com.test.app.startup.FrameTimingBenchmark"
    assert benchmark.total_runtime_ns == 1_000_000_000
    assert benchmark.warmup_iterations == 2
    assert benchmark.repeat_iterations == 3

    # Data
    assert isinstance(benchmark.data, FrameTimingMetric)
    assert benchmark.data.frame_count.metadata.name == "Frame Count"
    assert benchmark.data.frame_count.metadata.name_short == "FC"
    assert benchmark.data.frame_count.metadata.unit == ""
    assert benchmark.data.frame_count.runs == frame_count

    assert benchmark.data.frame_duration_ms.runs == [frame_duration, frame_duration, frame_duration]
    assert benchmark.data.frame_duration_ms.p50 == 1
    assert benchmark.data.frame_duration_ms.p90 == 2
    assert benchmark.data.frame_duration_ms.p95 == 3
    assert benchmark.data.frame_duration_ms.p99 == 4

    assert benchmark.data.frame_overrun_ms
    assert benchmark.data.frame_overrun_ms.runs == [frame_overrun, frame_overrun, frame_overrun]
    assert benchmark.data.frame_overrun_ms.p50 == 5
    assert benchmark.data.frame_overrun_ms.p90 == 6
    assert benchmark.data.frame_overrun_ms.p95 == 7
    assert benchmark.data.frame_overrun_ms.p99 == 8

    # Helpers
    assert benchmark.is_same_benchmark(benchmark)


def test_parse_memory_benchmark_max_mode():
    mem_rss_anon: list[float] = [1000, 2000]
    mem_rss_file: list[float] = [3000, 4000]
    mem_heap: list[float] = [5000, 6000]
    mem_gpu: list[float] = [7000, 8000]
    data = {
        "name": "MemoryBenchmark",
        "className": "com.test.app.startup.MemoryBenchmark",
        "totalRunTimeNs": 1_000_000_000,
        "warmupIterations": 2,
        "repeatIterations": 3,
        "metrics": {
            "memoryRssAnonMaxKb": {"runs": mem_rss_anon},
            "memoryRssFileMaxKb": {"runs": mem_rss_file},
            "memoryHeapSizeMaxKb": {"runs": mem_heap},
            "memoryGpuMaxKb": {"runs": mem_gpu},
        },
    }

    benchmark: Benchmark | None = _parse_benchmark(data)

    # Metadata
    assert benchmark is not None
    assert benchmark.name == "MemoryBenchmark"
    assert benchmark.class_name == "com.test.app.startup.MemoryBenchmark"
    assert benchmark.total_runtime_ns == 1_000_000_000
    assert benchmark.warmup_iterations == 2
    assert benchmark.repeat_iterations == 3

    # Data
    assert isinstance(benchmark.data, MemoryUsageMetric)

    assert benchmark.data.sampling_mode == MemorySamplingMode.MAX

    assert benchmark.data.memory_rss_anon_kb.metadata.name == "Memory Resident Set Size Anonymous Max"
    assert benchmark.data.memory_rss_anon_kb.metadata.name_short == "MEM_RSS_ANON_MAX"
    assert benchmark.data.memory_rss_anon_kb.metadata.unit == "Kb"
    assert benchmark.data.memory_rss_anon_kb.runs == mem_rss_anon

    assert benchmark.data.memory_rss_file_kb.metadata.name == "Memory Resident Set Size File Max"
    assert benchmark.data.memory_rss_file_kb.metadata.name_short == "MEM_RSS_FILE_MAX"
    assert benchmark.data.memory_rss_file_kb.metadata.unit == "Kb"
    assert benchmark.data.memory_rss_file_kb.runs == mem_rss_file

    assert benchmark.data.memory_heap_size_kb.metadata.name == "Memory Heap Size Max"
    assert benchmark.data.memory_heap_size_kb.metadata.name_short == "MEM_HEAP_SIZE_MAX"
    assert benchmark.data.memory_heap_size_kb.metadata.unit == "Kb"
    assert benchmark.data.memory_heap_size_kb.runs == mem_heap

    assert benchmark.data.memory_gpu_kb
    assert benchmark.data.memory_gpu_kb.metadata.name == "Memory GPU Max"
    assert benchmark.data.memory_gpu_kb.metadata.name_short == "MEM_GPU_MAX"
    assert benchmark.data.memory_gpu_kb.metadata.unit == "Kb"
    assert benchmark.data.memory_gpu_kb.runs == mem_gpu

    # Helper
    assert benchmark.is_same_benchmark(benchmark)


def test_parse_memory_benchmark_last_mode():
    mem_rss_anon: list[float] = [1000, 2000]
    mem_rss_file: list[float] = [3000, 4000]
    mem_heap: list[float] = [5000, 6000]
    mem_gpu: list[float] = [7000, 8000]
    data = {
        "name": "MemoryBenchmark",
        "className": "com.test.app.startup.MemoryBenchmark",
        "totalRunTimeNs": 1_000_000_000,
        "warmupIterations": 2,
        "repeatIterations": 3,
        "metrics": {
            "memoryRssAnonLastKb": {"runs": mem_rss_anon},
            "memoryRssFileLastKb": {"runs": mem_rss_file},
            "memoryHeapSizeLastKb": {"runs": mem_heap},
            "memoryGpuLastKb": {"runs": mem_gpu},
        },
    }

    benchmark: Benchmark | None = _parse_benchmark(data)

    # Metadata
    assert benchmark is not None
    assert benchmark.name == "MemoryBenchmark"
    assert benchmark.class_name == "com.test.app.startup.MemoryBenchmark"
    assert benchmark.total_runtime_ns == 1_000_000_000
    assert benchmark.warmup_iterations == 2
    assert benchmark.repeat_iterations == 3

    # Data
    assert isinstance(benchmark.data, MemoryUsageMetric)

    assert benchmark.data.sampling_mode == MemorySamplingMode.LAST

    assert benchmark.data.memory_rss_anon_kb.metadata.name == "Memory Resident Set Size Anonymous Last"
    assert benchmark.data.memory_rss_anon_kb.metadata.name_short == "MEM_RSS_ANON_LAST"
    assert benchmark.data.memory_rss_anon_kb.metadata.unit == "Kb"
    assert benchmark.data.memory_rss_anon_kb.runs == mem_rss_anon

    assert benchmark.data.memory_rss_file_kb.metadata.name == "Memory Resident Set Size File Last"
    assert benchmark.data.memory_rss_file_kb.metadata.name_short == "MEM_RSS_FILE_LAST"
    assert benchmark.data.memory_rss_file_kb.metadata.unit == "Kb"
    assert benchmark.data.memory_rss_file_kb.runs == mem_rss_file

    assert benchmark.data.memory_heap_size_kb.metadata.name == "Memory Heap Size Last"
    assert benchmark.data.memory_heap_size_kb.metadata.name_short == "MEM_HEAP_SIZE_LAST"
    assert benchmark.data.memory_heap_size_kb.metadata.unit == "Kb"
    assert benchmark.data.memory_heap_size_kb.runs == mem_heap

    assert benchmark.data.memory_gpu_kb
    assert benchmark.data.memory_gpu_kb.metadata.name == "Memory GPU Last"
    assert benchmark.data.memory_gpu_kb.metadata.name_short == "MEM_GPU_LAST"
    assert benchmark.data.memory_gpu_kb.metadata.unit == "Kb"
    assert benchmark.data.memory_gpu_kb.runs == mem_gpu

    # Helper
    assert benchmark.is_same_benchmark(benchmark)
