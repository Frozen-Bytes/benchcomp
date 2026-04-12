from statistics import mean, median, stdev
from typing import cast

import pytest

from benchcomp.core import (
    Benchmark,
    FrameFreezeMetric,
    FrameTimingMetric,
    Measurement,
    MemorySamplingMode,
    MemoryUsageMetric,
    StartupTimingMetric,
)
from benchcomp.parser_macrobenchmark import (
    _parse_benchmark,
    _parse_device,
    _parse_measurement,
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
    assert device.cpu_freq == 2000
    assert device.cpu_locked
    assert device.mem_size_bytes == 4_120_588_288
    assert device.mem_size_mb == 4_120_588_288 // (1000 * 1000)
    assert device.emulated is True
    assert device.sdk == 30
    assert device.sdk_codename == "REL"


def test_metric():
    runs: list[float] = [1500, 2000, 2500]
    data = {
        "minimum": min(runs),
        "maximum": max(runs),
        "median": median(runs),
        "coefficientOfVariation": stdev(runs) / mean(runs),
        "runs": runs
    }

    measurement: Measurement = _parse_measurement(data, name="Test", name_short="T", unit="t")

    # Metadata
    assert measurement.metadata.name == "Test"
    assert measurement.metadata.name_short == "T"
    assert measurement.metadata.unit == "t"

    # Data
    assert measurement.runs == runs

    # Statistical Functions
    assert measurement.min == min(runs)
    assert measurement.median == median(runs)
    assert measurement.max == max(runs)
    assert measurement.mean == mean(runs)
    assert measurement.stdev == pytest.approx(stdev(runs))
    assert measurement.cv == pytest.approx(stdev(runs) / mean(runs))


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
    assert len(benchmark.metrics) == 1
    assert isinstance(benchmark.metrics[0], StartupTimingMetric)
    metric = cast(StartupTimingMetric, benchmark.metrics[0])

    assert metric.time_to_full_display_ms
    assert metric.time_to_full_display_ms.runs == tfd_runs
    assert metric.time_to_full_display_ms.metadata.name == "Time to Full Display"
    assert metric.time_to_full_display_ms.metadata.name_short == "TFD"
    assert metric.time_to_full_display_ms.metadata.unit == "ms"

    assert metric.time_to_initial_display_ms.runs == tid_runs
    assert metric.time_to_initial_display_ms.metadata.name == "Time to Initial Display"
    assert metric.time_to_initial_display_ms.metadata.name_short == "TID"
    assert metric.time_to_initial_display_ms.metadata.unit == "ms"

    measurements = metric.get_measurements()
    assert len(measurements) == 2
    assert measurements[0] == metric.time_to_initial_display_ms
    assert measurements[1] == metric.time_to_full_display_ms

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
    assert len(benchmark.metrics) == 1
    assert isinstance(benchmark.metrics[0], FrameTimingMetric)
    metric = cast(FrameTimingMetric, benchmark.metrics[0])

    assert metric.frame_count.metadata.name == "Frame Count"
    assert metric.frame_count.metadata.name_short == "FC"
    assert metric.frame_count.metadata.unit == ""
    assert metric.frame_count.runs == frame_count

    assert metric.frame_duration_ms.runs == [frame_duration, frame_duration, frame_duration]
    assert metric.frame_duration_ms.p50 == 1
    assert metric.frame_duration_ms.p90 == 2
    assert metric.frame_duration_ms.p95 == 3
    assert metric.frame_duration_ms.p99 == 4

    assert metric.frame_overrun_ms
    assert metric.frame_overrun_ms.runs == [frame_overrun, frame_overrun, frame_overrun]
    assert metric.frame_overrun_ms.p50 == 5
    assert metric.frame_overrun_ms.p90 == 6
    assert metric.frame_overrun_ms.p95 == 7
    assert metric.frame_overrun_ms.p99 == 8

    # Helpers
    assert benchmark.is_same_benchmark(benchmark)


def test_frame_freeze_metric():
    frame_count: list[float] = [3, 3, 3]
    frame_duration: list[float] = [15, 13, 17]
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
                ],
            },
            "frameOverrunMs": {
                "P50": 5,
                "P90": 6,
                "P95": 7,
                "P99": 8,
                "runs": [
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
    assert len(benchmark.metrics) == 1
    assert isinstance(benchmark.metrics[0], FrameTimingMetric)
    metric = cast(FrameTimingMetric, benchmark.metrics[0])
    freeze_metric = metric.to_freeze_metric()

    measurements = metric.get_measurements()
    assert len(measurements) == 3
    assert measurements[0] == metric.frame_count
    assert measurements[1] == freeze_metric.frame_freeze_duration_ms
    assert measurements[2] == freeze_metric.frame_freeze_overrun_ms

    assert isinstance(freeze_metric, FrameFreezeMetric)
    assert freeze_metric.frame_freeze_duration_ms.runs == [ pytest.approx(17 - (1000 / 60)) ]
    assert freeze_metric.frame_freeze_overrun_ms
    assert freeze_metric.frame_freeze_overrun_ms.runs == [ 1 ]


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
    assert len(benchmark.metrics) == 1
    assert isinstance(benchmark.metrics[0], MemoryUsageMetric)
    metric = cast(MemoryUsageMetric, benchmark.metrics[0])

    assert metric.sampling_mode == MemorySamplingMode.MAX

    assert metric.memory_rss_anon_kb.metadata.name == "Memory Resident Set Size Anonymous Max"
    assert metric.memory_rss_anon_kb.metadata.name_short == "MEM_RSS_ANON_MAX"
    assert metric.memory_rss_anon_kb.metadata.unit == "Kb"
    assert metric.memory_rss_anon_kb.runs == mem_rss_anon

    assert metric.memory_rss_file_kb.metadata.name == "Memory Resident Set Size File Max"
    assert metric.memory_rss_file_kb.metadata.name_short == "MEM_RSS_FILE_MAX"
    assert metric.memory_rss_file_kb.metadata.unit == "Kb"
    assert metric.memory_rss_file_kb.runs == mem_rss_file

    assert metric.memory_heap_size_kb.metadata.name == "Memory Heap Size Max"
    assert metric.memory_heap_size_kb.metadata.name_short == "MEM_HEAP_SIZE_MAX"
    assert metric.memory_heap_size_kb.metadata.unit == "Kb"
    assert metric.memory_heap_size_kb.runs == mem_heap

    assert metric.memory_gpu_kb
    assert metric.memory_gpu_kb.metadata.name == "Memory GPU Max"
    assert metric.memory_gpu_kb.metadata.name_short == "MEM_GPU_MAX"
    assert metric.memory_gpu_kb.metadata.unit == "Kb"
    assert metric.memory_gpu_kb.runs == mem_gpu

    measurements = metric.get_measurements()
    assert len(measurements) == 5

    assert measurements[0] == metric.memory_rss_anon_kb
    assert measurements[1] == metric.memory_rss_file_kb
    assert measurements[2] == metric.memory_heap_size_kb
    assert measurements[3] == metric.memory_gpu_kb

    total_mem_measurement = measurements[4]
    assert total_mem_measurement.metadata.name == "Total RSS Memory Usage Max"
    assert total_mem_measurement.metadata.name_short == "MEM_RSS_MAX"
    assert total_mem_measurement.metadata.unit == "Kb"

    assert total_mem_measurement.runs == [
        anon + file
        for anon, file in zip(metric.memory_rss_anon_kb.runs, metric.memory_rss_file_kb.runs)
    ]

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
    assert len(benchmark.metrics) == 1
    assert isinstance(benchmark.metrics[0], MemoryUsageMetric)
    metric = cast(MemoryUsageMetric, benchmark.metrics[0])

    assert metric.sampling_mode == MemorySamplingMode.LAST

    assert metric.memory_rss_anon_kb.metadata.name == "Memory Resident Set Size Anonymous Last"
    assert metric.memory_rss_anon_kb.metadata.name_short == "MEM_RSS_ANON_LAST"
    assert metric.memory_rss_anon_kb.metadata.unit == "Kb"
    assert metric.memory_rss_anon_kb.runs == mem_rss_anon

    assert metric.memory_rss_file_kb.metadata.name == "Memory Resident Set Size File Last"
    assert metric.memory_rss_file_kb.metadata.name_short == "MEM_RSS_FILE_LAST"
    assert metric.memory_rss_file_kb.metadata.unit == "Kb"
    assert metric.memory_rss_file_kb.runs == mem_rss_file

    assert metric.memory_heap_size_kb.metadata.name == "Memory Heap Size Last"
    assert metric.memory_heap_size_kb.metadata.name_short == "MEM_HEAP_SIZE_LAST"
    assert metric.memory_heap_size_kb.metadata.unit == "Kb"
    assert metric.memory_heap_size_kb.runs == mem_heap

    assert metric.memory_gpu_kb
    assert metric.memory_gpu_kb.metadata.name == "Memory GPU Last"
    assert metric.memory_gpu_kb.metadata.name_short == "MEM_GPU_LAST"
    assert metric.memory_gpu_kb.metadata.unit == "Kb"
    assert metric.memory_gpu_kb.runs == mem_gpu

    measurements = metric.get_measurements()
    assert len(measurements) == 5

    assert measurements[0] == metric.memory_rss_anon_kb
    assert measurements[1] == metric.memory_rss_file_kb
    assert measurements[2] == metric.memory_heap_size_kb
    assert measurements[3] == metric.memory_gpu_kb

    total_mem_measurement = measurements[4]
    assert total_mem_measurement.metadata.name == "Total RSS Memory Usage Last"
    assert total_mem_measurement.metadata.name_short == "MEM_RSS_LAST"
    assert total_mem_measurement.metadata.unit == "Kb"

    assert total_mem_measurement.runs == [
        anon + file
        for anon, file in zip(metric.memory_rss_anon_kb.runs, metric.memory_rss_file_kb.runs)
    ]

    # Helper
    assert benchmark.is_same_benchmark(benchmark)
