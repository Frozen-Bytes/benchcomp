import json
from typing import Any

from benchcomp.core import (
    AnalysisReport,
    BenchmarkComparisonResult,
    Device,
    MeasurementComparisonResult,
)

FILE_FORMAT_VERSION = 1


def _json_write_device(obj: Device) -> dict[str, Any]:
    root: dict[str, Any] = {}

    root["brand"] = obj.brand
    root["model"] = obj.model
    root["device"] = obj.name
    root["alias"] = obj.alias
    root["cpuCoreCount"] = obj.cpu_cores
    root["cpuLocked"] = obj.cpu_locked
    root["cpuMaxFreqHz"] = obj.cpu_freq
    root["memTotalBytes"] = obj.mem_size_bytes
    root["emulated"] = obj.emulated
    root["version"] = {
        "sdk": obj.sdk,
        "codename": obj.sdk_codename,
    }

    return root


def _json_write_measure_comp_result(obj: MeasurementComparisonResult) -> dict[str, Any]:
    root: dict[str, Any] = {}

    root["label"] = obj.metadata.name
    root["unit"] = obj.metadata.unit

    root["compareResults"] = {
        r.metadata.id: {
            "label": r.metadata.name,
            "statistic": r.statistic,
            "verdict": r.verdict.name,
            "statistic_label": r.metadata.state_label,
        }
        for r in obj.result
    }

    root["minimum"] = [obj.a.min, obj.b.min]
    root["maximum"] = [obj.a.max, obj.b.max]
    root["median"] = [obj.a.median, obj.b.median]
    root["coefficientOfVariation"] = [obj.a.cv, obj.b.cv]
    root["runs"] = [obj.a.runs, obj.b.runs]

    return root


def _json_write_bench_comp_result(obj: BenchmarkComparisonResult) -> dict[str, Any]:
    root: dict[str, Any] = {}

    root["name"] = obj.benchmark_name
    root["class"] = obj.benchmark_class
    root["totalRunTimeNs"] = [obj.a.total_runtime_ns, obj.b.total_runtime_ns]
    root["warmupIterations"] = [obj.a.warmup_iterations, obj.b.warmup_iterations]
    root["repeatIterations"] = [obj.a.repeat_iterations, obj.b.repeat_iterations]
    root["thresholds"] = obj.thresholds

    root["metrics"] = {
        r.metadata.name_short: _json_write_measure_comp_result(r)
        for r in obj.results
    }

    return root


def json_write_analysis_report(obj: AnalysisReport) -> str:
    root: dict[str, Any] = {}

    root["formatVersion"] = FILE_FORMAT_VERSION

    root["devices"] = [
        _json_write_device(device)
        for device in obj.get_devices()
    ]

    root["benchmarks"] = [
        _json_write_bench_comp_result(comp_result)
        for comp_result in obj.comparisons
    ]

    return json.dumps(root, indent=2)
