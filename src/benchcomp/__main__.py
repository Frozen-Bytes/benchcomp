import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from statistics import median
from typing import Any

from scipy.stats import mannwhitneyu

DEFAULT_STEP_FIT_THRESHOLD: float = 25.0
DEFAULT_P_VALUE_THRESHOLD: float = 0.01
DEFAULT_FRAME_TIME_TARGET_MS: float = 1000 / 60

logger = logging.getLogger(__name__)

g_step_fit_threshold: float = DEFAULT_STEP_FIT_THRESHOLD
g_p_value_threshold: float = DEFAULT_P_VALUE_THRESHOLD
g_frame_time_target_ms: float = DEFAULT_FRAME_TIME_TARGET_MS

@dataclass
class BenchmarkCompareResult:
    minimum: tuple[float, float]
    maximum: tuple[float, float]
    median: tuple[float, float]
    metric: str
    iterations: int
    result: Any


@dataclass
class Device:
    brand: str = ""
    name: str = ""
    model: str = ""
    cpu_cores: int = 0
    cpu_freq: int = 0
    mem_size: int = 0
    emulated: bool = True


@dataclass
class Metric:
    minimum: float
    maximum: float
    median: float
    coefficient_of_variation: float
    runs: list[float]


@dataclass
class SampledMetric:
    p50: float
    p90: float
    p95: float
    p99: float
    runs: list[list[float]]


@dataclass
class StartupTimingMetric:
    # timeToInitialDisplayMs - Time from the system receiving a launch intent to rendering the first frame of the destination Activity.
    time_to_initial_display_ms: Metric

    # timeToFullDisplayMs - Time from the system receiving a launch intent until
    # the application reports fully drawn via android.app.Activity.reportFullyDrawn.
    #
    # The measurement stops at the completion of rendering the first frame after (or containing) the reportFullyDrawn() call.
    # This measurement may not be available prior to API 29.
    time_to_full_display_ms: Metric | None


@dataclass
class FrameTimingMetric:
    # frameCount - How many total frames were produced. This is a secondary metric which
    # can be used to understand why the above metrics changed. For example,
    # when removing unneeded frames that were incorrectly invalidated to save power,
    # frameOverrunMs and frameDurationCpuMs will often get worse, as the removed frames were trivial.
    # Checking frameCount can be a useful indicator in such cases.
    frame_count: Metric

    # frameDurationCpuMs - How much time the frame took to be produced on the CPU - on both the UI Thread, and RenderThread.
    # Note that this doesn't account for time before the frame started (before Choreographer#doFrame), as that data isn't available in traces prior to API 31.
    frame_duration_ms: SampledMetric

    # frameOverrunMs (Requires API 31) - How much time a given frame missed its deadline by.
    # Positive numbers indicate a dropped frame and visible jank / stutter,
    # negative numbers indicate how much faster than the deadline a frame was.
    frame_overrun_ms: SampledMetric | None

class MemoryMetricMode(Enum):
    UNKNOWN = 0
    LAST = 1
    MAX = 2

@dataclass
class MemoryUsageMetric:
    # There are two modes for measurement - Last, which represents the last observed
    # value during an iteration, and Max, which represents the largest sample observed per measurement.
    mode: MemoryMetricMode

    # memoryRssAnonKb - Anonymous resident/allocated memory owned by the process,
    # not including memory mapped files or shared memory.
    memory_rss_anon_kb: Metric

    # memoryRssFileKb - Memory allocated by the process to map files.
    memory_rss_file_kb: Metric

    # memoryHeapSizeKb - Heap memory allocations from the Android Runtime, sampled after each GC.
    memory_heap_size_kb: Metric

    # memoryGpuKb - GPU Memory allocated for the process.
    memory_gpu_kb: Metric | None

@dataclass
class Benchmark:
    name: str
    class_name: str
    total_run_time_ns: int
    warmup_iterations: int
    repeat_iterations: int
    data: StartupTimingMetric | FrameTimingMetric | MemoryUsageMetric | None


@dataclass
class BenchmarkReport:
    device: Device = field(default_factory=Device)
    benchmarks: dict[str, Benchmark] = field(default_factory=dict)


def parse_macrobechmark_report(path: Path | str) -> BenchmarkReport | None:
    def parse_device(data: dict[str, Any]) -> Device:
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

    def parse_metric(data: dict[str, Any]) -> Metric:
        return Metric(
            minimum=data.get("minimum", 0.0),
            maximum=data.get("maximum", 0.0),
            median=data.get("median", 0.0),
            coefficient_of_variation=data.get("coefficientOfVariation", 0.0),
            runs=data.get("runs", []),
        )

    def parse_sampled_metric(data: dict[str, Any]) -> SampledMetric:
        return SampledMetric(
            p50=data.get("P50", 0.0),
            p90=data.get("P90", 0.0),
            p95=data.get("P95", 0.0),
            p99=data.get("P99", 0.0),
            runs=data.get("runs", []),
        )

    def parse_startup_timing_metric(data: dict[str, Any]) -> StartupTimingMetric | None:
        metrics = data.get("metrics", {})
        time_to_full_display_ms: Metric | None = None

        if "timeToFullDisplayMs" in metrics:
            time_to_full_display_ms = parse_metric(metrics.get("timeToFullDisplayMs"))

        return StartupTimingMetric(
            time_to_initial_display_ms=parse_metric(metrics["timeToInitialDisplayMs"]),
            time_to_full_display_ms=time_to_full_display_ms,
        )

    def parse_frame_timing_metric(data: dict[str, Any]) -> FrameTimingMetric | None:
        metrics = data.get("metrics", {})
        sampled_metrics = data.get("sampledMetrics", {})
        frame_overrun_ms: SampledMetric | None = None

        if "frameOverrunMs" in sampled_metrics:
            frame_overrun_ms = parse_sampled_metric(sampled_metrics.get("frameOverrunMs"))

        return FrameTimingMetric(
            frame_count=parse_metric(metrics.get("frameCount", {})),
            frame_duration_ms=parse_sampled_metric(sampled_metrics.get("frameDurationCpuMs", {})),
            frame_overrun_ms=frame_overrun_ms,
        )

    def parse_memory_usage_metric(data: dict[str, Any]) -> MemoryUsageMetric | None:
        metrics = data.get("metrics", {})

        memory_mode: MemoryMetricMode = MemoryMetricMode.UNKNOWN
        memory_rss_anon_kb: Metric | None = None
        memory_rss_file_kb: Metric | None = None
        memory_heap_size_kb: Metric | None = None
        memory_gpu_kb: Metric | None = None

        if "memoryRssAnonMaxKb" in metrics:
            memory_mode = MemoryMetricMode.MAX
            memory_rss_anon_kb = parse_metric(metrics.get("memoryRssAnonMaxKb"))
        elif "memoryRssAnonLastKb":
            memory_mode = MemoryMetricMode.LAST
            memory_rss_anon_kb = parse_metric(metrics.get("memoryRssAnonLastKb"))

        if "memoryRssFileMaxKb" in metrics:
            memory_rss_file_kb = parse_metric(metrics.get("memoryRssFileMaxKb"))
        elif "memoryRssFileLastKb":
            memory_rss_file_kb = parse_metric(metrics.get("memoryRssFileLastKb"))

        if "memoryHeapSizeMaxKb" in metrics:
            memory_heap_size_kb = parse_metric(metrics.get("memoryHeapSizeMaxKb"))
        elif "memoryHeapSizeLastKb":
            memory_heap_size_kb = parse_metric(metrics.get("memoryHeapSizeLastKb"))

        if "memoryGpuMaxKb" in metrics:
            memory_gpu_kb = parse_metric(metrics.get("memoryGpuMaxKb"))
        elif "memoryGpuLastKb" in metrics:
            memory_gpu_kb = parse_metric(metrics.get("memoryGpuLastKb"))

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

    def parse_benchmark(data: dict[str, Any]) -> Benchmark | None:
        name: str = data.get("name", "")
        metrics_json = data.get("metrics", {})
        bench_data: StartupTimingMetric | FrameTimingMetric | MemoryUsageMetric | None = None

        # Detect if benchmark is of type StartupTimingMetric
        if "timeToInitialDisplayMs" in metrics_json:
            bench_data = parse_startup_timing_metric(data)
        # Detect if benchmark is of type FrameTimingMetric
        elif "frameCount" in metrics_json:
            bench_data = parse_frame_timing_metric(data)
        # Detect if benchmark is of type MemoryUsageMetric
        elif "memoryRssAnonMaxKb" in metrics_json:
            bench_data = parse_memory_usage_metric(data)
        # Detect if benchmark is of type MemoryUsageMetric
        elif "memoryRssAnonLastKb" in metrics_json:
            bench_data = parse_memory_usage_metric(data)
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

    report: BenchmarkReport = BenchmarkReport()
    with open(path, "r") as file:
        try:
            root = json.load(file)
            report.device = parse_device(root.get("context", {}))
            benchmarks = root.get("benchmarks", [])
            for bench_obj in benchmarks:
                try:
                    benchmark = parse_benchmark(bench_obj)
                    if benchmark is not None:
                        report.benchmarks[benchmark.name] = benchmark
                except Exception as e:
                    name = bench_obj.get("name", "")
                    logger.warning(f"failed to parse benchmark '{name}', skipping. ({e})")

        except json.JSONDecodeError | UnicodeDecodeError:
            logger.error(f"failed to parse json file '{path}', invalid JSON document")
            return None

    return report


def step_fit(a, b) -> float:
    def sum_squared_error(values):
        avg = sum(values) / len(values)
        return sum((v - avg) ** 2 for v in values)

    if not a or not b:
        return 0.0

    total_squared_error = sum_squared_error(a) + sum_squared_error(b)
    step_error = math.sqrt(total_squared_error) / (len(a) + len(b))
    if step_error == 0.0:
        return 0.0

    return (sum(a) / len(a) - sum(b) / len(b)) / step_error


def calculate_total_freeze_time_ms(frame_times: list[float], frame_time_target: float) -> float:
    total_freeze_ms: float = 0.0
    for ft in frame_times:
        if ft > frame_time_target:
            total_freeze_ms += ft - frame_time_target
    return total_freeze_ms


def compare_benchmark(a: Benchmark, b: Benchmark, compare_func, *args, **kwargs) -> BenchmarkCompareResult | None:
    compare_result = None

    a_values: list[float] = []
    b_values: list[float] = []
    v_minimum: tuple[float, float] = (0.0, 0.0)
    v_maximum: tuple[float, float] = (0.0, 0.0)
    v_median:  tuple[float, float] = (0.0, 0.0)
    metric: str = ""
    if isinstance(a.data, StartupTimingMetric) and isinstance(b.data, StartupTimingMetric):
        metric = "TID"

        a_values = a.data.time_to_initial_display_ms.runs
        b_values = b.data.time_to_initial_display_ms.runs

        v_minimum = (
            a.data.time_to_initial_display_ms.minimum,
            b.data.time_to_initial_display_ms.minimum,
        )
        v_maximum = (
            a.data.time_to_initial_display_ms.maximum,
            b.data.time_to_initial_display_ms.maximum,
        )
        v_median = (
            a.data.time_to_initial_display_ms.median,
            b.data.time_to_initial_display_ms.median,
        )
    elif isinstance(a.data, FrameTimingMetric) and isinstance(b.data, FrameTimingMetric):
        metric = "FFT"

        a_values = [ calculate_total_freeze_time_ms(run, g_frame_time_target_ms) for run in a.data.frame_duration_ms.runs ]
        b_values = [ calculate_total_freeze_time_ms(run, g_frame_time_target_ms) for run in b.data.frame_duration_ms.runs ]

        v_minimum = (min(a_values), min(b_values))
        v_maximum = (max(a_values), max(b_values))
        v_median  = (median(a_values), median(b_values))
    elif isinstance(a.data, MemoryUsageMetric) and isinstance(b.data, MemoryUsageMetric):
        metric = "MEMU"

        a_values = a.data.memory_rss_anon_kb.runs + a.data.memory_rss_file_kb.runs
        b_values = b.data.memory_rss_anon_kb.runs + b.data.memory_rss_file_kb.runs

        v_minimum = (min(a_values), min(b_values))
        v_maximum = (max(a_values), max(b_values))
        v_median  = (median(a_values), median(b_values))
    else:
        logger.warning(f"benchmark '{a.name}' type mismatch or unknown, skipping. (a_type: {type(a.data)}, b_type: {type(b.data)})")
        return None

    try:
        compare_result = compare_func(a_values, b_values, *args, **kwargs)
    except Exception as e:
        logger.warning(f"failed to compare benchmark '{a.name}' metric '{metric}', skipping. ({e})'")
        return None

    return BenchmarkCompareResult(
        minimum=v_minimum,
        maximum=v_maximum,
        median=v_median,
        metric=metric,
        iterations=a.repeat_iterations,
        result=compare_result,
    )


def print_device_specifications(device: Device) -> None:
    print(f"Device ({device.name}):")
    print(f"  Brand    : {device.brand}")
    print(f"  Model    : {device.model}")
    print(f"  Cores    : {device.cpu_cores}")
    print(f"  Core Freq: {device.cpu_freq}Hz")
    print(f"  Memory   : {device.mem_size}MB")
    print(f"  Emulator : {device.emulated}")


def print_step_fit_statistics(statistics: dict[str, BenchmarkCompareResult]) -> None:
    NAME_WIDTH: int = 40
    ITER_WIDTH: int = 3
    VERDICT_WIDTH: int = 7
    METRIC_WIDTH: int = 6
    STAT_WIDTH: int = 28
    NUM_WIDTH: int = 27
    METRIC_UNIT: dict[str, str] = {"TID": "ms", "FFT": "ms", "MEMU": "Kb"}

    header = (
        f"{'Benchmark:Iterations':<{NAME_WIDTH}} | "
        f"{'Metric':<{METRIC_WIDTH}} | "
        f"{'Verdict':<{VERDICT_WIDTH}} | "
        f"{f'Statistic (Threshold: {g_step_fit_threshold})':<{STAT_WIDTH}} | "
        f"{'Minimum':<{NUM_WIDTH}} | "
        f"{'Maximum':<{NUM_WIDTH}} | "
        f"{'Median':<{NUM_WIDTH}}"
    )

    print("Compare Function: Step fit")
    print(header)
    print("-" * len(header))

    regression_count: int = 0
    for name, result in statistics.items():
        verdict: str = ""
        if abs(result.result) <= g_step_fit_threshold:
            verdict = "Noise"
        elif result.result < 0:
            verdict = "Regress"
            regression_count += 1
        else:
            verdict = "Improve"

        display_name = name[:NAME_WIDTH-ITER_WIDTH] + f":{result.iterations}"
        row = (
            f"{display_name:<{NAME_WIDTH}} | "
            f"{result.metric:<{METRIC_WIDTH}} | "
            f"{verdict:<{VERDICT_WIDTH}} | "
            f"{f'fit: {result.result:.3f}':<{STAT_WIDTH}} | "
            f"{f'{result.minimum[0]:.3f}{METRIC_UNIT[result.metric]} ~ {result.minimum[1]:.3f}{METRIC_UNIT[result.metric]}':<{NUM_WIDTH}} | "
            f"{f'{result.maximum[0]:.3f}{METRIC_UNIT[result.metric]} ~ {result.maximum[1]:.3f}{METRIC_UNIT[result.metric]}':<{NUM_WIDTH}} | "
            f"{f'{result.median[0]:.3f}{METRIC_UNIT[result.metric]} ~ {result.median[1]:.3f}{METRIC_UNIT[result.metric]}':<{NUM_WIDTH}}"
        )
        print(row)

    print("-" * len(header))
    print(f"Regressions: {regression_count}")


def print_mannwhitneyu_statistics(statistics: dict[str, BenchmarkCompareResult]) -> None:
    NAME_WIDTH: int = 40
    ITER_WIDTH: int = 3
    VERDICT_WIDTH: int = 7
    METRIC_WIDTH: int = 6
    STAT_WIDTH: int = 35
    NUM_WIDTH: int = 27
    METRIC_UNIT: dict[str, str] = {"TID": "ms", "FFT": "ms", "MEMU": "Kb"}

    header = (
        f"{'Benchmark:Iterations':<{NAME_WIDTH}} | "
        f"{'Metric':<{METRIC_WIDTH}} | "
        f"{'Verdict':<{VERDICT_WIDTH}} | "
        f"{f'Statistic (pvalue Threshold: {g_p_value_threshold:.3f})':<{STAT_WIDTH}} | "
        f"{'Minimum':<{NUM_WIDTH}} | "
        f"{'Maximum':<{NUM_WIDTH}} | "
        f"{'Median':<{NUM_WIDTH}}"
    )

    print("Compare Function: Mann-Whitney's U-test")
    print(header)
    print("-" * len(header))

    regression_count: int = 0
    for name, result in statistics.items():
        r = result.result

        verdict: str = ""
        if r.pvalue < g_p_value_threshold:
            verdict = "Regress"
            regression_count += 1
        else:
            verdict = "Noise"

        display_name = name[:NAME_WIDTH-ITER_WIDTH] + f":{result.iterations}"
        row = (
            f"{display_name:<{NAME_WIDTH}} | "
            f"{result.metric:<{METRIC_WIDTH}} | "
            f"{verdict:<{VERDICT_WIDTH}} | "
            f"{f'pvalue: {r.pvalue:.3f}, stat: {r.statistic:.3f}':<{STAT_WIDTH}} | "
            f"{f'{result.minimum[0]:.3f}{METRIC_UNIT[result.metric]} ~ {result.minimum[1]:.3f}{METRIC_UNIT[result.metric]}':<{NUM_WIDTH}} | "
            f"{f'{result.maximum[0]:.3f}{METRIC_UNIT[result.metric]} ~ {result.maximum[1]:.3f}{METRIC_UNIT[result.metric]}':<{NUM_WIDTH}} | "
            f"{f'{result.median[0]:.3f}{METRIC_UNIT[result.metric]} ~ {result.median[1]:.3f}{METRIC_UNIT[result.metric]}':<{NUM_WIDTH}}"
        )
        print(row)

    print("-" * len(header))
    print(f"Regressions: {regression_count}")


def parse_commandline_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="benchcomp",
        description="Compare between macrobenchmark reports"
    )

    # Positional Arguments
    parser.add_argument(
        "baseline_dir",
        type=str,
        help="Path to the baseline macrobenchmark report directory",
    )
    parser.add_argument(
        "candidate_dir",
        type=str,
        help="Path to the candidate macrobenchmark report directory",
    )

    # Optional Arguments
    parser.add_argument(
        "--frametime",
        dest="frametime_target",
        type=float,
        default=DEFAULT_FRAME_TIME_TARGET_MS,
        metavar="MS",
        help=f"Target frame time in milliseconds (Default: {DEFAULT_FRAME_TIME_TARGET_MS:.3f}ms)",
    )
    parser.add_argument(
        "--fit",
        dest="step_fit_threshold",
        type=float,
        default=DEFAULT_STEP_FIT_THRESHOLD,
        metavar="VALUE",
        help=f"Threshold for step fit analysis (Default: {DEFAULT_STEP_FIT_THRESHOLD:.3f})",
    )
    parser.add_argument(
        "--alpha",
        dest="pvalue_threshold",
        type=float,
        default=DEFAULT_P_VALUE_THRESHOLD,
        metavar="VALUE",
        help=f"P-value threshold for Mann-Whitney U-test significance (Default: {DEFAULT_P_VALUE_THRESHOLD:.3f})",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_commandline_args()

    baseline_dir = Path(args.baseline_dir)
    candidate_dir = Path(args.candidate_dir)
    baseline_files = sorted(baseline_dir.glob("*.json"))
    candidate_files = sorted(candidate_dir.glob("*.json"))

    # set globals
    global g_step_fit_threshold
    global g_p_value_threshold
    global g_frame_time_target_ms
    g_step_fit_threshold = args.step_fit_threshold
    g_p_value_threshold = args.pvalue_threshold
    g_frame_time_target_ms =  args.frametime_target

    if len(baseline_files) <= 0:
        logger.critical('baseline has no macrobenchmark results')
        return 1

    if len(candidate_files) <= 0:
        logger.critical('candidate has no macrobenchmark results')
        return 1

    min_len = min(len(baseline_files), len(candidate_files))
    if len(baseline_files) != len(candidate_files):
        logger.warning(f"length mismatch, using first {min_len} samples. baseline: {len(baseline_files)}, candidate: {len(candidate_files)}")

    print('Macrobenchmark Result Mapping:')
    print('| Index | Baseline | Candidate |')
    print('--------------------------------')

    mismatch_count = 0
    for i in range(min_len):
        baseline_filename = baseline_files[i].name.upper()
        candidate_filename = candidate_files[i].name.upper()
        if baseline_filename != candidate_filename:
            mismatch_count += 1
            print('* ', end='')
        print(f'{i + 1} {baseline_files[i]} <-> {candidate_files[i]}')

    print('--------------------------------')
    print(f'# Match   : {min_len - mismatch_count}')
    print(f'# Mismatch: {mismatch_count}')
    if mismatch_count > 0:
        logger.warning("filename mapping mismatch detected. Output prediction may be incorrect")
    print()

    for i in range(min_len):
        print(f"Comparing Benchmark Run ({i + 1} / {min_len})")

        baseline_file:Path = baseline_files[i]
        candidate_file:Path = candidate_files[i]
        print(f"  > Baseline  benchmark report: {baseline_file.name}")
        print(f"  > Candidate benchmark report: {candidate_file.name}")
        print()

        baseline_report: BenchmarkReport | None = parse_macrobechmark_report(baseline_file)
        candidate_report: BenchmarkReport | None = parse_macrobechmark_report(candidate_file)
        if baseline_report is None or candidate_report is None:
            logger.error(f"invalid benchmark reports, skipping.\n  baseline: '{baseline_file}',\n  candidate: '{candidate_file}'")
            continue

        if baseline_report.device != candidate_report.device:
            logger.warning(f"benchmark device mismatch detected.\n  baseline: '{baseline_file}',\n  candidate: '{candidate_file}'")
            print("Baseline", end="")
            print_device_specifications(baseline_report.device)
            print("Candidate", end="")
            print_device_specifications(candidate_report.device)
        else:
            print_device_specifications(baseline_report.device)
        print()

        step_fit_statistics: dict[str, BenchmarkCompareResult] = {}
        mannwhitneyu_statistics: dict[str, BenchmarkCompareResult] = {}
        for name, candidate_benchmark in candidate_report.benchmarks.items():
            baseline_benchmark: Benchmark | None = baseline_report.benchmarks.get(name)
            if baseline_benchmark is None:
                logger.warning(f"baseline does not contain benchmark '{name}', skipping")
                continue

            # Step Fit
            result = compare_benchmark(baseline_benchmark, candidate_benchmark, step_fit)
            if result is not None:
                step_fit_statistics[name] = result
            else:
                logger.warning(f"couldn't compare 'Step Fit' benchmark '{name}', skipping")

            # Mann-Whitney's U-test
            result = compare_benchmark(
                baseline_benchmark,
                candidate_benchmark,
                mannwhitneyu,
                alternative="less",
                method="exact"
            )
            if result is not None:
                mannwhitneyu_statistics[name] = result
            else:
                logger.warning(f"couldn't compare 'Mann-Whitney's U-test' benchmark '{name}', skipping")

        print_step_fit_statistics(step_fit_statistics)
        print()
        print_mannwhitneyu_statistics(mannwhitneyu_statistics)

    return 0


if __name__ == "__main__":
    sys.exit(main())
