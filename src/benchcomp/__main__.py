import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from statistics import mean, median, stdev
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
    # Easier printing
    name: str
    name_short: str
    unit: str

    _runs: list[float]

    # Cached values (not in __init__)
    _min   : float | None = field(init=False, default=None)
    _max   : float | None = field(init=False, default=None)
    _median: float | None = field(init=False, default=None)
    _stdev : float | None = field(init=False, default=None)

    @property
    def runs(self) -> list[float]:
        return self._runs

    @runs.setter
    def runs(self, value: list[float]):
        self._runs = value
        self._min = None
        self._max = None
        self._median = None
        self._stdev = None

    def min(self) -> float:
        if len(self._runs) < 1:
            return float("nan")
        if self._min is None:
            self._min = min(self._runs)
        return self._min

    def max(self) -> float:
        if len(self._runs) < 1:
            return float("nan")
        if self._max is None:
            self._max = max(self._runs)
        return self._max

    def median(self) -> float:
        if len(self._runs) < 1:
            return float("nan")
        if self._median is None:
            self._median = median(self._runs)
        return self._median

    def mean(self) -> float:
        if len(self._runs) < 1:
            return float("nan")
        if self._mean is None:
            self._mean = mean(self._runs)
        return self._mean

    def stdev(self) -> float:
        if len(self._runs) < 2:
            return float("nan")
        if self._stdev is None:
            self._stdev = stdev(self._runs)
        return self._stdev

    def cv(self) -> float:
        if len(self._runs) < 2:
            return float("nan")
        if self.median() == 0:
            return float("nan")
        return self.stdev() / self.median()


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

    def calc_freeze_frame_duration_ms(self, target: float) -> list[float]:
        result: list[float] = []
        for run in self.frame_duration_ms.runs:
            freeze_ms: float = 0.0
            for ft in run:
                if ft > target:
                    freeze_ms += ft - target
            result.append(freeze_ms)
        return result

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

    def parse_metric(data: dict[str, Any], name: str, name_short: str, unit: str) -> Metric:
        return Metric(
            _runs=data.get("runs", []),
            name=name,
            name_short=name_short,
            unit=unit,
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
            time_to_full_display_ms = parse_metric(
                metrics.get("timeToFullDisplayMs"),
                name="Time to Full Display",
                name_short="TFD",
                unit="ms"
            )

        return StartupTimingMetric(
            time_to_initial_display_ms=parse_metric(
                metrics["timeToInitialDisplayMs"],
                name="Time to Initial Display",
                name_short="TID",
                unit="ms"
            ),
            time_to_full_display_ms=time_to_full_display_ms,
        )

    def parse_frame_timing_metric(data: dict[str, Any]) -> FrameTimingMetric | None:
        metrics = data.get("metrics", {})
        sampled_metrics = data.get("sampledMetrics", {})
        frame_overrun_ms: SampledMetric | None = None

        if "frameOverrunMs" in sampled_metrics:
            frame_overrun_ms = parse_sampled_metric(sampled_metrics.get("frameOverrunMs"))

        return FrameTimingMetric(
            frame_count=parse_metric(
                metrics.get("frameCount", {}),
                name="Frame Count",
                name_short="FC",
                unit=""
            ),
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
            memory_rss_anon_kb = parse_metric(
                metrics.get("memoryRssAnonMaxKb"),
                name="Memory Resident Set Size Anonymous Max",
                name_short="MEM_RSS_ANON_MAX",
                unit="Kb"
            )
        elif "memoryRssAnonLastKb":
            memory_mode = MemoryMetricMode.LAST
            memory_rss_anon_kb = parse_metric(
                metrics.get("memoryRssAnonLastKb"),
                name="Memory Resident Set Size Anonymous Last",
                name_short="MEM_RSS_ANON_LAST",
                unit="Kb"
            )

        if "memoryRssFileMaxKb" in metrics:
            memory_rss_file_kb = parse_metric(
                metrics.get("memoryRssFileMaxKb"),
                name="Memory Resident Set Size File Max",
                name_short="MEM_RSS_FILE_MAX",
                unit="Kb"
            )
        elif "memoryRssFileLastKb":
            memory_rss_file_kb = parse_metric(
                metrics.get("memoryRssFileLastKb"),
                name="Memory Resident Set Size File Last",
                name_short="MEM_RSS_FILE_Last",
                unit="Kb"
            )

        if "memoryHeapSizeMaxKb" in metrics:
            memory_heap_size_kb = parse_metric(
                metrics.get("memoryHeapSizeMaxKb"),
                name="Memory Heap Size Max",
                name_short="MEM_HEAP_SIZE_MAX",
                unit="Kb"
            )
        elif "memoryHeapSizeLastKb":
            memory_heap_size_kb = parse_metric(
                metrics.get("memoryHeapSizeLastKb"),
                name="Memory Heap Size LAST",
                name_short="MEM_HEAP_SIZE_LAST",
                unit="Kb"
            )

        if "memoryGpuMaxKb" in metrics:
            memory_gpu_kb = parse_metric(
                metrics.get("memoryGpuMaxKb"),
                name="Memory GPU Max",
                name_short="MEM_GPU_MAX",
                unit="Kb"
            )
        elif "memoryGpuLastKb" in metrics:
            memory_gpu_kb = parse_metric(
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


def compare_benchmark(
    a: Benchmark,
    b: Benchmark,
    method: str,
    threshold: float,
    frametime_target: float = DEFAULT_FRAME_TIME_TARGET_MS,
    *args,
    **kwargs,
) -> BenchmarkCompareResult | None:
    def step_fit(a: list[float], b: list[float]) -> float:
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

    assert a.name == b.name

    if a.repeat_iterations != b.repeat_iterations:
        logger.warning(f"benchmark '{a.name}' iteration mismatch, (a: {a.repeat_iterations}, b: {b.repeat_iterations})")

    a_metric: Metric
    b_metric: Metric
    if isinstance(a.data, StartupTimingMetric) and isinstance(b.data, StartupTimingMetric):
        a_metric = a.data.time_to_initial_display_ms
        b_metric = b.data.time_to_initial_display_ms
    elif isinstance(a.data, FrameTimingMetric) and isinstance(b.data, FrameTimingMetric):
        metric_name = "Freeze Frame Duration"
        metric_name_short = "FFD"
        metric_unit = "ms"

        a_metric = Metric(
            _runs=a.data.calc_freeze_frame_duration_ms(frametime_target),
            name=metric_name,
            name_short=metric_name_short,
            unit=metric_unit,
        )
        b_metric = Metric(
            _runs=b.data.calc_freeze_frame_duration_ms(frametime_target),
            name=metric_name,
            name_short=metric_name_short,
            unit=metric_unit,
        )
    elif isinstance(a.data, MemoryUsageMetric) and isinstance(b.data, MemoryUsageMetric):
        metric_name = "Total Memory Usage"
        metric_name_short = "MEMU"
        metric_unit = "Kb"

        a_metric = Metric(
            _runs=a.data.memory_rss_anon_kb.runs + a.data.memory_rss_file_kb.runs,
            name=metric_name,
            name_short=metric_name_short,
            unit=metric_unit,
        )
        b_metric = Metric(
            _runs=b.data.memory_rss_anon_kb.runs + b.data.memory_rss_file_kb.runs,
            name=metric_name,
            name_short=metric_name_short,
            unit=metric_unit,
        )
    else:
        logger.warning(f"benchmark '{a.name}' type mismatch or unknown, skipping. (a_type: {type(a.data)}, b_type: {type(b.data)})")
        return None

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
        logger.warning(f"failed to compare benchmark '{a.name}' metric '{a_metric.name}', skipping. ({e})'")
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


def print_device_specifications(device: Device) -> None:
    print(f"Device ({device.name}):")
    print(f"  Brand    : {device.brand}")
    print(f"  Model    : {device.model}")
    print(f"  Cores    : {device.cpu_cores}")
    print(f"  Core Freq: {device.cpu_freq}Hz")
    print(f"  Memory   : {device.mem_size}MB")
    print(f"  Emulator : {device.emulated}")


@dataclass
class TableFormatterConfig:
    field_width_name: int = 40
    field_width_iteration: int = 3
    field_width_metric: int = 6
    field_width_metric_value: int = 35
    field_width_number: int = 25


class TableFormatter():
    statistics: list[BenchmarkCompareResult]
    state_str: str
    title: str | None
    config: TableFormatterConfig

    _COL_LABEL_BENCHMARK : str = "Benchmark:Iterations"
    _COL_LABEL_METRIC    : str = "Metric"
    _COL_LABEL_MINIMUM   : str = "Minimum"
    _COL_LABEL_MEDIAN    : str = "Median"
    _COL_LABEL_MAXIMUM   : str = "Maximum"
    _COL_LABEL_STDEV     : str = "Standard Deviation"

    _col_width_iteration    : int
    _col_width_benchmark    : int
    _col_width_metric       : int
    _col_width_metric_value : int
    _col_width_number       : int

    def __init__(
        self,
        statistics: list[BenchmarkCompareResult],
        state_str: str,
        title: str | None = None,
        formatter_conf: TableFormatterConfig=TableFormatterConfig(),
    ) -> None:
        self.statistics = statistics
        self.state_str = state_str
        self.title = title
        self.config = formatter_conf

        self._compute_column_widths()

    def print(self) -> None:
        header = self._build_header()
        line = "-" * len(header)

        print(line)
        if self.title:
            print(self.title.center(len(header)))
            print(line)

        print(header)
        print(line)

        for stat in self.statistics:
            print(self._build_row(stat))

        print(line)
        regressions = [ r.a_bench_ref.name for r in self.statistics if r.verdict == Verdict.REGRESSION ]
        print(f"Regressions ({len(regressions)}): {regressions}")
        print()

    def _build_header(self):
        return " | ".join(
            [
                self._format_col(self._COL_LABEL_BENCHMARK, self._col_width_benchmark),
                self._format_col(self._COL_LABEL_METRIC, self._col_width_metric),
                self._format_col(self._COL_LABEL_MEDIAN, self._col_width_metric_value),
                self._format_col(self._COL_LABEL_MINIMUM, self._col_width_number),
                self._format_col(self._COL_LABEL_MAXIMUM, self._col_width_number),
                self._format_col(self._COL_LABEL_STDEV, self._col_width_number),
            ]
        )

    def _build_row(self, r: BenchmarkCompareResult) -> str:
        benchmark_col = self._build_benchmark_name(r)
        metric_col = self._format_col(r.a_metric.name_short, self._col_width_metric)
        median_col = self._build_range(
            r.a_metric.median(),
            r.b_metric.median(),
            infix=self._verdict_symbol(r.verdict),
            suffix=f"({self.state_str}={r.result:.3f})",
            unit=r.a_metric.unit,
            width=self._col_width_metric_value,
        )
        min_col = self._build_range(r.a_metric.min(), r.b_metric.min(), unit=r.a_metric.unit)
        max_col = self._build_range(r.a_metric.max(), r.b_metric.max(), unit=r.a_metric.unit)
        stdev_col = self._build_range(r.a_metric.stdev(), r.b_metric.stdev(), unit=r.a_metric.unit)
        return " | ".join([benchmark_col, metric_col, median_col, min_col, max_col, stdev_col])

    def _build_range(
        self,
        a: float,
        b: float,
        *,
        infix: str = "-",
        suffix: str = "",
        unit: str = "",
        width: int | None = None,
    ) -> str:
        width = width or self._col_width_number
        main = f"{a:.3f}{unit} {infix} {b:.3f}{unit}"

        if not suffix:
            return self._format_col(main, width)

        space_for_main = width - len(suffix) - 1
        return f"{main:<{space_for_main}} {suffix}"

    def _build_benchmark_name(self, r: BenchmarkCompareResult) -> str:
        name_width = self._col_width_benchmark - self._col_width_iteration
        name = r.a_bench_ref.name[:name_width]
        return self._format_col(
            f"{name}:{r.a_bench_ref.repeat_iterations}",
            self._col_width_benchmark,
        )

    def _format_col(self, value: str, width: int) -> str:
        return f"{value:<{width}}"

    def _verdict_symbol(self, verdict: Verdict) -> str:
        return {
            Verdict.NOT_SIGNIFICANT: "~",
            Verdict.IMPROVEMENT: "<",
            Verdict.REGRESSION: ">",
        }.get(verdict, "-")

    def _compute_column_widths(self) -> None:
        self._col_width_iteration = self.config.field_width_iteration
        self._col_width_benchmark = (max(len(self._COL_LABEL_BENCHMARK), self.config.field_width_name) + self._col_width_iteration)
        self._col_width_metric = max(len(self._COL_LABEL_METRIC), self.config.field_width_metric)
        self._col_width_metric_value = max(len(self._COL_LABEL_MEDIAN), self.config.field_width_metric_value )
        self._col_width_number = max(len(self._COL_LABEL_MINIMUM), self.config.field_width_number)


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

        statistics: dict[str, Any]= {
            "stepfit": {
                "header": "Step Fit",
                "state": "fit",
                "results": [],
            },
            "mannwhitneyu": {
                "header": "Mann-Whitney U test",
                "state": "pval",
                "results": [],
            },
        }
        for name, candidate_benchmark in candidate_report.benchmarks.items():
            baseline_benchmark: Benchmark | None = baseline_report.benchmarks.get(name)
            if baseline_benchmark is None:
                logger.warning(f"baseline does not contain benchmark '{name}', skipping")
                continue

            # Step Fit
            result = compare_benchmark(
                baseline_benchmark,
                candidate_benchmark,
                method="stepfit",
                threshold=g_step_fit_threshold,
            )
            if result is not None:
                statistics["stepfit"]["results"].append(result)
            else:
                logger.warning(f"couldn't compare 'Step Fit' benchmark '{name}', skipping")

            # Mann-Whitney's U-test
            result = compare_benchmark(
                baseline_benchmark,
                candidate_benchmark,
                method="mannwhitneyu",
                threshold=g_p_value_threshold
            )
            if result is not None:
                statistics["mannwhitneyu"]["results"].append(result)
            else:
                logger.warning(f"couldn't compare 'Mann-Whitney's U-test' benchmark '{name}', skipping")

        for _, v in statistics.items():
            TableFormatter(v["results"], state_str=v["state"], title=v["header"]).print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
