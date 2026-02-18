from dataclasses import dataclass, field
from enum import Enum
from statistics import mean, median, stdev


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
    _min: float | None = field(init=False, default=None)
    _max: float | None = field(init=False, default=None)
    _median: float | None = field(init=False, default=None)
    _stdev: float | None = field(init=False, default=None)

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


@dataclass
class BenchmarkReport:
    device: Device = field(default_factory=Device)
    benchmarks: dict[str, Benchmark] = field(default_factory=dict)
