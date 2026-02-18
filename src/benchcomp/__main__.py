import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchcomp.compare import BenchmarkCompareResult, Verdict, compare_benchmark
from benchcomp.parser_cli import parse_commandline_args
from benchcomp.parser_common import Benchmark, BenchmarkReport, Device
from benchcomp.parser_macrobenchmark import parse_macrobechmark_report

logger = logging.getLogger(__name__)

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




def main() -> int:
    args = parse_commandline_args()

    baseline_dir = Path(args.baseline_dir)
    candidate_dir = Path(args.candidate_dir)
    baseline_files = sorted(baseline_dir.glob("*.json"))
    candidate_files = sorted(candidate_dir.glob("*.json"))

    # set globals
    step_fit_threshold = args.step_fit_threshold
    alpha = args.pvalue_threshold
    frametim_target_ms =  args.frametime_target

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
                threshold=step_fit_threshold,
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
                threshold=alpha,
                frametime_target=frametim_target_ms
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
