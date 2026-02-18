import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tabulate import tabulate

from benchcomp.compare import BenchmarkCompareResult, Verdict, compare_benchmark
from benchcomp.parser_cli import parse_commandline_args
from benchcomp.parser_common import Benchmark, BenchmarkReport, Device
from benchcomp.parser_macrobenchmark import parse_macrobechmark_report

logger = logging.getLogger(__name__)

@dataclass
class CompareMethodStruct:
    method_id: str
    name: str
    header: str
    results: list[BenchmarkCompareResult]

def print_device_specifications(device: Device) -> None:
    print(f"Device ({device.name}):")
    print(f"  Brand    : {device.brand}")
    print(f"  Model    : {device.model}")
    print(f"  Cores    : {device.cpu_cores}")
    print(f"  Core Freq: {device.cpu_freq}Hz")
    print(f"  Memory   : {device.mem_size}MB")
    print(f"  Emulator : {device.emulated}")


# TODO: This is bad
def print_compare_method_results(cr_a: BenchmarkCompareResult, cr_b: BenchmarkCompareResult) -> None:
    def to_seconds(time_ns: float):
        return time_ns / (1000 * 1000 * 1000)

    benchmark_name = cr_a.a_bench_ref.name
    assert benchmark_name == cr_a.b_bench_ref.name

    benchmark_class = cr_a.a_bench_ref.class_name
    assert benchmark_class == cr_a.b_bench_ref.class_name

    metric_name_long = cr_a.a_metric.name
    assert metric_name_long == cr_a.b_metric.name

    metric_name_short = cr_a.a_metric.name_short
    assert metric_name_short == cr_a.b_metric.name_short

    baseline_runs = ", ".join(f"{x:.3f}" for x in cr_a.a_metric.runs)
    candidate_runs = ", ".join(f"{x:.3f}" for x in cr_a.b_metric.runs)

    total_run_time_sec: float = to_seconds(
        cr_a.a_bench_ref.total_run_time_ns + cr_a.a_bench_ref.total_run_time_ns
    )
    a_run_time_sec: float = to_seconds(cr_a.a_bench_ref.total_run_time_ns)
    b_run_time_sec: float = to_seconds(cr_a.b_bench_ref.total_run_time_ns)

    print(f"> Benchmark '{benchmark_name}':")
    print(f"    Class               : {benchmark_class}")
    print(f"    Total Run Time (s)  : {total_run_time_sec:.3f} (Baseline: {a_run_time_sec:.3f}, Candidate: {b_run_time_sec:.3f})")
    print(f"    Warmup   Iterations : (Baseline: {cr_a.a_bench_ref.warmup_iterations}, Candidate: {cr_a.b_bench_ref.warmup_iterations})")
    print(f"    Repeated Iterations : (Baseline: {cr_a.a_bench_ref.repeat_iterations}, Candidate: {cr_a.b_bench_ref.repeat_iterations})")
    print(f"    Metric              : {metric_name_long} ({metric_name_short})")
    print(f"    Baseline  Runs ({cr_a.a_metric.unit}) : [{baseline_runs}]")
    print(f"    Candidate Runs ({cr_a.a_metric.unit}) : [{candidate_runs}]")
    print(f"    Verdict             : ({cr_a.method}: {cr_a.verdict}, {cr_b.method}: {cr_b.verdict})")
    print(f"    Statistic           : ({cr_a.method}: {cr_a.result:.3f}, {cr_b.method}: {cr_b.result:.3f})")


def _tabulate(
    statistics_results: list[BenchmarkCompareResult],
    stat_label: str,
    title: str = "",
) -> str:
    def _format_cell(
        a: float,
        b: float,
        infix: str = "-",
        suffix: str = "",
        unit: str = "",
    ) -> str:
        if infix:
            infix = f" {infix} "
        if suffix:
            suffix = f" {suffix}"
        return f"{a:.3f}{unit}{infix}{unit}{b:.3f}{suffix}"

    def _format_verdict(verdict: Verdict) -> str:
        return {
            Verdict.NOT_SIGNIFICANT: "~",
            Verdict.IMPROVEMENT: ">",
            Verdict.REGRESSION: "<",
        }.get(verdict, "-")

    header = [
        "Benchmark:Iteration",
        "Metric",
        "Median",
        "Minimum",
        "Maximum",
        "Standard Deviation"
    ]

    tabular_data: list[list[Any]] = []
    for result in statistics_results:
        benchmark_name = result.a_bench_ref.name
        assert benchmark_name == result.b_bench_ref.name

        iteration = result.a_bench_ref.repeat_iterations
        assert iteration == result.b_bench_ref.repeat_iterations

        metric = result.a_metric.name_short
        assert metric == result.b_metric.name_short

        v_median = _format_cell(
            result.a_metric.median(),
            result.b_metric.median(),
            infix = _format_verdict(result.verdict),
            suffix = f"({stat_label}: {result.result:.3f})"
        )
        v_min   = _format_cell(result.a_metric.min(), result.b_metric.min())
        v_max   = _format_cell(result.a_metric.max(), result.b_metric.max())
        v_stdev = _format_cell(result.a_metric.stdev(), result.b_metric.stdev())

        tabular_data.append(
            [
                f"{benchmark_name}:{iteration}",
                metric,
                v_median,
                v_min,
                v_max,
                v_stdev,
            ]
        )

    table = tabulate(
        tabular_data = tabular_data,
        headers = header,
        tablefmt="rounded_outline"
    )

    out: list[str] = []
    if title:
        width = max(len(line) for line in table.split('\n'))
        out.append(title.center(width))
    out.append(table)

    regressions = [
        r.a_bench_ref.name
        for r in statistics_results
        if r.verdict == Verdict.REGRESSION
    ]
    out.append(f"Regressions ({len(regressions)}): {regressions}")

    return "\n".join(out)


def main() -> int:
    args = parse_commandline_args()

    baseline_dir = Path(args.baseline_dir)
    candidate_dir = Path(args.candidate_dir)
    baseline_files = sorted(baseline_dir.glob("*.json"))
    candidate_files = sorted(candidate_dir.glob("*.json"))

    # set globals
    step_fit_threshold = args.step_fit_threshold
    alpha = args.pvalue_threshold
    frametime_target_ms =  args.frametime_target
    is_verbose = args.verbose

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
                "threshold": step_fit_threshold,
                "results": [],
            },
            "mannwhitneyu": {
                "header": "Mann-Whitney U-Test",
                "state": "pval",
                "threshold": alpha,
                "results": [],
            },
        }
        for name, candidate_benchmark in candidate_report.benchmarks.items():
            baseline_benchmark: Benchmark | None = baseline_report.benchmarks.get(name)
            if baseline_benchmark is None:
                logger.warning(f"baseline does not contain benchmark '{name}', skipping")
                continue

            results: list[BenchmarkCompareResult] = []
            for method, method_info in statistics.items():
                result: BenchmarkCompareResult | None = compare_benchmark(
                    baseline_benchmark,
                    candidate_benchmark,
                    method=method,
                    threshold=method_info["threshold"],
                    frametime_target=frametime_target_ms
                )
                if result is not None:
                    results.append(result)
                else:
                    logger.warning(f"couldn't compare '{method}' benchmark '{name}', skipping")

            for result in results:
                statistics[result.method]["results"].append(result)

            # TODO: Verbose mode is hardcoded to support 2 methods (stepfit, and mannwhitneyu) only,
            # if we add more we need to handle this.
            if is_verbose and len(results) >= 2:
                assert len(results) == 2
                print_compare_method_results(results[0], results[1])
                print()

        for _, method_info in statistics.items():
            print(
                _tabulate(
                    statistics_results=method_info["results"],
                    stat_label=method_info["state"],
                    title=method_info["header"],
                )
            )
            print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
