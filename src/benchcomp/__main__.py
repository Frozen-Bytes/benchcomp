import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tabulate import tabulate

from benchcomp.compare import BenchmarkCompareResult, Verdict, compare_benchmark
from benchcomp.parser_cli import parse_commandline_args
from benchcomp.parser_common import (
    Benchmark,
    BenchmarkReport,
    Device,
    MetricMetadata,
    calc_total_iterations,
    calc_total_run_time,
)
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


def print_compare_method_results(
    cr_a: BenchmarkCompareResult,
    cr_b: BenchmarkCompareResult,
    total_run_time_s: float | None
) -> None:
    ref_cr = cr_a

    benchmark_name: str = ref_cr.benchmark_name
    benchmark_class: str = ref_cr.benchmark_class
    metric_metadata: MetricMetadata = cr_a.metric_metadata
    method: str = ref_cr.method

    a_total_rum_time_s = calc_total_run_time(ref_cr.a_bench_ref)
    b_total_rum_time_s = calc_total_run_time(ref_cr.b_bench_ref)
    ab_run_time_s = a_total_rum_time_s + b_total_rum_time_s
    run_time_percent: str = ""
    if total_run_time_s:
        run_time_percent = f" ({(ab_run_time_s / total_run_time_s) * 100:.2f}%)"

    a_total_warm_iters = calc_total_iterations(ref_cr.a_bench_ref, "warm")
    b_total_warm_iters = calc_total_iterations(ref_cr.b_bench_ref, "warm")
    a_total_repeat_iters = calc_total_iterations(ref_cr.a_bench_ref, "repeat")
    b_total_repeat_iters = calc_total_iterations(ref_cr.b_bench_ref, "repeat")

    baseline_runs = ", ".join(f"{x:.3f}" for x in ref_cr.a_metric.runs)
    candidate_runs = ", ".join(f"{x:.3f}" for x in ref_cr.b_metric.runs)

    print(f"> Benchmark '{benchmark_name}':")
    print(f"    Class               : {benchmark_class}")
    print(f"    Run Time (s)        : {ab_run_time_s:.3f} (Baseline: {a_total_rum_time_s:.3f}, Candidate: {b_total_rum_time_s:.3f}){run_time_percent}")
    print(f"    Warmup   Iterations : (Baseline: {a_total_warm_iters}, Candidate: {b_total_warm_iters})")
    print(f"    Repeated Iterations : (Baseline: {a_total_repeat_iters}, Candidate: {b_total_repeat_iters})")
    print(f"    Metric              : {metric_metadata.name} ({metric_metadata.name_short})")
    print(f"    Baseline  Runs ({metric_metadata.unit}) : [{baseline_runs}]")
    print(f"    Candidate Runs ({metric_metadata.unit}) : [{candidate_runs}]")
    print(f"    Verdict             : ({method}: {cr_a.verdict}, {cr_b.method}: {cr_b.verdict})")
    print(f"    Statistic           : ({method}: {cr_a.result:.3f}, {cr_b.method}: {cr_b.result:.3f})")


def print_report_paths(label: str, reports: list[BenchmarkReport]):
    print(f"  > {label} benchmark reports:")
    for r in reports:
        print(f"    - {r.filepath}")


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
        iterations: str = ""
        a_total_repeat_iters = calc_total_iterations(result.a_bench_ref, "repeat")
        b_total_repeat_iters = calc_total_iterations(result.b_bench_ref, "repeat")
        if a_total_repeat_iters == b_total_repeat_iters:
            iterations = f"{a_total_repeat_iters}"
        else:
            iterations = f"{a_total_repeat_iters}, {b_total_repeat_iters}"

        metric_metadata: MetricMetadata = result.metric_metadata

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
                f"{result.benchmark_name}:{iterations}",
                metric_metadata.name_short,
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
        r.benchmark_name
        for r in statistics_results
        if r.verdict == Verdict.REGRESSION
    ]
    out.append(f"Regressions ({len(regressions)}): {regressions}")

    return "\n".join(out)


def get_files(baseline_path: Path, candidate_path: Path) -> tuple[list[Path], list[Path]]:
    """
    Resolves paths into lists of JSON benchmark files.

    Args:
        baseline_path: Path to baseline directory or file.
        candidate_path: Path to candidate directory or file.

    Returns:
        A tuple containing (baseline_files, candidate_files).
    """
    def _get_files_from_path(p: Path) -> list[Path]:
        if p.is_dir():
            return sorted(p.glob("*.json"))
        if p.is_file():
            return [p]
        return []

    baseline_files: list[Path] = _get_files_from_path(baseline_path)
    candidate_files: list[Path] = _get_files_from_path(candidate_path)
    return baseline_files, candidate_files


def read_reports(files: list[Path]) -> list[BenchmarkReport]:
    """Parses benchmark files into BenchmarkReport objects, skipping invalid ones."""
    reports: list[BenchmarkReport] = []
    for f in files:
        report: BenchmarkReport | None = parse_macrobechmark_report(f)
        if report:
            reports.append(report)
        else:
            logger.error(f"invalid benchmark report '{f}', skipping.\n")
    return reports


def collect_benchmarks(reports: list[BenchmarkReport]) -> dict[str, list[Benchmark]]:
    """Groups benchmark results by their names across multiple reports."""
    benchmarks: dict[str, list[Benchmark]] = defaultdict(list)
    for report in reports:
        for name, benchmark in report.benchmarks.items():
            benchmarks[name].append(benchmark)
    return benchmarks


def main() -> int:
    args = parse_commandline_args()

    step_fit_threshold = args.step_fit_threshold
    alpha = args.pvalue_threshold
    frametime_target_ms = args.frametime_target
    is_verbose = args.verbose
    aggregation_method = args.aggregate_method

    baseline_files, candidate_files = get_files(args.baseline, args.candidate)
    if len(baseline_files) <= 0:
        logger.critical('baseline has no Macrobenchmark results')
        return 1
    if len(candidate_files) <= 0:
        logger.critical('candidate has no Macrobenchmark results')
        return 1

    baseline_reports: list[BenchmarkReport] = read_reports(baseline_files)
    candidate_reports: list[BenchmarkReport] = read_reports(candidate_files)
    comparison_groups = []
    if args.aggregate_method == "none":
        min_len = min(len(baseline_reports), len(candidate_reports))
        if len(baseline_reports) != len(candidate_reports):
            logger.warning(f"length mismatch, using first {min_len} samples. baseline: {len(baseline_reports)}, candidate: {len(candidate_reports)}")

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
            b, c = baseline_reports[i], candidate_reports[i]
            title = f"Comparing Benchmark Run ({i + 1} / {min_len})"
            common_names = b.benchmarks.keys() & c.benchmarks.keys()
            # TODO: Check for missing common names
            pairs = {name: ([b.benchmarks[name]], [c.benchmarks[name]]) for name in common_names}
            comparison_groups.append((title, pairs, [b], [c]))
    else:
        title = f"Comparing Benchmark Run (Aggregation: {aggregation_method})"
        b_benchmarks = collect_benchmarks(baseline_reports)
        c_benchmarks = collect_benchmarks(candidate_reports)
        common_names = b_benchmarks.keys() & c_benchmarks.keys()
        # TODO: Check for missing common names
        pairs = {name: (b_benchmarks[name], c_benchmarks[name]) for name in common_names}
        comparison_groups.append((title, pairs, baseline_reports, candidate_reports))

    for title, benchmark_pairs, b_reps, c_reps in comparison_groups:
        print(title)
        print_report_paths("Baseline", b_reps)
        print_report_paths("Candidate", c_reps)
        print()

        # if b.device != c.device:
        #     logger.warning(f"benchmark device mismatch detected.\n  baseline: '{b.filepath}',\n  candidate: '{c.filepath}'")
        #     print("Baseline", end="")
        #     print_device_specifications(b.device)
        #     print("Candidate", end="")
        #     print_device_specifications(c.device)
        # else:
        #     print_device_specifications(b.device)
        # print()

        stats_config = {
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

        total_run_time = 0.0
        for name, (b_list, c_list) in benchmark_pairs.items():
            current_pair_results = []

            for method, info in stats_config.items():
                result = compare_benchmark(
                    b_list,
                    c_list,
                    method=method,
                    threshold=info["threshold"],
                    frametime_target=frametime_target_ms,
                    aggregate=aggregation_method,
                )

                if result:
                    current_pair_results.append(result)
                    info["results"].append(result)
                else:
                    logger.warning(f"Skipping '{method}' for '{name}'")

            total_run_time += calc_total_run_time(b_list + c_list)
            if is_verbose and len(current_pair_results) >= 2:
                assert len(current_pair_results) == 2
                print_compare_method_results(current_pair_results[0], current_pair_results[1], total_run_time)
                print()

        for _, info in stats_config.items():
            if info["results"]:
                print(_tabulate(info["results"], info["state"], info["header"]))
                print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
