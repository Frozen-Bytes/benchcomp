import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from tabulate import tabulate

from benchcomp.common import (
    AnalysisReport,
    BenchmarkReport,
    Device,
    MetricMetadata,
    calc_total_iterations,
    calc_total_runtime,
    get_unique_devices,
)
from benchcomp.compare import COMPARE_METHODS, BenchmarkComparisonResult, Verdict

logger = logging.getLogger(__name__)


def print_file_pair_mapping(baseline: list[Path], candidate: list[Path]) -> None:
    def relative_diff_paths(a: Path, b: Path) -> tuple[str, str]:
        a_parts = a.resolve().parts
        b_parts = b.resolve().parts

        common_prefix_len = 0
        for x, y in zip(a_parts, b_parts):
            if x != y:
                break
            common_prefix_len += 1

        # If no common root, just return original
        if common_prefix_len == 0:
            return str(a), str(b)

        common_root = Path(*a_parts[:common_prefix_len])
        return (
            str(a.relative_to(common_root)),
            str(b.relative_to(common_root)),
        )

    title: str = "Macrobenchmark Report File Mapping"
    rows = []

    mismatch_count = 0
    for i, (b_filepath, c_filepath) in enumerate(zip(baseline, candidate)):
        index: str = f"{i}"
        if b_filepath.name != c_filepath.name:
            mismatch_count += 1
            index = "* " + index
        rows.append([index, *relative_diff_paths(b_filepath, c_filepath)])

    table = tabulate(
        tabular_data=rows,
        headers=[ "Index", "Baseline", "Candidate" ],
        tablefmt="rounded_outline",
    )

    width = max(len(line) for line in table.split('\n'))
    print(title.center(width))
    print(table)
    print(f'Matches   : {len(baseline) - mismatch_count}')
    print(f'Mismatches: {mismatch_count}')


def print_device_specifications(device: Device) -> None:
    print(f"  > Device ({device.name}):")
    print(f"      Brand    : {device.brand}")
    print(f"      Model    : {device.model}")
    print(f"      Cores    : {device.cpu_cores}")
    print(f"      Core Freq: {device.cpu_freq}Hz")
    print(f"      Memory   : {device.mem_size_mb}MB")
    print(f"      Emulator : {device.emulated}")


def print_comparison_details(
    comparisons: list[BenchmarkComparisonResult],
    total_runtime_s: float | None,
) -> None:
    ref_comparison = comparisons[0]

    benchmark_name: str = ref_comparison.benchmark_name
    benchmark_class: str = ref_comparison.benchmark_class
    metric_metadata: MetricMetadata = ref_comparison.metric_metadata

    a_total_runtime_s = calc_total_runtime(ref_comparison.a_bench_ref)
    b_total_runtime_s = calc_total_runtime(ref_comparison.b_bench_ref)
    ab_runtime_s = a_total_runtime_s + b_total_runtime_s
    runtime_percent: str = ""
    if total_runtime_s:
        runtime_percent = f" ({(ab_runtime_s / total_runtime_s) * 100:.2f}%)"

    a_total_warm_iters = calc_total_iterations(ref_comparison.a_bench_ref, "warm")
    b_total_warm_iters = calc_total_iterations(ref_comparison.b_bench_ref, "warm")

    a_total_repeat_iters = calc_total_iterations(ref_comparison.a_bench_ref, "repeat")
    b_total_repeat_iters = calc_total_iterations(ref_comparison.b_bench_ref, "repeat")

    baseline_runs = ", ".join(f"{x:.3f}" for x in ref_comparison.a_metric.runs)
    candidate_runs = ", ".join(f"{x:.3f}" for x in ref_comparison.b_metric.runs)

    verdicts: list[str] = []
    statistics: list[str] = []
    for r in comparisons:
        verdicts.append(f"{r.comparison_method}{r.verdict}")
        statistics.append(f"{r.comparison_method}: {r.comparison_result:.3f}")

    print(f"> Benchmark '{benchmark_name}':")
    print(f"    Class               : {benchmark_class}")
    print(f"    Run Time (s)        : {ab_runtime_s:.3f} (Baseline: {a_total_runtime_s:.3f}, Candidate: {b_total_runtime_s:.3f}){runtime_percent}")
    print(f"    Warmup   Iterations : (Baseline: {a_total_warm_iters}, Candidate: {b_total_warm_iters})")
    print(f"    Repeated Iterations : (Baseline: {a_total_repeat_iters}, Candidate: {b_total_repeat_iters})")
    print(f"    Metric              : {metric_metadata.name} ({metric_metadata.name_short})")
    print(f"    Baseline  Runs ({metric_metadata.unit}) : [{baseline_runs}]")
    print(f"    Candidate Runs ({metric_metadata.unit}) : [{candidate_runs}]")
    print(f"    Verdict             : ({', '.join(verdicts)})")
    print(f"    Statistic           : ({', '.join(statistics)})")


def print_report_paths(label: str, reports: list[BenchmarkReport]):
    print(f"  > {label} benchmark reports:")
    for r in reports:
        print(f"    - {r.filepath}")


def print_summary_table(
    comparisons: list[BenchmarkComparisonResult],
    stat_label: str,
    title: str = "",
) -> None:
    def format_value_cell(
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

    def format_verdict(verdict: Verdict) -> str:
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
        "Standard Deviation",
        "Coefficient of Variance",
    ]

    rows: list[list[Any]] = []
    for c in comparisons:
        iterations: str = ""
        a_total_repeat_iters = calc_total_iterations(c.a_bench_ref, "repeat")
        b_total_repeat_iters = calc_total_iterations(c.b_bench_ref, "repeat")
        if a_total_repeat_iters == b_total_repeat_iters:
            iterations = f"{a_total_repeat_iters}"
        else:
            iterations = f"{a_total_repeat_iters}, {b_total_repeat_iters}"

        metric_metadata: MetricMetadata = c.metric_metadata

        median_cell = format_value_cell(
            c.a_metric.median(),
            c.b_metric.median(),
            infix = format_verdict(c.verdict),
            suffix = f"({stat_label}: {c.comparison_result:.3f})"
        )
        min_cell = format_value_cell(c.a_metric.min(), c.b_metric.min())
        max_cell = format_value_cell(c.a_metric.max(), c.b_metric.max())
        stdev_cell = format_value_cell(c.a_metric.stdev(), c.b_metric.stdev())
        cv_cell = format_value_cell(c.a_metric.cv(), c.b_metric.cv())

        rows.append(
            [
                f"{c.benchmark_name}:{iterations}",
                metric_metadata.name_short,
                median_cell,
                min_cell,
                max_cell,
                stdev_cell,
                cv_cell
            ]
        )

    table = tabulate(
        tabular_data = rows,
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
        for r in comparisons
        if r.verdict == Verdict.REGRESSION
    ]
    out.append(f"Regressions ({len(regressions)}): {regressions}")

    print("\n".join(out))


def print_analysis_reports(
    reports: list[AnalysisReport],
    is_verbose: bool = False,
) -> None:
    for report in reports:
        print(report.title)
        print_report_paths("Baseline", report.baseline_reports)
        print_report_paths("Candidate", report.candidate_reports)
        print()

        unique_device: list[Device] = get_unique_devices(
            [
                r.device
                for r in report.baseline_reports + report.candidate_reports
            ]
        )

        if len(unique_device) > 1:
            logger.warning("multiple benchmark devices detected in the same group, comparison result may not bias")

        for device in unique_device:
            print("Device Specifications:")
            print_device_specifications(device)
            print()

        comparisons_by_name: dict[str, list[BenchmarkComparisonResult]] = defaultdict(list)
        comparisons_by_method: dict[str, list[BenchmarkComparisonResult]] = defaultdict(list)

        for comparison in report.comparisons:
            comparisons_by_name[comparison.benchmark_name].append(comparison)
            comparisons_by_method[comparison.comparison_method].append(comparison)

        if is_verbose:
            total_runtime: float = sum(
                calc_total_runtime(list(rr.benchmarks.values()))
                for rr in report.baseline_reports + report.candidate_reports
            )

            for comparison in comparisons_by_name.values():
                print_comparison_details(comparison, total_runtime_s=total_runtime)
                print()

        for method, comparison in comparisons_by_method.items():
            config = COMPARE_METHODS[method]
            print_summary_table(comparison, config["state"], config["header"])
            print()
