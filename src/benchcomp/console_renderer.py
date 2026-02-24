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
    calc_total_run_time,
)
from benchcomp.compare import COMPARE_METHODS, BenchmarkCompareResult, Verdict


def print_file_pair_mapping(baseline: list[Path], candidate: list[Path]) -> None:
    def relative_diff_paths(a: Path, b: Path) -> tuple[str, str]:
        a_parts = a.resolve().parts
        b_parts = b.resolve().parts

        prefix_len = 0
        for x, y in zip(a_parts, b_parts):
            if x == y:
                prefix_len += 1
            else:
                break

        # If no common root, just return original
        if prefix_len == 0:
            return str(a), str(b)

        common_root = Path(*a_parts[:prefix_len])

        return (
            str(a.relative_to(common_root)),
            str(b.relative_to(common_root)),
        )

    title: str = "Macrobenchmark Report File Mapping"
    rows = []

    mismatch_count = 0
    for i, (b_filepath, c_filepath) in enumerate(zip(baseline, candidate)):
        index: str
        if b_filepath.name != c_filepath.name:
            mismatch_count += 1
            index = f"* {i}"
        else:
            index = f"{i}"

        rows.append(
            [
                index,
                *relative_diff_paths(
                    b_filepath,
                    c_filepath
                )
            ]
        )

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
    print(f"Device ({device.name}):")
    print(f"  Brand    : {device.brand}")
    print(f"  Model    : {device.model}")
    print(f"  Cores    : {device.cpu_cores}")
    print(f"  Core Freq: {device.cpu_freq}Hz")
    print(f"  Memory   : {device.mem_size}MB")
    print(f"  Emulator : {device.emulated}")


def print_compare_method_results(
    results: list[BenchmarkCompareResult],
    total_run_time_s: float | None
) -> None:
    ref_cr = results[0]

    benchmark_name: str = ref_cr.benchmark_name
    benchmark_class: str = ref_cr.benchmark_class
    metric_metadata: MetricMetadata = ref_cr.metric_metadata

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

    verdicts: list[str] = []
    statistics: list[str] = []

    for r in results:
        verdicts.append(f"{r.method}{r.verdict}")
        statistics.append(f"{r.method}: {r.result:.3f}")

    print(f"> Benchmark '{benchmark_name}':")
    print(f"    Class               : {benchmark_class}")
    print(f"    Run Time (s)        : {ab_run_time_s:.3f} (Baseline: {a_total_rum_time_s:.3f}, Candidate: {b_total_rum_time_s:.3f}){run_time_percent}")
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


def tabulate_(
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


def render_to_console(
    reports: list[AnalysisReport],
    is_verbose: bool = False,
) -> None:
    for r in reports:
        print(r.title)
        print_report_paths("Baseline", r.baseline_reports)
        print_report_paths("Candidate", r.candidate_reports)
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
        #

        results_by_name: dict[str, list[BenchmarkCompareResult]] = defaultdict(list)
        results_by_method: dict[str, list[BenchmarkCompareResult]] = defaultdict(list)

        for res in r.results:
            results_by_name[res.benchmark_name].append(res)
            results_by_method[res.method].append(res)

        if is_verbose:
            total_run_time: float = sum(
                calc_total_run_time(list(rr.benchmarks.values()))
                for rr in r.baseline_reports + r.candidate_reports
            )

            for bench_results in results_by_name.values():
                print_compare_method_results(bench_results, total_run_time_s=total_run_time)
                print()

        for method, bench_results in results_by_method.items():
            config = COMPARE_METHODS[method]
            print(tabulate_(bench_results, config["state"], config["header"]))
            print()
