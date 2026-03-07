import logging
import math
import textwrap
from pathlib import Path

from tabulate import tabulate

from benchcomp.core import (
    AnalysisReport,
    BenchmarkComparisonResult,
    BenchmarkReport,
    Device,
    Verdict,
)

logger = logging.getLogger(__name__)

def to_sec_from_ns(ns: float | int) -> float:
    return ns / 1_000_000_000.0


def _relative_diff(a: float, b: float) -> float:
    return (b - a) / a if not math.isclose(a, 0.0) else 0.0


def _format_device(device: Device) -> str:
    lines = [
        f"Device ({device.name}):",
        f"    Brand : {device.brand}",
        f"    Model : {device.model}",
        f"    Cores : {device.cpu_cores} @ {device.cpu_freq}Hz",
        f"    Memory: {device.mem_size_mb}MB",
        f"    Type  : {'Emulated' if device.emulated else 'Physical'}",
    ]
    return "\n".join(lines)


def _format_benchmark_comparison(
    comparison: BenchmarkComparisonResult,
    report_runtime_sec: float | int = 0,
) -> str:
    def _format_run(runs: list[float]) -> str:
        return ", ".join(f"{v:.3f}" for v in runs)

    lines: list[str] = []

    lines.append(f"Benchmark '{comparison.benchmark_name}':")
    lines.append(f"  Class              : {comparison.benchmark_class}")

    baseline_runtime_sec = to_sec_from_ns(comparison.a.total_runtime_ns)
    candidate_runtime_sec = to_sec_from_ns(comparison.b.total_runtime_ns)
    total_runtime_sec = baseline_runtime_sec + candidate_runtime_sec
    runtime_percentage = f" ({total_runtime_sec / float(report_runtime_sec) :2.2%})" if report_runtime_sec != 0 else ""
    lines.append(f"  Run Time (s)       : {total_runtime_sec:.3f} (Baseline: {baseline_runtime_sec:.3f}, Candidate: {candidate_runtime_sec:.3f}){runtime_percentage}")

    lines.append(f"  Warmup   Iterations: (Baseline: {comparison.a.warmup_iterations}, Candidate: {comparison.b.warmup_iterations})")
    lines.append(f"  Repeated Iterations: (Baseline: {comparison.a.repeat_iterations}, Candidate: {comparison.b.repeat_iterations})")

    for measurement_result in comparison.results:
        lines.append(f"\n  Metric '{measurement_result.metadata.name}' ({measurement_result.metadata.name_short}):")
        lines.append(f"    Baseline  Runs ({measurement_result.metadata.unit}): [{_format_run(measurement_result.a.runs)}]")
        lines.append(f"    Candidate Runs ({measurement_result.metadata.unit}): [{_format_run(measurement_result.b.runs)}]")

        change_percent =  [ ]
        for a, b in zip(measurement_result.a.runs, measurement_result.b.runs):
            change = (b - a) / a if a != 0 else 0
            change_percent.append(f"{change:+2.2%}")
        lines.append(f"    Change             : [{', '.join(change_percent)}]")
        lines.append(f"    Median             : [{measurement_result.a.median:.3f}, {measurement_result.b.median:.3f}] ({measurement_result.b.median - measurement_result.a.median:+.3f} ~ {_relative_diff(measurement_result.a.median, measurement_result.b.median):+2.2%})")

        method_label_width = 0
        for compare_result in measurement_result.result:
            method_label_width = max(method_label_width, len(compare_result.metadata.name))

        for compare_result in measurement_result.result:
            method_str = f"'{compare_result.metadata.name}'"
            line = (
                f"Method {method_str:{method_label_width + 2}}: "
                f"{compare_result.verdict.name}, "
                f"{compare_result.metadata.state_label}={compare_result.statistic:.3f}, "
                f"threshold={comparison.thresholds[compare_result.metadata.id]}"
            )
            lines.append(f"    {line}")

    return "\n".join(lines)


def _build_summary_table(
    comparisons: list[BenchmarkComparisonResult],
    compare_method_id: str = "",
    title: str = "",
) -> str:
    def _format_cell(
        a: float,
        b: float,
        infix: str = "-",
        suffix: str = "",
        unit: str = "",
        computer_diff: bool = False,
    ) -> str:
        if infix:
            infix = f" {infix} "
        if suffix:
            suffix = f" {suffix}"
        diff_str = f" ({(b - a):+.3f} ~ {_relative_diff(a, b):+2.2%})" if computer_diff else ""
        return f"{a:.3f}{unit}{infix}{unit}{b:.3f}{suffix}{diff_str}"

    def _format_verdict(verdict: Verdict ) -> str:
        return {
            Verdict.NOT_SIGNIFICANT: "N  ",
            Verdict.IMPROVEMENT: "I +",
            Verdict.REGRESSION: "R -",
        }.get(verdict, " ")

    def _build_rows(comparison: BenchmarkComparisonResult) -> list[list[str]]:
        rows: list[list[str]] = []

        for measurement_result in comparison.results:
            columns: list[str] = []

            iterations = f"{comparison.a.repeat_iterations}"
            if comparison.a.repeat_iterations != comparison.b.repeat_iterations:
                iterations += f", {comparison.b.repeat_iterations}"
            columns.append(f"{comparison.benchmark_name}:{iterations}")

            metadata = measurement_result.metadata
            columns.append(metadata.name_short)

            for compare_result in measurement_result.result:
                if compare_result.metadata.id == compare_method_id:
                    verdict_str = _format_verdict(compare_result.verdict)
                    columns.append(f"{verdict_str} {compare_result.metadata.state_label} = {compare_result.statistic:+.3f}")
                    break

            a = measurement_result.a
            b = measurement_result.b
            columns.extend(
                [
                    _format_cell(a.median, b.median, computer_diff=True),
                    # _format_cell(a.min, b.min),
                    # _format_cell(a.max, b.max),
                    _format_cell(a.stdev, b.stdev),
                    _format_cell(a.cv, b.cv),
                ]
            )

            rows.append(columns)

        return rows

    lines: list[str] = []

    headers = [
        "Benchmark:Iteration",
        "Metric",
        "Verdict",
        "Median",
        # "Minimum",
        # "Maximum",
        "Standard Deviation",
        "Coefficient of Variance",
    ]

    rows: list[list[str]] = []
    sorted_comparisons = sorted(comparisons, key=lambda c: c.benchmark_id)
    for comparison in sorted_comparisons:
        rows.extend(_build_rows(comparison))

    table = tabulate(rows, headers = headers, tablefmt="rounded_outline")

    if title:
        width = len(table.split('\n')[0])
        lines.append(title.center(width))

    lines.append(table)

    regressions = [ c.benchmark_name for c in comparisons if c.has_regressed(compare_method_id) ]
    lines.append(f"Regressions ({len(regressions)}): {regressions}")

    return "\n".join(lines)


def format_analysis_report(report: AnalysisReport, is_verbose: bool = False) -> str:
    def _format_paths(label: str, reports: list[BenchmarkReport]):
        lines: list[str] = []
        lines.append(f"{label} benchmark reports:")
        for r in reports:
            lines.append(f"  - {r.filepath}")
        return "\n".join(lines)

    lines: list[str] = []

    if report.title:
        lines.append(report.title)

    lines.append(_format_paths("Baseline", report.baseline_reports))
    lines.append(_format_paths("Candidate", report.candidate_reports))

    lines.append("")
    lines.append("Device Specifications:")
    devices = report.get_devices()
    for device in devices:
        lines.append(textwrap.indent(_format_device(device), '  '))
        lines.append("")
    lines.pop()

    if is_verbose:
        total_runtime_s = 0.0
        for compariosn in report.comparisons:
            total_runtime_s += to_sec_from_ns(compariosn.a.total_runtime_ns)
            total_runtime_s += to_sec_from_ns(compariosn.b.total_runtime_ns)

        for comparison in report.comparisons:
            lines.append("")
            lines.append(
                _format_benchmark_comparison(
                    comparison,
                    report_runtime_sec=total_runtime_s,
                )
            )

    for method_metadata in report.methods:
        lines.append("")
        lines.append(
            _build_summary_table(
                report.comparisons,
                compare_method_id=method_metadata.id,
                title=method_metadata.name,
            )
        )

    return "\n".join(lines)


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
