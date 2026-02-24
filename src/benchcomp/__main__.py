import logging
import sys
from collections import defaultdict
from pathlib import Path

from benchcomp.common import AnalysisReport, Benchmark, BenchmarkReport
from benchcomp.compare import COMPARE_METHODS, compare_benchmark
from benchcomp.console_renderer import print_file_pair_mapping, render_to_console
from benchcomp.parser_cli import parse_commandline_args
from benchcomp.parser_macrobenchmark import parse_macrobechmark_report

logger = logging.getLogger(__name__)

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

    frametime_target_ms = args.frametime_target
    is_verbose = args.verbose
    aggregation_method = args.aggregate_method
    thresholds: dict[str, float] = {
        "stepfit": args.step_fit_threshold,
        "mannwhitneyu": args.pvalue_threshold,
    }

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

        print_file_pair_mapping(baseline_files, candidate_files)
        print()

        mismatch_count = sum(
            b_filepath.name != c_filepath.name
            for b_filepath, c_filepath in zip(baseline_files, candidate_files)
        )
        if mismatch_count > 0:
            logger.warning("filename mapping mismatch detected. Output prediction may be incorrect")

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

    aReports: list[AnalysisReport] = []
    for title, benchmark_pairs, b_reps, c_reps in comparison_groups:
        ar = AnalysisReport(title=title, baseline_reports=b_reps, candidate_reports=c_reps)
        for name, (b_list, c_list) in benchmark_pairs.items():
            for method, _ in COMPARE_METHODS.items():
                result = compare_benchmark(
                    b_list,
                    c_list,
                    method=method,
                    threshold=thresholds[method],
                    frametime_target=frametime_target_ms,
                    aggregate=aggregation_method,
                )

                if result:
                    ar.results.append(result)
                else:
                    logger.warning(f"Skipping '{method}' for '{name}'")

        aReports.append(ar)

    render_to_console(aReports, is_verbose=is_verbose)

    return 0


if __name__ == "__main__":
    sys.exit(main())
