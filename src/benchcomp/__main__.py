import logging
import sys
from collections import defaultdict
from pathlib import Path

from benchcomp.common import AnalysisReport, Benchmark, BenchmarkReport
from benchcomp.compare import COMPARE_METHODS, compare_benchmarks
from benchcomp.console_renderer import print_file_pair_mapping, print_analysis_reports
from benchcomp.parser_cli import parse_commandline_args
from benchcomp.parser_macrobenchmark import load_macrobenchmark_report

logger = logging.getLogger(__name__)

def resolve_search_path(path: Path) -> list[Path]:
    """
    Resolve a path into a list of JSON benchmark files.

    If the given path is:
    - A directory: returns all `.json` files in that directory (non-recursive), sorted.
    - A file: returns a single-item list containing that file.
    - Neither an existing file nor directory: returns an empty list.

    Args:
        path (Path): Path to a JSON file or a directory containing JSON files.

    Returns:
        list[Path]: A list of resolved JSON file paths.
    """

    if path.is_dir():
        return sorted(path.glob("*.json"))

    if path.is_file():
        return [path]

    return []


def load_reports(files: list[Path]) -> list[BenchmarkReport]:
    """Parses benchmark files into BenchmarkReport objects, skipping invalid ones."""
    reports: list[BenchmarkReport] = []
    for f in files:
        try:
            report: BenchmarkReport | None = load_macrobenchmark_report(f)
            if report:
                reports.append(report)
            else:
                logger.error(f"invalid benchmark report '{f}', skipping.\n")
            raise OSError("test")
        except OSError:
            logger.exception(f"failed to open file {f} for reading.")
    return reports


def group_benchmarks_by_name(reports: list[BenchmarkReport]) -> dict[str, list[Benchmark]]:
    """Groups benchmark results by their names across multiple reports."""
    benchmarks: dict[str, list[Benchmark]] = defaultdict(list)
    for report in reports:
        for name, benchmark in report.benchmarks.items():
            benchmarks[name].append(benchmark)
    return benchmarks


def get_common_benchmark_pairs(
    baseline_benchmarks: dict[str, list[Benchmark]] | dict[str, Benchmark],
    candidate_benchmarks: dict[str, list[Benchmark]] | dict[str, Benchmark]
) -> tuple[list[str], dict[str, tuple[list[Benchmark], list[Benchmark]]]]:
    baseline_benchmarks_by_name: dict[str, list[Benchmark]] = {}
    candidate_benchmarks_by_name: dict[str, list[Benchmark]] = {}

    for name, bench in baseline_benchmarks.items():
        if isinstance(bench, list):
            baseline_benchmarks_by_name[name] = bench
        else:
            baseline_benchmarks_by_name[name] = [bench]

    for name, bench in candidate_benchmarks.items():
        if isinstance(bench, list):
            candidate_benchmarks_by_name[name] = bench
        else:
            candidate_benchmarks_by_name[name] = [bench]

    missing_in_baseline: set[str] = candidate_benchmarks_by_name.keys() - baseline_benchmarks_by_name.keys()
    missing_in_candidate: set[str] = baseline_benchmarks_by_name.keys() - candidate_benchmarks_by_name.keys()
    if missing_in_candidate:
        logger.warning(f"Benchmarks present in baseline but missing in candidate: {missing_in_candidate}")
        print()
    if missing_in_baseline:
        logger.warning(f"Benchmarks present in candidate but missing in baseline: {missing_in_baseline}")
        print()

    common_names: list[str] = list(baseline_benchmarks_by_name.keys() & candidate_benchmarks_by_name.keys())
    pairs: dict[str, tuple[list[Benchmark], list[Benchmark]]] = {
        name: (baseline_benchmarks_by_name[name], candidate_benchmarks_by_name[name])
        for name in common_names
    }

    return common_names, pairs


def main() -> int:
    args = parse_commandline_args()

    frame_time_target_ms = args.frametime_target
    is_verbose = args.verbose
    aggregation_function = args.aggregate_method
    thresholds: dict[str, float] = {
        "stepfit": args.step_fit_threshold,
        "mannwhitneyu": args.pvalue_threshold,
    }

    baseline_files = resolve_search_path(args.baseline)
    candidate_files = resolve_search_path(args.candidate)
    if len(baseline_files) <= 0:
        logger.critical('baseline has no Macrobenchmark results')
        return 1
    if len(candidate_files) <= 0:
        logger.critical('candidate has no Macrobenchmark results')
        return 1

    baseline_reports: list[BenchmarkReport] = load_reports(baseline_files)
    candidate_reports: list[BenchmarkReport] = load_reports(candidate_files)
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
            _, pairs = get_common_benchmark_pairs(b.benchmarks, c.benchmarks)
            comparison_groups.append((title, pairs, [b], [c]))
    else:
        title = f"Comparing Benchmark Run (Aggregation: {aggregation_function})"
        b_benchmarks = group_benchmarks_by_name(baseline_reports)
        c_benchmarks = group_benchmarks_by_name(candidate_reports)
        _, pairs = get_common_benchmark_pairs(b_benchmarks, c_benchmarks)
        comparison_groups.append((title, pairs, baseline_reports, candidate_reports))

    analysis_reports: list[AnalysisReport] = []
    for title, benchmark_pairs, b_reps, c_reps in comparison_groups:
        analysis_report = AnalysisReport(title=title, baseline_reports=b_reps, candidate_reports=c_reps)
        for name, (b_list, c_list) in benchmark_pairs.items():
            for method, _ in COMPARE_METHODS.items():
                comparison_result = compare_benchmarks(
                    b_list,
                    c_list,
                    comparison_method=method,
                    threshold=thresholds[method],
                    frame_time_target=frame_time_target_ms,
                    aggregation_function=aggregation_function,
                )

                if comparison_result:
                    analysis_report.comparisons.append(comparison_result)
                else:
                    logger.warning(f"failed to compare '{name}' using '{method}' method")

        analysis_reports.append(analysis_report)

    print_analysis_reports(analysis_reports, is_verbose=is_verbose)

    return 0


if __name__ == "__main__":
    sys.exit(main())
