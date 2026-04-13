import logging
import sys
from collections import defaultdict
from pathlib import Path

from benchcomp.console_renderer import format_analysis_report, print_file_pair_mapping
from benchcomp.core import (
    COMPARE_METHODS,
    AnalysisReport,
    Benchmark,
    BenchmarkReport,
    compare_benchmarks,
    set_frame_time_target_ms,
)
from benchcomp.json_serializer import json_write_analysis_report
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
        except OSError:
            logger.exception(f"failed to open file {f} for reading.")
    return reports


def aggregate_benchmarks(
    reports: list[BenchmarkReport],
    aggregate_function: str,
) -> list[Benchmark]:
    benchmarks_agg: list[Benchmark] = []

    benchmark_by_id: dict[str, list[Benchmark]] = defaultdict(list)
    for report in reports:
        for bench in report.benchmarks:
            benchmark_by_id[bench.id].append(bench)

    for benchmarks in benchmark_by_id.values():
        benchmarks_agg.append(
            Benchmark.aggregate(
                benchmarks,
                aggregate_function
            )
        )

    return benchmarks_agg


def main() -> int:
    conf = parse_commandline_args()

    set_frame_time_target_ms(conf.frame_time_target_ms)

    baseline_files = resolve_search_path(conf.baseline_path)
    if len(baseline_files) <= 0:
        logger.critical('baseline has no Macrobenchmark results')
        return 1

    candidate_files = resolve_search_path(conf.candidate_path)
    if len(candidate_files) <= 0:
        logger.critical('candidate has no Macrobenchmark results')
        return 1

    baseline_reports: list[BenchmarkReport] = load_reports(baseline_files)
    candidate_reports: list[BenchmarkReport] = load_reports(candidate_files)

    for r in (baseline_reports + candidate_reports):
        r.device.alias = conf.device_alias

    comparison_groups: list[
        tuple[
            tuple[list[Benchmark], list[Benchmark]],
            list[BenchmarkReport],
            list[BenchmarkReport],
        ]
    ] = []

    if conf.aggregate_function:
        b_benchmarks = aggregate_benchmarks(baseline_reports, conf.aggregate_function)
        c_benchmarks = aggregate_benchmarks(candidate_reports, conf.aggregate_function)
        comparison_groups.append(
            (
                (b_benchmarks, c_benchmarks),
                baseline_reports,
                candidate_reports,
            )
        )
    else:
        min_len = min(len(baseline_reports), len(candidate_reports))
        if len(baseline_reports) != len(candidate_reports):
            logger.warning(f"length mismatch, using first {min_len} samples. baseline: {len(baseline_reports)}, candidate: {len(candidate_reports)}")

        if min_len > 1:
            print_file_pair_mapping(baseline_files, candidate_files)
            print()

        mismatch_count = sum(
            b_filepath.name != c_filepath.name
            for b_filepath, c_filepath in zip(baseline_files, candidate_files)
        )
        if mismatch_count > 0:
            logger.warning("filename mapping mismatch detected. Output prediction may be incorrect")

        for i in range(min_len):
            comparison_groups.append(
                (
                    (baseline_reports[i].benchmarks, candidate_reports[i].benchmarks),
                    [baseline_reports[i]],
                    [candidate_reports[i]],
                )
            )

    compare_methods = conf.methods
    thresholds = {"stepfit": conf.fit, "mannwhitneyu": conf.alpha}
    analysis_reports: list[AnalysisReport] = []
    for benchmark_pairs, b_reports, c_reports in comparison_groups:
        try:
            comparisons = compare_benchmarks(
                a=benchmark_pairs[0],
                b=benchmark_pairs[1],
                methods=compare_methods,
                thresholds=thresholds,
                measures=conf.measures,
            )

            report =  AnalysisReport(
                baseline_reports=b_reports,
                candidate_reports=c_reports,
                comparisons=comparisons,
                methods=[ COMPARE_METHODS[method] for method in compare_methods]
            )
            analysis_reports.append(report)

            if len(report.get_devices()) > 1:
                logger.warning("multiple device configurations detected in the same patch; benchmark comparison results may be skewed")

        except Exception:
            benchmark_ids = [bench.id for bench in benchmark_pairs[0]]
            logger.exception(f"Failed to compare benchmarks pair: {benchmark_ids}")

    for analysis_report in analysis_reports:
        print(format_analysis_report(analysis_report, is_verbose=conf.is_verbose))

    if conf.output_path:
        if len(analysis_reports) == 1:
            with open(conf.output_path, "w") as f:
                f.write(json_write_analysis_report(analysis_reports[0]))
        else:
            logger.error("file output does not support writing multiple analysis reports at once")

    return 0


if __name__ == "__main__":
    sys.exit(main())
