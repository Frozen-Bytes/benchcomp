import argparse
from dataclasses import dataclass
from pathlib import Path

from benchcomp.__version__ import __version__
from benchcomp.core import (
    AGGREGATE_FUNCTIONS,
    COMPARE_METHODS,
    DEFAULT_FRAME_TIME_TARGET_MS,
    DEFAULT_THRESHOLDS,
    MEASURES,
)

DEFAULT_MEASURES: list[str] = [
    "TID",          # Time To Initial Display
    "FFD",          # Frame Freeze Duration
    "MEM_RSS_MAX",  # Total RSS Memory Usage Max
    "MEM_RSS_LAST", # Total RSS Memory Usage Last
]

DEFAULT_METHOD: str = "mannwhitneyu"

@dataclass
class Config:
    baseline_path: Path
    candidate_path: Path
    frame_time_target_ms: float
    fit: float
    alpha: float
    is_verbose: bool
    measures: set[str]
    methods: set[str]
    aggregate_function: str | None


def parse_commandline_args() -> Config:
    parser = argparse.ArgumentParser(
        prog="benchcomp",
        description="Compare between macrobenchmark reports",
        usage="%(prog)s [OPTION ...] BASELINE CANDIDATE",
        formatter_class=lambda prog: argparse.HelpFormatter(prog, width=80)
    )

    # Positional Arguments
    parser.add_argument(
        "baseline",
        type=Path,
        metavar="BASELINE",
        help="path to the baseline macrobenchmark report directory or file",
    )
    parser.add_argument(
        "candidate",
        type=Path,
        metavar="CANDIDATE",
        help="path to the candidate macrobenchmark report directory or file",
    )

    measures_group = parser.add_argument_group(
        "measures options",
        "options related to benchmark metric measures",
    )
    measures_group.add_argument(
        "-g",
        "--aggregate",
        dest="aggregate_function",
        type=str,
        metavar="FUNC",
        choices=AGGREGATE_FUNCTIONS.keys(),
        default=None,
        help="method to aggregate benchmark results. Options: %(choices)s",
    )
    measures_group.add_argument(
        "-m",
        "--measures",
        dest="measures",
        type=str,
        metavar="MEASURE",
        default=DEFAULT_MEASURES,
        nargs="+",
        choices=MEASURES,
        help="list of measures to compute for benchmark results. Options: %(choices)s",
    )
    measures_group.add_argument(
        "-a",
        "--all-measures",
        dest="do_all_measures",
        action="store_true",
        default=False,
        help="compare all available benchmark metric measures.",
    )
    measures_group.add_argument(
        "--frametime",
        dest="frame_time_target_ms",
        type=float,
        default=DEFAULT_FRAME_TIME_TARGET_MS,
        metavar="MS",
        help="target frame time in milliseconds. (Default: %(default).3fms)",
    )

    comp_method_group = parser.add_argument_group(
        "compare options",
        "options related to compare methods",
    )
    comp_method_group.add_argument(
        "-M",
        "--methods",
        dest="methods",
        type=str,
        metavar="METHOD",
        default=[DEFAULT_METHOD],
        nargs="+",
        choices=COMPARE_METHODS.keys(),
        help="list of statistical methods used to compare benchmark. Options: %(choices)s",
    )
    comp_method_group.add_argument(
        "-t",
        "--fit",
        dest="fit",
        type=float,
        default=DEFAULT_THRESHOLDS["stepfit"],
        metavar="FIT",
        help="threshold for step fit analysis (Default: %(default).3f)",
    )
    comp_method_group.add_argument(
        "-p",
        "--alpha",
        dest="alpha",
        type=float,
        default=DEFAULT_THRESHOLDS["mannwhitneyu"],
        metavar="ALPHA",
        help="p-value threshold for Mann-Whitney U-test significance (Default: %(default).3f)",
    )
    comp_method_group.add_argument(
        "-A",
        "--all-methods",
        dest="do_all_methods",
        action="store_true",
        default=False,
        help="compare using all available compare methods.",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        help="print additional metric information",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    args = parser.parse_args()

    return Config(
        baseline_path=args.baseline,
        candidate_path=args.candidate,
        frame_time_target_ms=args.frame_time_target_ms,
        fit=args.fit,
        alpha=args.alpha,
        aggregate_function=args.aggregate_function,
        measures=set(MEASURES) if args.do_all_measures else args.measures,
        methods=set(COMPARE_METHODS.keys()) if args.do_all_methods else args.methods,
        is_verbose=args.verbose,
    )
