import argparse
from dataclasses import dataclass
from pathlib import Path

from benchcomp.__version__ import __version__
from benchcomp.compare import (
    AGGREGATE_FUNCTIONS,
    DEFAULT_AGGREGATE_FUNCTION,
    DEFAULT_FRAME_TIME_TARGET_MS,
    DEFAULT_P_VALUE_THRESHOLD,
    DEFAULT_STEP_FIT_THRESHOLD,
)

@dataclass
class Config:
    baseline_path: Path
    candidate_path: Path
    frame_time_target_ms: float
    fit: float
    alpha: float
    is_verbose: bool
    aggregate_function: str


def parse_commandline_args() -> Config:
    parser = argparse.ArgumentParser(
        prog="benchcomp",
        description="Compare between macrobenchmark reports",
    )

    # Positional Arguments
    parser.add_argument(
        "baseline",
        type=Path,
        help="Path to the baseline macrobenchmark report directory or file",
    )
    parser.add_argument(
        "candidate",
        type=Path,
        help="Path to the candidate macrobenchmark report directory or file",
    )

    # Optional Arguments
    parser.add_argument(
        "--frametime",
        dest="frame_time_target_ms",
        type=float,
        default=DEFAULT_FRAME_TIME_TARGET_MS,
        metavar="MS",
        help=f"Target frame time in milliseconds (Default: {DEFAULT_FRAME_TIME_TARGET_MS:.3f}ms)",
    )
    parser.add_argument(
        "--fit",
        dest="fit",
        type=float,
        default=DEFAULT_STEP_FIT_THRESHOLD,
        metavar="VALUE",
        help=f"Threshold for step fit analysis (Default: {DEFAULT_STEP_FIT_THRESHOLD:.3f})",
    )
    parser.add_argument(
        "--alpha",
        dest="alpha",
        type=float,
        default=DEFAULT_P_VALUE_THRESHOLD,
        metavar="VALUE",
        help=f"P-value threshold for Mann-Whitney U-test significance (Default: {DEFAULT_P_VALUE_THRESHOLD:.3f})",
    )
    parser.add_argument(
        "--aggregate",
        dest="aggregate_function",
        type=str,
        metavar="VALUE",
        choices=AGGREGATE_FUNCTIONS.keys(),
        default=DEFAULT_AGGREGATE_FUNCTION,
        help=f"Method to aggregate benchmark results. Options: {', '.join(AGGREGATE_FUNCTIONS)} (default: %(default)s)"
    )
    parser.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        help="Print additional metric information"
    )
    parser.add_argument(
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
        is_verbose=args.verbose
    )
