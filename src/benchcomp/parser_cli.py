import argparse
from pathlib import Path

from benchcomp.compare import (
    AGGREGATE_METHODS,
    DEFAULT_FRAME_TIME_TARGET_MS,
    DEFAULT_P_VALUE_THRESHOLD,
    DEFAULT_STEP_FIT_THRESHOLD,
)


def parse_commandline_args() -> argparse.Namespace:
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
        dest="frametime_target",
        type=float,
        default=DEFAULT_FRAME_TIME_TARGET_MS,
        metavar="MS",
        help=f"Target frame time in milliseconds (Default: {DEFAULT_FRAME_TIME_TARGET_MS:.3f}ms)",
    )
    parser.add_argument(
        "--fit",
        dest="step_fit_threshold",
        type=float,
        default=DEFAULT_STEP_FIT_THRESHOLD,
        metavar="VALUE",
        help=f"Threshold for step fit analysis (Default: {DEFAULT_STEP_FIT_THRESHOLD:.3f})",
    )
    parser.add_argument(
        "--alpha",
        dest="pvalue_threshold",
        type=float,
        default=DEFAULT_P_VALUE_THRESHOLD,
        metavar="VALUE",
        help=f"P-value threshold for Mann-Whitney U-test significance (Default: {DEFAULT_P_VALUE_THRESHOLD:.3f})",
    )
    parser.add_argument(
        "--aggregate",
        dest="aggregate_method",
        type=str,
        metavar="VALUE",
        choices=AGGREGATE_METHODS,
        default=AGGREGATE_METHODS[0],
        help=f"Method to aggregate benchmark results. Options: {', '.join(AGGREGATE_METHODS)} (default: %(default)s)"
    )
    parser.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        help="Print additional metric information"
    )

    return parser.parse_args()
