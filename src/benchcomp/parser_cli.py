import argparse

from benchcomp.compare import (
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
        "baseline_dir",
        type=str,
        help="Path to the baseline macrobenchmark report directory",
    )
    parser.add_argument(
        "candidate_dir",
        type=str,
        help="Path to the candidate macrobenchmark report directory",
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

    return parser.parse_args()
