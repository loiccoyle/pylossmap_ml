import argparse
import logging
import sys
from pathlib import Path
from typing import List

import pandas as pd

from .fetchers import (
    LOGGER,
    fetch,
    fetch_ion_stable,
    fetch_proton_preramp,
    fetch_proton_stable,
)

PRECONFIGURED_MAP = {
    "ion_stable": fetch_ion_stable,
    "proton_preramp": fetch_proton_preramp,
    "proton_stable": fetch_proton_stable,
}


def parse_args(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "destination_dir",
        type=Path,
        help="The directory in which to write the BLM data.",
    )
    parser.add_argument(
        "--preconfigured",
        type=str,
        choices=PRECONFIGURED_MAP.keys(),
        help="Preconfigured fetching.",
    )
    parser.add_argument(
        "--start",
        type=pd.to_datetime,
        help="Start time.",
    )
    parser.add_argument(
        "--end",
        type=pd.to_datetime,
        help="End time.",
    )
    parser.add_argument(
        "--beam-modes-fills",
        type=str,
        nargs="*",
        default=["STABLE"],
        help="Fetch data for fills which contain the provided beam modes.",
    )
    parser.add_argument(
        "--beam-modes-fetch",
        type=str,
        nargs="*",
        default=["STABLE"],
        help="Fetch data of the provided beam modes.",
    )
    parser.add_argument(
        "--BLM_var",
        type=str,
        default="LHC.BLMI:LOSS_RS01",
        help="The timber BLM data variable.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Verbosity.",
        action="count",
        default=0,
    )
    return parser.parse_args(args)


def main() -> None:
    args = parse_args(sys.argv[1:])

    if args.verbose > 0:
        verbose_map = {1: logging.INFO, 2: logging.DEBUG}
        level = verbose_map[args.verbose]
        LOGGER.setLevel(level)

    LOGGER.debug("Args: %s", args)
    if args.preconfigured is not None:
        fetch_method = PRECONFIGURED_MAP[args.preconfigured]
        LOGGER.info("Using method: %s", fetch_method)
        fetch_method(args.destination_dir, BLM_var=args.BLM_var)
    else:
        if args.start is None:
            raise argparse.ArgumentTypeError("No --start time provided.")
        if args.stop is None:
            raise argparse.ArgumentTypeError("No --stop time provided.")

        fetch(
            args.start.tz_localize("Europe/Zurich"),
            args.stop.tz_localize("Europe/Zurich"),
            args.destination_dir,
            args.beam_modes_fills,
            args.beam_modes_fetch,
            args.BLM_var,
        )
