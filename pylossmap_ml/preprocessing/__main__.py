import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

from pylossmap import BLMData

from . import preprocessor, spoolers

SPOOLERS = ["SerialH5", "SerialSingleH5"]
PREPROCESSORS = ["NormMaxNoDump", "NormSumNoDump", "PassThrough", "RollingWindowSum"]
LOGGER = logging.getLogger(__name__)


def type_dir(maybe_dir: str) -> Path:
    path_maybe_dir = Path(maybe_dir)
    if not path_maybe_dir.is_dir():
        raise NotADirectoryError(f"'{maybe_dir}' is not a directory.")
    return path_maybe_dir


def type_file(maybe_file: str) -> Path:
    path_maybe_file = Path(maybe_file)
    if not path_maybe_file.is_dir():
        raise FileNotFoundError(f"'{maybe_file}' is not a file.")
    return path_maybe_file


def type_spooler(spooler: str) -> spoolers.BaseSpooler:
    return getattr(spoolers, spooler)


def type_preprocessor(pre: str) -> preprocessor.BasePreprocessor:
    return getattr(preprocessor, pre)


def parse_args(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess a BLM dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "raw_data_dir",
        help="Directory containing the raw BLM data files.",
        type=type_dir,
    )
    parser.add_argument(
        "preprocessor",
        help="Data preprocessor.",
        choices=PREPROCESSORS,
    )
    parser.add_argument(
        "spooler",
        help="Data spooler.",
        choices=SPOOLERS,
    )
    parser.add_argument(
        "destination",
        help="Destination.",
        type=Path,
    )
    parser.add_argument(
        "--preprocessor-kwargs",
        help='Preprocessor args, json format. e.g. "{"drop_blm_names": true}"',
        type=json.loads,
    )
    parser.add_argument(
        "--h5-kwargs",
        help='pd.to_hdf & pd.from_hdf args, json format. e.g. "{"complib": "blosc"}"',
        type=json.loads,
    )
    parser.add_argument(
        "--blm-filter",
        help="Only a subset of the BLMs, e.g. 'BLM[BQTE2M]I' for ionization chamber BLMs.",
        type=str,
    )
    parser.add_argument(
        "--blm-list-file",
        help="File containing the subset of BLMs, one per line.",
        type=type_file,
    )
    parser.add_argument(
        "--concat-path",
        help="Concatenate the preprocessed data into a single h5 file, if implemented in the spooler.",
        type=Path,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Verbosity.",
        action="count",
        default=0,
    )
    return parser.parse_args(args)


def get_blm_list_from_filter(data_file: Path, filter: str) -> List[str]:
    """Load some BLM data, aplly the BLM regex filter and return what is left."""

    blm_data = BLMData.load(data_file)
    LOGGER.debug("Number of BLMs pre filter: %s", len(blm_data.columns))
    filtered = blm_data.filter(filter)
    LOGGER.debug("Number of BLMs post filter: %s", len(filtered.columns))
    return filtered.df.columns.to_list()


def get_blm_list_from_file(blm_file: Path) -> List[str]:
    """Read the BLMs from a file, one per line."""

    with open(blm_file, "r") as fp:
        blms = [line.strip() for line in fp]
    LOGGER.info("Number of blms in the file: %s", len(blms))
    return blms


def main() -> None:
    args = parse_args(sys.argv[1:])

    if args.verbose > 0:
        verbose_map = {1: logging.INFO, 2: logging.DEBUG}
        level = verbose_map[args.verbose]
        LOGGER.setLevel(level)

    LOGGER.debug("Args: %s", args)
    if not args.destination.parent.is_dir():
        LOGGER.debug("Creating destination parent dir.")
        args.destination.parent.mkdir(parents=True)

    raw_data_files = list(args.raw_data_dir.glob("*"))
    if args.verbose > 0:
        LOGGER.info("Raw data dir: %s", args.raw_data_dir)
        for path in raw_data_files[:5]:
            LOGGER.info(path.name)

    if args.blm_list_file is not None:
        blm_list = get_blm_list_from_file(args.blm_list_file)
    elif args.blm_filter is not None:
        blm_list = get_blm_list_from_filter(raw_data_files[0], args.blm_filter)
    else:
        blm_list = None

    preproc = type_preprocessor(
        args.preprocessor(blm_list=blm_list, **args.preprocessor_kwargs)
    )
    LOGGER.info("Preprocessor: %s", preprocessor)
    spooler = type_spooler(args.spooler(preproc, raw_data_files, args.destination))
    LOGGER.info("Spooler: %s", spooler)
    LOGGER.info("Spooling")
    spooler.spool(**args.h5_kwargs)
    if args.concat_path is not None:
        if hasattr(spooler, "concat"):
            LOGGER.info("Running concat.")
            spooler.concat(**args.h5_kwrags)
        else:
            LOGGER.warning("Spooler does not implement concat.")
