import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Callable, List

from pylossmap import BLMData

from ..utils import PARTICLE_TYPE_MAP
from . import preprocessor, spoolers

SPOOLERS = ["SerialH5", "SerialSingleH5"]
PREPROCESSORS = [
    "NoDump",
    "NoDumpDiff",
    "NormMaxNoDump",
    "NormMaxNoDumpDiff",
    "NormSumNoDump",
    "NormSumNoDumpDiff",
    "PassThrough",
    "RollingWindowSum",
]
LOGGER = logging.getLogger(__name__)


def type_dir(maybe_dir: str) -> Path:
    path_maybe_dir = Path(maybe_dir)
    if not path_maybe_dir.is_dir():
        raise NotADirectoryError(f"'{maybe_dir}' is not a directory.")
    return path_maybe_dir


def type_file(maybe_file: str) -> Path:
    path_maybe_file = Path(maybe_file)
    if not path_maybe_file.is_file():
        raise FileNotFoundError(f"'{maybe_file}' is not a file.")
    return path_maybe_file


def type_spooler(spooler: str) -> Callable[..., spoolers.BaseSpooler]:
    return getattr(spoolers, spooler)


def type_preprocessor(pre: str) -> Callable[..., preprocessor.BasePreprocessor]:
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
    # TODO: this does not work because the argparse complains about the positional args
    parser.add_argument(
        "--load-defaults-path",
        help="Load the default args from a json file, e.g. .preprocess_info.json",
        type=type_file,
    )
    parser.add_argument(
        "--preprocessor-kwargs",
        help='Preprocessor args, json format. e.g. "{"drop_blm_names": false}"',
        type=json.loads,
        default={},
    )
    parser.add_argument(
        "--h5-kwargs",
        help='pd.to_hdf & pd.from_hdf args, json format. e.g. "{"complib": "blosc"}"',
        type=json.loads,
        default={},
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
        "--particle-type",
        help="Only preprocess fills with the provided particle type.",
        choices=PARTICLE_TYPE_MAP.values(),
        type=str,
    )
    parser.add_argument(
        "--concat-path",
        help="Concatenate the preprocessed data into a single h5 file, if implemented in the spooler.",
        type=Path,
    )
    parser.add_argument(
        "--copy-dataset-info",
        help="Copy the dataset info file over.",
        action="store_true",
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
    LOGGER.debug("Number of BLMs pre filter: %s", len(blm_data.df.columns))
    filtered = blm_data.filter(filter)
    LOGGER.debug("Number of BLMs post filter: %s", len(filtered.df.columns))
    return filtered.df.columns.to_list()


def get_blm_list_from_file(blm_file: Path) -> List[str]:
    """Read the BLMs from a file, one per line."""

    with open(blm_file, "r") as fp:
        blms = [line.strip() for line in fp]
    LOGGER.info("Number of blms in the file: %s", len(blms))
    return blms


def args_to_file(args_dict: dict) -> None:
    """Write the args to file."""
    if args_dict["destination"].is_dir():
        destination_file = args_dict["destination"] / ".preprocess_info.json"
    else:
        destination_file = (
            args_dict["destination"].parent
            / f".{args_dict['destination'].stem}_preprocess_info.json"
        )

    to_str = ["blm_list_file", "concat_path", "destination", "raw_data_dir"]
    for key in to_str:
        if args_dict[key] is not None:
            args_dict[key] = str(args_dict[key].resolve())

    with open(destination_file, "w") as fp:
        LOGGER.info(
            "Writing args to file: %s",
            destination_file.resolve(),
        )
        json.dump(args_dict, fp, indent=2)


def args_from_file(json_file: Path) -> dict:
    """Read a file written by `args_to_file` and return the dictionary."""
    with open(json_file, "r") as fp:
        LOGGER.info("Reading args from: %s", json_file)
        args_dict = json.load(fp)

    to_path = ["blm_list_file", "concat_path", "destination", "raw_data_dir"]
    for key in to_path:
        if args_dict[key] is not None:
            args_dict[key] = Path(args_dict[key])
    return args_dict


def copy_dataset_info(raw_data_dir: Path, destination: Path) -> None:
    """Copy the dataset info json file over."""
    info_file = raw_data_dir / ".dataset_info.json"
    if info_file.is_file():
        if destination.is_dir():
            destination_file = destination / ".dataset_info.json"
        else:
            destination_file = (
                destination.parent / f".{destination.stem}_dataset_info.json"
            )

        LOGGER.info(
            "Copying dataset info: %s -> %s",
            info_file.resolve(),
            destination_file.resolve(),
        )
        shutil.copy(info_file, destination_file)


def main() -> None:
    args = vars(parse_args(sys.argv[1:]))

    if args["verbose"] > 0:
        verbose_map = {1: logging.INFO, 2: logging.DEBUG}
        level = verbose_map[args["verbose"]]
        LOGGER.setLevel(level)

    if args["load_defaults_path"]:
        default_args = args_from_file(args["load_defaults_path"])
        # get the args which have been set
        args = {key: value for key, value in vars(args).items() if value}
        # apply them ontop the new default args
        args = {**default_args, **args}

    LOGGER.debug("Args: %s", args)
    if not args["destination"].parent.is_dir():
        LOGGER.debug("Creating destination parent dir.")
        args["destination"].parent.mkdir(parents=True)

    raw_data_files = sorted(args["raw_data_dir"].glob("*.h5"))
    if args["verbose"] > 0:
        LOGGER.info("Raw data dir: %s", args["raw_data_dir"])
        for path in raw_data_files[:5]:
            LOGGER.info(path.name)

    if args["blm_list_file"] is not None:
        blm_list = get_blm_list_from_file(args["blm_list_file"])
    elif args["blm_filter"] is not None:
        blm_list = get_blm_list_from_filter(raw_data_files[0], args["blm_filter"])
    else:
        blm_list = None

    if args["verbose"] > 0 and blm_list is not None:
        for blm in blm_list[:5]:
            LOGGER.info(blm)

    preproc = type_preprocessor(args["preprocessor"])(
        blm_list=blm_list,
        particle_type=args["particle_type"],
        **args["preprocessor_kwargs"],
    )
    LOGGER.info("Preprocessor: %s", preproc)
    spooler = type_spooler(args["spooler"])(
        preproc, raw_data_files, args["destination"]
    )
    LOGGER.info("Spooler: %s", spooler)
    LOGGER.info("Spooling")
    spooler.spool(**args["h5_kwargs"])
    if args["concat_path"] is not None:
        if hasattr(spooler, "concat"):
            LOGGER.info("Running concat: %s", args["concat_path"])
            spooler.concat(args["concat_path"], **args["h5_kwargs"])  # type: ignore
        else:
            LOGGER.warning("Spooler does not implement concat.")

    if args["copy_dataset_info"]:
        copy_dataset_info(args["raw_data_dir"], args["destination"])
    args_to_file(args)
