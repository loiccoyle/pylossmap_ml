import logging
from pathlib import Path
from typing import List
import json

import pandas as pd
from pylossmap import BLMDataFetcher
from tqdm.auto import tqdm

from ..db import DB

LOGGER = logging.getLogger(__name__)


def save_dataset_info(file_path: Path, **kwargs) -> None:
    """Save the dataset fetch parameters to file.

    Args:
        file_path: the file in which to write the fetch parameters.
        **kwargs: the fetch parameters which will be written to file.
    """
    LOGGER.info("Saving dataset info to: %s", file_path)
    with open(file_path, "w") as file:
        json.dump(kwargs, file, indent=2)


def fetch(
    t1: pd.Timestamp,
    t2: pd.Timestamp,
    destination_dir: Path,
    beam_modes_fills: List[str] = ["STABLE"],
    beam_modes_fetch: List[str] = ["STABLE"],
    BLM_var: str = "LHC.BLMI:LOSS_RS01",
) -> None:
    """Fetch a BLM dataset.

    Args:
        t1: start timestamp.
        t2: stop timestamp.
        destination_dir: folder in which to write the data.
        beam_modes_fills: fetch data for fills which reach these beam modes.
        beam_modes_fetch: for the fills that pass, fetch data for this beam mode.
        BLM_var: the timber BLM var to fetch.
    """
    destination_dir = Path(destination_dir)
    if not destination_dir.is_dir():
        destination_dir.mkdir(parents=True)
    else:
        LOGGER.warning("'%s' already exists.", destination_dir.resolve())

    fetcher = BLMDataFetcher(pbar=False, BLM_var=BLM_var)

    fills = DB.getLHCFillsByTime(t1, t2, beam_modes=beam_modes_fills, unixtime=True)
    LOGGER.info("Found %i fills.", len(fills))
    save_dataset_info(
        destination_dir / ".dataset_info.json",
        t1=str(t1),
        t2=str(t2),
        destination_dir=str(destination_dir),
        beam_modes_fills=beam_modes_fills,
        beam_modes_fetch=beam_modes_fetch,
        BLM_var=BLM_var,
    )

    for fill in tqdm(fills, desc="Fetching data"):
        save_path = (destination_dir / str(fill["fillNumber"])).with_suffix(".h5")
        LOGGER.debug("save path: %s", save_path)
        if save_path.exists():
            LOGGER.info("save path already exists, skipping.")
            continue
        try:
            out = fetcher.from_fill(fill["fillNumber"], beam_modes=beam_modes_fetch)
            out.save(save_path)
            del out
        except Exception:
            LOGGER.error(
                "Failed to fetch data for fill: %s",
                fill["fillNumber"],
                exc_info=True,
                stack_info=True,
            )


def fetch_ion_stable(destination_dir: Path, **kwargs) -> None:
    t1 = pd.to_datetime("2018-09-01 00:00:00").tz_localize("Europe/Zurich")
    t2 = pd.to_datetime("2018-12-31 00:00:00").tz_localize("Europe/Zurich")
    return fetch(
        t1,
        t2,
        destination_dir,
        beam_modes_fills=["STABLE"],
        beam_modes_fetch=["STABLE"],
        **kwargs
    )


def fetch_proton_stable(destination_dir: Path, **kwargs) -> None:
    t1 = pd.to_datetime("2018-05-01 00:00:00").tz_localize("Europe/Zurich")
    t2 = pd.to_datetime("2018-08-31 00:00:00").tz_localize("Europe/Zurich")
    return fetch(
        t1,
        t2,
        destination_dir,
        beam_modes_fills=["STABLE"],
        beam_modes_fetch=["STABLE"],
        **kwargs
    )


def fetch_proton_preramp(destination_dir: Path, **kwargs) -> None:
    t1 = pd.to_datetime("2018-05-01 00:00:00").tz_localize("Europe/Zurich")
    t2 = pd.to_datetime("2018-08-31 00:00:00").tz_localize("Europe/Zurich")
    return fetch(
        t1,
        t2,
        destination_dir,
        beam_modes_fills=["STABLE"],
        beam_modes_fetch=["PRERAMP"],
        **kwargs
    )
