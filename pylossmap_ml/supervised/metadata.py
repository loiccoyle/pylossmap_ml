from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from pylossmap import BLMData
from pylossmap.lossmap import LossMap
from tqdm.auto import tqdm


def ufo_stable_proton(ufo_meta: pd.DataFrame) -> pd.DataFrame:
    ufo_meta = ufo_meta[ufo_meta["beam_mode"] == "STABLE"]
    ufo_meta = ufo_meta[ufo_meta["particle_b1"] == "protons"]
    ufo_meta = ufo_meta[ufo_meta["particle_b2"] == "protons"]
    return ufo_meta


def ufo_on_blms(ufo_meta: pd.DataFrame, blms: List[str]) -> pd.DataFrame:
    """Only keep ufos which occur on the provided blms.

    Args:
        ufo_meta: metadata of the ufo events
        blms: keep ufos which occur on these blms

    Returns:
        The filtered ufo metadata.
    """
    blms_in_ufo = list(set(blms) & set(ufo_meta["blm"].unique()))
    ufo_meta = ufo_meta.set_index("blm").loc[blms_in_ufo].reset_index()
    return ufo_meta


def load_raw_fill(file_path: Path) -> BLMData:
    """Load the raw blm data.

    Args:
        file_path: the path to the hdf file

    Returns:
        The raw BLM data.
    """
    blm_data = BLMData.load(file_path)
    blm_data.df.drop_duplicates(inplace=True)
    return blm_data


def get_ufo_data(ufo_meta: pd.DataFrame, raw_data_dir: Path) -> pd.DataFrame:
    """Load the ufo event blm data.

    Args:
        ufo_meta: metadata of the ufo events
        raw_data_dir: directory containing the raw blm data

    Returns:
        The raw blm data.
    """
    ufo_blm_data = []
    for idx, row in tqdm(ufo_meta.reset_index().iterrows(), total=len(ufo_meta)):
        #     try:
        print(idx, row.datetime, row.blm, row.fill)
        try:
            blm_data = load_raw_fill(raw_data_dir / f"{row.fill}.h5")
        except FileNotFoundError:
            print(f"file not found {row.fill}")
            continue
        loss_map = blm_data.loss_map(row.datetime + pd.Timedelta("1s"))
        ufo_blm_data.append(loss_map.df["data"])
    return pd.DataFrame(ufo_blm_data).reset_index(drop=True)


def ufo_above_threshold(
    ufo_meta: pd.DataFrame,
    raw_data_dir: Path,
    dcum_range: int = 3000,
    blm_threshold: float = 1e-3,
    n_above: int = 2,
) -> pd.DataFrame:
    """Keep ufo event which have `n_above` blms within `dcum_range` above `blm_threshold`.

    Args:
        ufo_meta: metadata of the ufos
        raw_data_dir: directory containing the raw blm data
        dcum_range: +- distance wwithin which to look for neighboring high blms
        blm_threshold: blm amplitude threshold
        n_above: how many neighboring blms should be above `blm_threshold`

    Returns:
        The metadata of the ufo events which pass the conditions.
    """
    keep_ufo_idx = []
    for fill, fill_grp in tqdm(
        ufo_meta.reset_index().sort_values("datetime").groupby("fill")
    ):
        try:
            blm_data_fill = load_raw_fill(raw_data_dir / f"{fill}.h5")
        except FileNotFoundError:
            print(f"file not found {fill}")
            continue
        for idx, ufo in fill_grp.iterrows():
            #         print(idx)
            blm_data_lm = blm_data_fill.loss_map(ufo.datetime + pd.Timedelta("1s"))
            blm_data_around = blm_data_lm.df[
                (blm_data_lm.df["dcum"] >= ufo.dcum - dcum_range)
                & (blm_data_lm.df["dcum"] <= ufo.dcum + dcum_range)
            ]
            if (blm_data_around["data"] >= blm_threshold).sum() >= n_above:
                keep_ufo_idx.append(idx)
            else:
                print(f"{idx} does not pass threshold check.")
    return ufo_meta.iloc[keep_ufo_idx]


def plot_ufos(ufo_meta: pd.DataFrame, raw_data_dir: Path):
    """Plot the ufos for all the ufo events in `ufo_meta`.

    Args:
        ufo_meta: metadata of the ufos
        raw_data_dir: directory containing the raw blm data
    """
    for idx, row in tqdm(ufo_meta.reset_index().iterrows(), total=len(ufo_meta)):
        try:
            ufo_fill_data = load_raw_fill(raw_data_dir / f"{row.fill}.h5")
            ufo_data = ufo_fill_data.loss_map(row.datetime + pd.Timedelta("1s"))
            plot_ufo_box(ufo_data, row.dcum)
            plt.show()
        except FileNotFoundError:
            print(f"File not found. Skipping index {idx}.")
            continue


def plot_ufo_box(
    loss_map: LossMap,
    ufo_dcum: int,
    dcum_width: int = 20000,
    zoom_width: int = 100000,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the loss map containing a ufo, and a zoomed in loss map around the ufo.

    Args:
        ufo_dcum: the position of the ufo
        dcum_width: the with of the vertical surrounding the ufo
        zoom_width: the dcum range to show on the zoomed in plot

    Returns:
        The figure and axes
    """
    fig, axes = plt.subplots(2, figsize=(15, 4))
    _, ax0 = loss_map.plot(ax=axes[0], title=str(loss_map.datetime))
    ax0.axvline(ufo_dcum - dcum_width / 2, color="purple", linewidth=1)
    ax0.axvline(ufo_dcum + dcum_width / 2, color="purple", linewidth=1)

    _, ax1 = loss_map.plot(ax=axes[1], title=str(loss_map.datetime))
    ax1.set_xlim([ufo_dcum - zoom_width, ufo_dcum + zoom_width])
    return fig, axes


def plot_checks(ufo_meta: pd.DataFrame) -> List[plt.Axes]:
    """Plot a bunch of differents plots, to double check the distribution of the ufos.

    Args:
        ufo_meta: metadata of the ufos

    Returns:
        A list of all the plotted `plt.Axes`.
    """
    figs = []
    figs.append(ufo_meta.groupby("fill")["fill"].count().plot.bar(figsize=(16, 6)))
    plt.show()
    figs.append(ufo_meta["dcum"].plot.hist(bins=200, figsize=(16, 6)))
    plt.show()
    figs.append(
        ufo_meta.groupby(ufo_meta["datetime"].dt.week)["datetime"]
        .count()
        .plot.bar(figsize=(10, 6))
    )
    plt.show()
    figs.append(
        ufo_meta.reset_index().groupby("blm")["blm"].count().plot.bar(figsize=(16, 6))
    )
    plt.show()
    figs.append(ufo_meta.groupby("fill")["fill"].count().plot.pie())
    plt.show()
    return figs
