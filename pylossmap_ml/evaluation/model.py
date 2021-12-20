import json
import logging
import random
from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import load_model
from pylossmap import BLMData
from tqdm.auto import tqdm

from ..db import DB
from ..training.generator import DataGenerator

UFO_METADATA_FILE = Path(
    "/nfs/cs-ccr-adtobsnfs/lhc_adtobsbox_data/lcoyle/lossmap_data/lossmap_data/ufo_metadata/2018/ufo_metadata.h5"
)

logger = logging.getLogger(__name__)


class AnomalyDetectionModel:
    @classmethod
    def load(cls, save_path: Union[Path, str]) -> "AnomalyDetectionModel":
        """Load an instance from disk.

        Args:
            save_path: The path to the folder where the data was saved.
        """
        if not isinstance(save_path, Path):
            save_path = Path(save_path)
        logger.info("Loading kwargs.")
        with open(save_path / "evaluation_kwargs.json", "r") as fp:
            kwargs = json.load(fp)
        if kwargs["model_path"] is not None:
            kwargs["model_path"] = Path(kwargs["model_path"])
        if kwargs["raw_data_path"] is not None:
            kwargs["raw_data_path"] = Path(kwargs["raw_data_path"])

        out = cls(**kwargs)
        logger.info("Loading metadata train.")
        out._metadata_train = pd.read_hdf(save_path / "metadata_train.h5", "data")
        logger.info("Loading metadata val.")
        out._metadata_val = pd.read_hdf(save_path / "metadata_val.h5", "data")
        logger.info("Loading error train.")
        out._error_train = np.load(save_path / "error_train.npy")
        logger.info("Loading error val.")
        out._error_val = np.load(save_path / "error_val.npy")
        return out

    def __init__(
        self,
        model_path: Path,
        raw_data_path: Optional[Path] = None,
        threshold: Optional[float] = None,
    ) -> None:
        self.model_path = model_path
        self.raw_data_path = raw_data_path
        self.threshold = threshold
        self._model = None
        self._train_kwargs = None
        self._history = None
        self._model_kwargs = None
        self._generator_train = None
        self._generator_val = None
        self._metadata_train = None
        self._metadata_val = None
        self._error_train = None
        self._error_val = None

    @property
    def model(self):
        """The model property."""
        if self._model is None:
            logger.info("Loading model.")
            self._model = load_model(self.model_path)
        return self._model

    @property
    def train_kwargs(self) -> dict:
        if self._train_kwargs is None:
            logger.info("Loading training kwargs.")
            with (self.model_path / "train_kwargs.json").open("r") as fp:
                self._train_kwargs = json.load(fp)
        return self._train_kwargs

    @property
    def history(self) -> dict:
        if self._history is None:
            logger.info("Loading history.")
            with (self.model_path / "history.json").open("r") as fp:
                self._history = json.load(fp)
        return self._history

    @property
    def model_kwargs(self) -> dict:
        if self._model_kwargs is None:
            logger.info("Loading model kwargs.")
            with (self.model_path / "model_kwargs.json").open("r") as fp:
                self._model_kwargs = json.load(fp)
        return self._model_kwargs

    @property
    def generator_train(self) -> DataGenerator:
        if self._generator_train is None:
            logger.info("Loading train generator.")
            self._generator_train = DataGenerator.from_json(
                self.model_path / "generator_train.json"
            )
        return self._generator_train

    @property
    def generator_val(self) -> DataGenerator:
        if self._generator_val is None:
            logger.info("Loading validation generator.")
            self._generator_val = DataGenerator.from_json(
                self.model_path / "generator_val.json"
            )
        return self._generator_val

    @property
    def metadata_train(self) -> pd.DataFrame:
        if self._metadata_train is None:
            logger.info("Loading train metadata.")
            self._metadata_train = pd.DataFrame.from_records(
                self.generator_train.get_metadata(),
                columns=["fill_number", "beam_mode", "timestamp"],
            )
            self._metadata_train["dataset"] = "train"

            if self._error_train is not None:
                self._metadata_train["MSE"] = self._error_train
                self._metadata_train["rank"] = np.argsort(self.error_train)

        return self._metadata_train

    @property
    def metadata_val(self) -> pd.DataFrame:
        if self._metadata_val is None:
            logger.info("Loading validation metadata.")
            self._metadata_val = pd.DataFrame.from_records(
                self.generator_val.get_metadata(),
                columns=["fill_number", "beam_mode", "timestamp"],
            )
            self._metadata_val["dataset"] = "val"

            if self._error_val is not None:
                self._metadata_val["rank"] = np.argsort(self.error_val)
                self._metadata_val["MSE"] = self._error_val

        return self._metadata_val

    @property
    def metadata(self) -> pd.DataFrame:
        return pd.concat([self.metadata_train, self.metadata_val])

    def _chunk_predict_MSE(self, generator: DataGenerator) -> np.ndarray:
        """Iteratively compute the models prediction on the generator."""
        MSE_chunks = []
        for chunk, _ in tqdm(generator):
            chunk_pred = self.model.predict(chunk)
            chunk_MSE = ((chunk_pred - chunk) ** 2).mean(axis=1)
            MSE_chunks.append(chunk_MSE)
        return np.vstack(MSE_chunks).squeeze()

    def _chunk_predict_MAE(self, generator: DataGenerator) -> np.ndarray:
        MAE_chunks = []
        for chunk, _ in tqdm(generator):
            chunk_pred = self.model.predict(chunk)
            chunk_MAE = (np.abs(chunk_pred - chunk)).mean(axis=1)
            MAE_chunks.append(chunk_MAE)
        return np.vstack(MAE_chunks).squeeze()

    @property
    def error_train(self) -> np.ndarray:
        if self._error_train is None:
            logger.info("Computing error train.")
            self._error_train = self._chunk_predict_MSE(self.generator_train)

            if self._metadata_train is not None:
                self._metadata_train["MSE"] = self._error_train
                self._metadata_train["rank"] = np.argsort(self.error_train)

        return self._error_train

    @property
    def error_val(self) -> np.ndarray:
        if self._error_val is None:
            logger.info("Computing error validation.")
            self._error_val = self._chunk_predict_MSE(self.generator_val)

            if self._metadata_val is not None:
                self._metadata_val["MSE"] = self._error_val
                self._metadata_val["rank"] = np.argsort(self.error_val)

        return self._error_val

    @property
    def anomalies(self) -> pd.DataFrame:
        if self.threshold is None:
            raise ValueError("'threshold' is not set.")
        if self._error_train is None:
            assert self.error_train is not None
        if self._error_val is None:
            assert self.error_val is not None
        anomalies = self.metadata[self.metadata["MSE"] >= self.threshold]
        anomalies.sort_values("MSE", ascending=False, inplace=True)
        return anomalies

    @property
    def error(self) -> np.ndarray:
        return np.hstack([self.error_train, self.error_val])

    def drop_fill(self, fill_number: int) -> None:
        self._metadata_val = self.metadata_val[
            self.metadata_val["fill_number"] != fill_number
        ]
        self._metadata_train = self.metadata_train[
            self.metadata_train["fill_number"] != fill_number
        ]
        self._error_val = self.metadata_val["MSE"]
        self._error_train = self.metadata_train["MSE"]

    def add_fill_beammode_timings(
        self, anomalies: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Add the beam mode start and end timings for each fill.

        Args:
            anomalies: The anomaly `pd.DataFrame`.

        Returns:
            The anomaly dataframe with beam mode timing info added in the
                'beam_mode_start', 'beam_mode_end' and 'timestamp_rel_bm' columns.

        """
        if anomalies is None:
            anomalies = self.anomalies.copy()

        beam_mode = anomalies["beam_mode"].iloc[0]
        logger.info("Using beam mode: %s", beam_mode)
        for fill_number in tqdm(anomalies["fill_number"].unique()):
            fill_info = DB.getLHCFillData(int(fill_number))
            logger.debug("Fill number: %i", fill_number)
            logger.debug("Fill info: %s", fill_info)
            for mode in fill_info["beamModes"]:
                if mode["mode"] == beam_mode:
                    mode_start = pd.to_datetime(
                        mode["startTime"], unit="s", utc=True
                    ).tz_convert("Europe/Zurich")
                    mode_end = pd.to_datetime(
                        mode["endTime"], unit="s", utc=True
                    ).tz_convert("Europe/Zurich")

                    anomalies.loc[
                        anomalies["fill_number"] == fill_number, "beam_mode_start"
                    ] = mode_start
                    anomalies.loc[
                        anomalies["fill_number"] == fill_number, "beam_mode_end"
                    ] = mode_end
                    break

        anomalies["beam_mode_start"] = pd.to_datetime(anomalies["beam_mode_start"])
        anomalies["beam_mode_end"] = pd.to_datetime(anomalies["beam_mode_end"])
        anomalies["timestamp_rel_bm"] = (
            anomalies["timestamp"] - anomalies["beam_mode_start"]
        ) / (anomalies["beam_mode_end"] - anomalies["beam_mode_start"])
        return anomalies

    def add_ufo_check(
        self,
        anomalies: Optional[pd.DataFrame] = None,
        ufo_metadata: Optional[pd.DataFrame] = None,
        dt: str = "1s",
        **kwargs,
    ) -> pd.DataFrame:
        """Add presence in UFO metadata information to anomaly DataFrame.

        Args:
            anomalies: The anomaly `pd.DataFrame`.
            ufo_metadata: The ufo metadat `pd.DataFrame`.
            dt: `pd.Timedelta` string describing the delta t around the timestamp
                to consider as being a matching UFO.

        Returns:
            The anomaly DataFrame with UFO info added.
        """
        if anomalies is None:
            anomalies = self.anomalies.copy()

        if ufo_metadata is None:
            ufo_metadata = self.load_ufo_metadata(**kwargs)

        def ufo_check_func(row: pd.Series) -> bool:
            return (
                len(
                    self.check_ufo_around(
                        row.timestamp, dt=dt, ufo_metadata=ufo_metadata
                    )
                )
                > 0
            )

        anomalies["in_ufo_db"] = anomalies.apply(ufo_check_func, axis=1)
        return anomalies

    def threshold_from_quantile(
        self, quantile: float = 0.99, datasets: Union[List[str], str] = ["train", "val"]
    ):
        """Compute an anomaly threshold for a given quantile.

        Args:
            quantile: quantile value.
            datasets: On which dataset(s) to compute the quantile, "train", "val".

        Returns:
            The threshold value.
        """
        if not isinstance(datasets, list):
            datasets = list(datasets)
        datasets = np.hstack([getattr(self, "error_" + dset) for dset in datasets])
        return np.quantile(datasets, quantile)

    def check_ufo_around(
        self,
        timestamp: pd.Timestamp,
        dt: str,
        ufo_metadata: Optional[pd.DataFrame],
        **kwargs,
    ) -> pd.DataFrame:
        """Check if `timestamp` is in the `ufo_metadata` DataFrame within `dt`.

        Args:
            timestamp: The timestamp to check.
            dt: The delta t allowed around `timestamp`.
            ufo_metadata: The UFO metadata DataFrame.

        Returns:
            The UFOs within `timestamp` ± `dt`.
        """
        if ufo_metadata is None:
            ufo_metadata = self.load_ufo_metadata(**kwargs)

        dt = pd.Timedelta(dt)

        ufos = ufo_metadata[
            (ufo_metadata["datetime"] > (timestamp - dt))
            & (ufo_metadata["datetime"] < (timestamp + dt))
        ]
        return ufos

    def load_ufo_metadata(
        self,
        ufo_metadata_path: Path = UFO_METADATA_FILE,
        beam_mode: str = "STABLE",
        particle: str = "protons",
        only_straight_section: bool = False,
    ) -> pd.DataFrame:
        """Load the UFO metadata and apply relevant filtering.

        Args:
            ufo_metadata_path: the path to the hdf file containing the UFO metadata.
            beam_mode: Filter based on beam mode.
            particle: filter based on particle type.
            only_straight_section: Only keep UFOs which occur in the straight sections.

        Returns:
            The UFO metadata DataFrame.
        """
        ufo_metadata = pd.read_hdf(ufo_metadata_path)
        ufo_metadata = ufo_metadata[ufo_metadata["beam_mode"] == beam_mode]
        ufo_metadata = ufo_metadata[ufo_metadata["particle_b1"] == particle]
        ufo_metadata = ufo_metadata[ufo_metadata["particle_b2"] == particle]
        ufo_metadata = ufo_metadata[
            ufo_metadata["fill"].isin(self.metadata["fill_number"].unique())
        ]
        try:
            ufo_metadata["datetime"] = ufo_metadata["datetime"].dt.tz_localize(
                "Europe/Zurich"
            )
        except TypeError:
            logger.info("Ufo metadata already contains timezone info.")
            pass

        if only_straight_section:
            if self.raw_data_path is None:
                raise ValueError(
                    "'raw_data_path' required for 'only_straight_section' filtering."
                )
            random_raw_data_file = random.choice(list(self.raw_data_path.glob("*.h5")))
            blm_data = BLMData.load(random_raw_data_file)
            blm_data = blm_data.cell(*range(12, 33 + 1))
            arc_blms = blm_data.df.columns.to_list()
            arc_blms_in_ufo = list(set(arc_blms) & set(ufo_metadata.index.unique()))
            ufo_metadata = ufo_metadata.loc[arc_blms_in_ufo]
        return ufo_metadata

    def plot_error(
        self, n_bins: int = 100, threshold: Optional[float] = None
    ) -> Tuple[plt.Figure, plt.Axes, np.ndarray]:
        if threshold is None:
            threshold = self.threshold
        fig, ax = plt.subplots(1, 1, figsize=(9, 6))
        _, bins, _ = ax.hist(self.error_train, bins=n_bins, label="Training dataset")
        ax.hist(self.error_val, bins=bins, label="Validation dataset")
        ax.set_yscale("log")
        if threshold is not None:
            ax.axvline(threshold)
        ax.legend()
        return fig, ax, bins

    def load_raw_data_fill(self, fill: int) -> BLMData:
        if self.raw_data_path is None:
            raise ValueError("No 'raw_data_path' provided.")
        return BLMData.load(self.raw_data_path / f"{fill}.h5")

    def plot_anomaly_fill_timings(
        self, anomalies: Optional[pd.DataFrame] = None, **kwargs
    ) -> Tuple[plt.Figure, List[plt.Axes]]:
        """Plot the distribution of the anomalies in the beam mode of each fill.

        Args:
            anomalies: the dataframe containing the anomalies.
            **kwargs: passed to `plt.scatter`.

        Returns:
            The plt.Figure and plt.Axes.
        """
        if anomalies is None:
            anomalies = self.anomalies
        if "timestamp_rel_bm" not in anomalies.columns:
            anomalies = self.add_fill_beammode_timings(anomalies)

        anomalies = anomalies.sort_values("fill_number")

        fig = plt.figure(
            constrained_layout=True, figsize=kwargs.pop("figsize", (12, 3))
        )
        gs = fig.add_gridspec(1, 4)
        axes = []
        axes.append(fig.add_subplot(gs[0, :-1]))
        axes.append(fig.add_subplot(gs[0, -1:], sharey=axes[-1]))

        axes[0].scatter(
            anomalies["fill_number"].apply(str), anomalies["timestamp_rel_bm"], **kwargs
        )
        axes[0].tick_params(axis="x", labelrotation=45)
        axes[0].set_ylabel("Position in the beam mode")
        axes[0].set_xlabel("Fill number")
        axes[1].hist(anomalies["timestamp_rel_bm"], orientation="horizontal")
        axes[1].set_xlabel("Count")
        return fig, axes

    def plot_history(self, **kwargs) -> plt.Axes:
        """Plot the training history.

        Args:
            kwargs: passed to `pd.DataFrame.plot`.

        Returns:
            The plotted `plt.Axes`.
        """
        return pd.DataFrame(self.history).plot(**kwargs)

    def plot_anomaly_fills(
        self, anomalies: Optional[pd.DataFrame] = None, **kwargs
    ) -> plt.Axes:
        """Plot the distribution of the number of events among fills in a pie plot.

        Args:
            anomalies: DataFrame of the anomalies.
            kwargs: passed to `pd.DataFrame.plot.pie`.

        Returns:
            The plotter `plt.Axes`.
        """

        if anomalies is None:
            anomalies = self.anomalies
        return (
            anomalies.groupby("fill_number")["fill_number"].count().plot.pie(**kwargs)
        )

    def plot_sample(self, fill: int, timestamp: pd.Timestamp, around: int = 1):
        """Plot the raw data of a timestamp.

        Args:
            fill: the fill in which the event occurreds.
            timestamp: the timestamp of the event.
            around: the number of surrounding events to show.
        """
        raw_data_fill = self.load_raw_data_fill(fill)
        raw_data_fill.df.drop_duplicates(inplace=True)

        for dt in range(-around, around + 1):
            loss_map = raw_data_fill.loss_map(timestamp + pd.Timedelta(f"{dt}s"))
            loss_map.plot(figsize=(14, 3), title=f"{loss_map.datetime} {dt}")

        # Plotting the diff
        pre_lm = raw_data_fill.loss_map(timestamp - pd.Timedelta("1s"))
        lm = raw_data_fill.loss_map(timestamp)
        lm.df["data"] = ((lm.df["data"] - pre_lm.df["data"]) / pre_lm.df["data"]).abs()
        print("Max diff:", lm.df.iloc[lm.df["data"].argmax()])
        _, ax = lm.plot(figsize=(14, 3), title=f"diff {lm.datetime}")
        ax.set_yscale("linear")
        ax.relim()
        ax.autoscale(axis="y")

    def save(self, save_path: Optional[Path] = None) -> None:
        """Save the attributes to disk.

        Args:
            save_path: The path to the folder where the data should be saved.
        """
        kwarg_attributes = [
            "model_path",
            "raw_data_path",
            "threshold",
        ]
        np_attributes = [
            "error_train",
            "error_val",
        ]
        df_attributes = [
            "metadata_train",
            "metadata_val",
        ]
        if save_path is None:
            save_path = Path.cwd() / self.model_path.name
            if not save_path.is_dir():
                save_path.mkdir(parents=True)

        save_dict = {
            attribute: getattr(self, attribute) for attribute in kwarg_attributes
        }
        save_dict = {
            key: str(value) if value is not None else value
            for key, value in save_dict.items()
        }
        logger.info("Saving kwargs.")
        with open(save_path / "evaluation_kwargs.json", "w") as fp:
            json.dump(save_dict, fp)

        for attribute in np_attributes:
            logger.info(f"Saving {attribute}.")
            np_array = getattr(self, attribute)
            np.save(save_path / f"{attribute}.npy", np_array)

        for attribute in df_attributes:
            logger.info(f"Saving {attribute}.")
            df = getattr(self, attribute)
            df.to_hdf(save_path / f"{attribute}.h5", "data")
