import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import load_model
from pylossmap import BLMData
from tqdm.auto import tqdm

from ..training.generator import DataGenerator

logger = logging.getLogger(__name__)


class AnomalyDetectionModel:
    @classmethod
    def load(cls, save_path: Path) -> "AnomalyDetectionModel":
        with open(save_path / "evaluation_kwargs.json", "r") as fp:
            kwargs = json.load(fp)

        out = cls(**kwargs)
        out._metadata_train = pd.read_hdf(save_path / "metadata_train.h5", "data")
        out._metadata_val = pd.read_hdf(save_path / "metadata_val.h5", "data")
        out._mse_train = np.load(save_path / "mse_train.npy")
        out._mse_val = np.load(save_path / "mse_val.npy")
        return out

    def __init__(self, model_path: Path, raw_data_path: Optional[Path] = None) -> None:
        self.model_path = model_path
        self.raw_data_path = raw_data_path
        self._model = None
        self._train_kwargs = None
        self._history = None
        self._model_kwargs = None
        self._generator_train = None
        self._generator_val = None
        self._metadata_train = None
        self._metadata_val = None
        self._mse_train = None
        self._mse_val = None

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
            self._metadata_train = self.generator_train.get_metadata()
        return self._metadata_train

    @property
    def metadata_val(self) -> pd.DataFrame:
        if self._metadata_val is None:
            logger.info("Loading validation metadata.")
            self._metadata_val = self.generator_val.get_metadata()
        return self._metadata_val

    def _chunk_predict_MSE(self, generator) -> np.ndarray:
        """Iteratively compute the models prediction on the generator."""
        MSE_chunks = []
        for chunk, _ in tqdm(generator):
            chunk_pred = self.model.predict(chunk)
            chunk_MSE = ((chunk_pred - chunk) ** 2).mean(axis=1)
            MSE_chunks.append(chunk_MSE)
        return np.vstack(MSE_chunks).squeeze()

    @property
    def mse_train(self) -> np.ndarray:
        if self._mse_train is None:
            logger.info("Computing MSE train.")
            self._mse_train = self._chunk_predict_MSE(self.generator_train)
        return self._mse_train

    @property
    def mse_val(self) -> np.ndarray:
        if self._mse_val is None:
            logger.info("Computing MSE validation.")
            self._mse_val = self._chunk_predict_MSE(self.generator_val)
        return self._mse_val

    def plot_mse(self, n_bins: int = 100) -> Tuple[plt.Figure, plt.Axes]:
        fig, ax = plt.subplots(1, 1, figsize=(9, 6))
        _, bins, _ = ax.hist(self.mse_train, bins=n_bins)
        ax.hist(self.mse_val, bins=bins)
        ax.set_yscale("log")
        return fig, ax

    def load_raw_data_fill(self, fill: int) -> BLMData:
        if self.raw_data_path is None:
            raise ValueError("No 'raw_data_path' provided.")
        return BLMData.load(self.raw_data_path / f"{fill}.h5")

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
        kwarg_attributes = [
            "model_path",
            "raw_data_path",
        ]
        df_attributes = [
            "metadata_train",
            "metadata_val",
        ]
        np_attributes = [
            "mse_train",
            "mse_val",
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
        with open(save_path / "evaluation_kwargs.json", "w") as fp:
            json.dump(save_dict, fp)

        for attribute in df_attributes:
            df = getattr(self, attribute)
            df.to_hdf(save_path / f"{attribute}.h5", "data")

        for attribute in np_attributes:
            np_array = getattr(self, attribute)
            np.save(save_path / f"{attribute}.npy", np_array)