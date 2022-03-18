import logging
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pylossmap.lossmap import LossMap
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model

from . import metadata
from .preprocessing import (
    NON_UFO_LABEL_ARGMAX,
    UFO_LABEL_ARGMAX,
    DataGenerator,
    rolling_window,
)


class SupervisedModel:
    def __init__(self, model_path: Path, generator_path: Path, raw_data_dir: Path):
        self.model_path = model_path
        self.generator_path = generator_path
        self.raw_data_dir = raw_data_dir
        self.model = load_model(self.model_path)
        self.generator = DataGenerator.load(self.generator_path)
        self._pred = None
        self._log = logging.getLogger(__name__)

    @property
    def pred(self) -> np.ndarray:
        if self._pred is None:
            self._pred = self.model.predict(self.generator.data)
        return self._pred

    def confusion_matrix(self) -> np.ndarray:
        """Compute the confusion matrix."""
        return confusion_matrix(
            self.generator.labels.squeeze().argmax(axis=1),
            self.pred.squeeze().argmax(axis=1),
        )

    def select(
        self, label_class: int, pred_class: int
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        labels_am = self.generator.labels.squeeze().argmax(axis=1)
        pred_am = self.pred.squeeze().argmax(axis=1)

        label_class_mask = labels_am == label_class
        pred_class_mask = pred_am == pred_class
        class_mask = np.logical_and(label_class_mask, pred_class_mask)

        data_class = self.generator.data[class_mask]
        meta_class = self.generator.metadata.iloc[class_mask]
        return data_class, meta_class

    def pred_rolling_window(
        self, meta: pd.Series, window_size: Optional[int] = None
    ) -> Tuple[LossMap, np.ndarray]:
        """Load a loss map, roll a window across it, predict on every window.

        Args:
            meta: the loss map's metadata
            window_size: the size of the rolling window

        Returns:
            The loss map data and the model's prediction across the windows.
        """
        if window_size is None:
            window_size = self.generator.data.shape[1]
        raw_data = metadata.load_raw_fill(self.raw_data_dir / f"{meta.fill}.h5")
        lm_data = raw_data.loss_map(meta.datetime + pd.Timedelta("1s"))
        lm_data_np = lm_data["data"].df.to_numpy()
        lm_data_np = rolling_window(
            np.pad(lm_data_np, int((window_size - 1) / 2), mode="wrap"), window_size
        )
        self._log.debug("post rolling shape: %s", lm_data_np.shape)
        lm_data_np = self.generator._pre_norm(lm_data_np)
        lm_data_np = self.generator._norm_func(lm_data_np)
        lm_data_np = self.generator._expand(lm_data_np)
        return lm_data, self.model.predict(lm_data_np)

    def _plot_class(
        self, label_class: int, pred_class: int, n_sample: Optional[int] = None
    ):
        data_class, meta_class = self.select(label_class, pred_class)
        if n_sample is not None:
            sample_indices = np.random.choice(
                np.arange(len(data_class)), n_sample, replace=False
            )
            data_class = data_class[sample_indices]
            meta_class = meta_class.iloc[sample_indices]

        for data, (_, meta) in zip(data_class, meta_class.iterrows()):
            fig, axes = plt.subplots(1, 1)
            axes.plot(data)
            axes.set_title("Processed data")
            raw_fill_data = metadata.load_raw_fill(
                self.raw_data_dir / f"{meta.fill}.h5"
            )
            ufo_data = raw_fill_data.loss_map(meta.datetime + pd.Timedelta("1s"))
            metadata.plot_ufo_box(ufo_data, meta.dcum)
            plt.show()

    def plot_ufos(self, n_sample: Optional[int] = None):
        self._plot_class(UFO_LABEL_ARGMAX, UFO_LABEL_ARGMAX, n_sample=n_sample)

    def plot_non_ufos(self, n_sample: Optional[int] = None):
        self._plot_class(NON_UFO_LABEL_ARGMAX, NON_UFO_LABEL_ARGMAX, n_sample=n_sample)

    def plot_false_positive(self, n_sample: Optional[int] = None):
        self._plot_class(NON_UFO_LABEL_ARGMAX, UFO_LABEL_ARGMAX, n_sample=n_sample)

    def plot_false_negative(self, n_sample: Optional[int] = None):
        self._plot_class(UFO_LABEL_ARGMAX, NON_UFO_LABEL_ARGMAX, n_sample=n_sample)

    def plot_rolling_window(
        self, lossmap: LossMap, rolling_pred: np.ndarray
    ) -> Tuple[plt.Figure, plt.Axes]:
        blm_dcum = lossmap.df["dcum"]
        #     plot_labels = (lm_pred.squeeze().argmax(axis=1) * 0.01) + 0.001
        ufo_certainty = rolling_pred.squeeze()[:, 0]
        non_ufo_certainty = rolling_pred.squeeze()[:, 1]

        fig, ax = plt.subplots(2, figsize=(16, 5), sharex=True)
        #     plt.figure(figsize=(16, 4))
        lossmap.normalize().plot(ax=ax[0])
        ax[0].set_ylim(1e-4, 10)
        ax[0].set_ylabel("Normalized losses [a.u.]")
        #     ax[0].plot(blm_dcum, lm_data.df["data"].to_numpy())
        #     ax[0].set_yscale("log")
        ax[1].plot(blm_dcum, ufo_certainty, label="ufo certainty")
        #     ax[1].plot(blm_dcum, non_ufo_certainty, label="non ufo certainty")
        ax[1].legend()
        ax[1].set_xlabel("dcum [cm]")
        ax[1].set_ylabel("UFO probablity")

        return fig, ax
