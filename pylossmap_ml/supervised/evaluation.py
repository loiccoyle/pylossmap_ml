from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model

from . import metadata
from .preprocessing import NON_UFO_LABEL_ARGMAX, UFO_LABEL_ARGMAX, DataGenerator


class SupervisedModel:
    def __init__(self, model_path: Path, generator_path: Path, raw_data_dir: Path):
        self.model_path = model_path
        self.generator_path = generator_path
        self.raw_data_dir = raw_data_dir
        self.model = load_model(self.model_path)
        self.generator = DataGenerator.load(self.generator_path)
        self._pred = None

    @property
    def pred(self) -> np.ndarray:
        if self._pred is None:
            self._pred = self.model.predict(self.generator.data)
        return self._pred

    def 

    def confusion_matrix(self) -> np.ndarray:
        """Compute the confusion matrix."""
        return confusion_matrix(
            self.generator.labels.squeeze().argmax(axis=1),
            self.pred.squeeze().argmax(axis=1),
        )

    def _plot_class(
        self, label_class: int, pred_class: int, n_sample: Optional[int] = None
    ):
        labels_am = self.generator.labels.squeeze().argmax(axis=1)
        pred_am = self.pred.squeeze().argmax(axis=1)

        label_class_mask = labels_am == label_class
        pred_class_mask = pred_am == pred_class
        class_mask = np.logical_and(label_class_mask, pred_class_mask)

        data_class = self.generator.data[class_mask]
        meta_class = self.generator.metadata.iloc[class_mask]
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
        return data_class, meta_class

    def plot_ufos(self, n_sample: Optional[int] = None):
        return self._plot_class(UFO_LABEL_ARGMAX, UFO_LABEL_ARGMAX, n_sample=n_sample)

    def plot_non_ufos(self, n_sample: Optional[int] = None):
        return self._plot_class(
            NON_UFO_LABEL_ARGMAX, NON_UFO_LABEL_ARGMAX, n_sample=n_sample
        )

    def plot_false_positive(self, n_sample: Optional[int] = None):
        return self._plot_class(
            NON_UFO_LABEL_ARGMAX, UFO_LABEL_ARGMAX, n_sample=n_sample
        )

    def plot_false_negative(self, n_sample: Optional[int] = None):
        return self._plot_class(
            UFO_LABEL_ARGMAX, NON_UFO_LABEL_ARGMAX, n_sample=n_sample
        )
