import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from pylossmap import BLMData

# from tensorflow.keras.utils import Sequence
from tqdm.auto import tqdm

UFO_LABEL = [1, 0]
UFO_LABEL_ARGMAX = 0
NON_UFO_LABEL = [0, 1]
NON_UFO_LABEL_ARGMAX = 1


def augment_mirror(data: np.ndarray) -> np.ndarray:
    """Augment the data with the mirrored data.

    Args:
        data: data to augment

    Returns:
        the data with the mirrored data appended to the data.
    """
    return np.vstack([data, data[:, ::-1]])


def create_labels(
    ufo: np.ndarray, non_ufo: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Create the label arrays.

    Args:
        ufo: ufo data
        non_ufo: non ufo data

    Returns:
        The labels of the ufo and non ufo data.
    """
    ufo_labels = np.array([UFO_LABEL] * len(ufo))[:, None]
    non_ufo_labels = np.array([NON_UFO_LABEL] * len(non_ufo))[:, None]
    return ufo_labels, non_ufo_labels


def truncate_data(data: List[pd.DataFrame], target_length: int) -> np.ndarray:
    """Truncate the rows to a given length, centered.

    Args:
        data: iterable containing vector data to truncate
        target_length: the desired length of the vector conatining the blm signals

    Returns:
        Array containing the truncated data.
    """
    truncated_rows = []
    for row in data:
        length = row.shape[1]
        half_delta = (length - target_length) / 2
        start_shift = int(np.floor(half_delta))
        end_cutoff = int(np.ceil(half_delta))
        row_chunk = row.iloc[0, start_shift:-end_cutoff]
        truncated_rows.append(row_chunk.to_numpy())

    truncated_rows = np.array(truncated_rows)

    return truncated_rows


def create_peak_dataset(
    ufo_meta: pd.DataFrame,
    raw_data_dir: Path,
    dcum_around: int = 24000,
    target_length: int = 33,
    prior_dt: int = 3,
    post_dt: int = 3,
    non_ufo_threshold: float = 1e-3,
    include_meta: bool = True,
) -> Dict[str, np.ndarray]:
    """Create a ufo and non ufo peak dataset.

    Args:
        ufo_meta: metadata of the ufo events
        raw_data_dir: directory containing the raw data
        dcum_around: dcum range around the ufo
        target_length: the desired length of the vector conatining the blm signals
        prior_dt: how many seconds back to get the prior events
        post_dt: how many seconds forward to get the post events
        non_ufo_threshold: don't include non ufo samples when the max is above threshold
        include_meta: include the metadata of the samples in the returned dictionary

    Returns:
        Dictionary containing the ufo and non ufo data and metadata.
    """
    non_ufo_prior = []
    non_ufo_prior_meta = []
    peaks = []
    peaks_meta = []
    non_ufo_post = []
    non_ufo_post_meta = []
    for idx, ufo in tqdm(ufo_meta.iterrows(), total=len(ufo_meta)):
        raw_fill_data = BLMData.load(raw_data_dir / f"{ufo.fill}.h5")
        raw_fill_data.df = raw_fill_data.df.droplevel("mode")
        raw_fill_data.df = raw_fill_data.df.iloc[~raw_fill_data.df.index.duplicated()]

        raw_idx = raw_fill_data.df.index.get_loc(ufo.datetime, method="nearest") + 1

        around_blms = raw_fill_data.meta[
            (raw_fill_data.meta["dcum"] < ufo.dcum + dcum_around)
            & (raw_fill_data.meta["dcum"] > ufo.dcum - dcum_around)
        ]

        around_data = raw_fill_data.df[around_blms.index].iloc[raw_idx : raw_idx + 1]
        if around_data.shape[1] < target_length:
            print("skipping sample, not enough blms.")
            continue
        peaks.append(around_data)
        if include_meta:
            peaks_meta.append(ufo)

        around_prior_data = raw_fill_data.df[around_blms.index].iloc[
            raw_idx - prior_dt : raw_idx + 1 - prior_dt
        ]
        around_post_data = raw_fill_data.df[around_blms.index].iloc[
            raw_idx + post_dt : raw_idx + 1 + post_dt
        ]

        print("===============")
        print("prior max: ", around_prior_data.max().max())
        print("prior min: ", around_prior_data.min().min())
        print("prior shape: ", around_prior_data.shape)
        if around_prior_data.max().max() > non_ufo_threshold:
            print("High value, skipping")
            print(idx, ufo)
        elif around_prior_data.min().min() == 0:
            print("found a zero min value, skipping")
            print(idx, ufo)
        else:
            non_ufo_prior.append(around_prior_data)
            if include_meta:
                prior_meta = ufo.copy()
                prior_meta["datetime"] = prior_meta["datetime"] - pd.Timedelta(
                    f"{prior_dt}s"
                )
                non_ufo_prior_meta.append(prior_meta)

        print("post max: ", around_post_data.max().max())
        print("post min: ", around_post_data.min().min())
        print("post shape: ", around_post_data.shape)
        if around_post_data.max().max() > non_ufo_threshold:
            print("High value, skipping")
            print(idx, ufo)
        elif around_post_data.min().min() == 0:
            print("found a zero min value, skipping")
            print(idx, ufo)
        else:
            non_ufo_post.append(around_post_data)
            if include_meta:
                post_meta = ufo.copy()
                post_meta["datetime"] = post_meta["datetime"] + pd.Timedelta(
                    f"{post_dt}s"
                )
                non_ufo_post_meta.append(post_meta)

    out = {
        "ufo": truncate_data(peaks, target_length=target_length),
        "non_ufo_prior": truncate_data(non_ufo_prior, target_length=target_length),
        "non_ufo_post": truncate_data(non_ufo_post, target_length=target_length),
    }
    if include_meta:
        out["ufo_meta"] = pd.DataFrame(peaks_meta)
        out["non_ufo_prior_meta"] = pd.DataFrame(non_ufo_prior_meta)
        out["non_ufo_post_meta"] = pd.DataFrame(non_ufo_post_meta)
    return out


def rolling_window(a: np.ndarray, window: int) -> np.ndarray:
    """Create a rolling window over the provided array.

    Args:
        a: array on which to perform the rolling window
        window: the size of the rolling window

    Returns:
        An array of the rolling window.
    """
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def create_rolling_window_dataset(
    ufo_meta: pd.DataFrame,
    raw_data_dir: Path,
    window_size: int,
    safety_gap: int,
    include_meta: bool = True,
) -> Dict[str, np.ndarray]:
    """Create a ufo non ufo dataset with rolling windows.

    Args:
        ufo_meta: metadata of the ufo events
        raw_data_dir: directory containing the raw data
        window_size: the size of the rolling window
        safety_gap: empty gap around the ufo which are not assigned to non ufo
        include_meta: include the dataset's metadata

    Returns:
        A dictionary with the windows containing the ufos and the windows which don't.
    """
    half_window = int((window_size - 1) / 2)
    rolling_ufo_dataset = []
    rolling_ufo_meta = []
    rolling_non_ufo_dataset = []
    rolling_non_ufo_meta = []
    for fill, grp in tqdm(ufo_meta.groupby("fill")):
        try:
            raw_fill_data = BLMData.load(raw_data_dir / f"{fill}.h5")
            raw_fill_data.df = raw_fill_data.df.droplevel("mode")
            raw_fill_data.df = raw_fill_data.df.iloc[
                ~raw_fill_data.df.index.duplicated()
            ]
        except Exception as e:
            print(f"file error {fill}, skipping {e}")
            continue

        for _, ufo in grp.iterrows():
            print(ufo.blm, ufo.dcum)

            # raw_idx = raw_fill_data.df.index.get_loc(ufo.datetime, method="nearest") + 1
            ufo_lm = raw_fill_data.loss_map(ufo.datetime + pd.Timedelta("1s"))
            ufo_np = ufo_lm.df["data"].to_numpy()
            ufo_rolling = rolling_window(
                np.pad(ufo_np, half_window, mode="wrap"), window_size
            )

            try:
                ufo_rolling_ufo = ufo_rolling[
                    ufo_lm.df.index.get_loc(ufo.blm)
                    - half_window : ufo_lm.df.index.get_loc(ufo.blm)
                    + half_window
                    + 1
                ]

                ufo_rolling_non_ufo = np.vstack(
                    [
                        ufo_rolling[
                            : ufo_lm.df.index.get_loc(ufo.blm)
                            - half_window
                            - safety_gap
                        ],
                        ufo_rolling[
                            ufo_lm.df.index.get_loc(ufo.blm)
                            + half_window
                            + safety_gap
                            + 1 :
                        ],
                    ]
                )
            except KeyError:
                continue

            rolling_ufo_dataset.append(ufo_rolling_ufo)
            rolling_non_ufo_dataset.append(ufo_rolling_non_ufo)
            if include_meta:
                rolling_ufo_meta.extend([ufo] * len(ufo_rolling_ufo))
                rolling_non_ufo_meta.extend([ufo] * len(ufo_rolling_non_ufo))
    out = {
        "ufo": np.vstack(rolling_ufo_dataset),
        "non_ufo": np.vstack(rolling_non_ufo_dataset),
    }
    if include_meta:
        out["ufo_meta"] = pd.DataFrame(rolling_ufo_meta)
        out["non_ufo_meta"] = pd.DataFrame(rolling_non_ufo_meta)
    return out


def create_full_lm_dataset(
    ufo_meta: pd.DataFrame,
    raw_data_dir: Path,
    prior_dt: int = 5,
    post_dt: int = 5,
    include_meta: bool = True,
):
    """Create a dataset of the full loss maps of the ufos, and as non ufos the loss maps
    before and after.

    Args:
        ufo_meta: metadata of the ufo events
        raw_data_dir: directory containing the raw data
        prior_dt: how many seconds before the ufo to take as non ufo
        post_dt: how many seoconds after the ufo to take as non ufo
        include_meta: include the dataset's metadata

    Returns:
        Dictionary with the ufo and non ufo events, along with the metadata if requested.
    """
    ufo_dataset = []
    ufo_dataset_meta = []
    non_ufo_post_dataset = []
    non_ufo_post_dataset_meta = []
    non_ufo_prior_dataset = []
    non_ufo_prior_dataset_meta = []

    for fill, grp in tqdm(ufo_meta.groupby("fill")):
        try:
            raw_fill_data = BLMData.load(raw_data_dir / f"{fill}.h5")
            raw_fill_data.df = raw_fill_data.df.droplevel("mode")
            raw_fill_data.df = raw_fill_data.df.iloc[
                ~raw_fill_data.df.index.duplicated()
            ]
        except Exception as e:
            print(f"file error {fill}, skipping {e}")
            continue

        for idx, ufo in grp.iterrows():
            print(ufo.blm, ufo.dcum)
            if ufo.dcum >= 317796 - 20000 and ufo.dcum <= 317796 + 40000:
                print("skipping, ufo location")
                continue

            ufo_datetime = ufo["datetime"] + pd.Timedelta("1s")
            ufo_lm = raw_fill_data.loss_map(ufo_datetime)
            non_ufo_post_lm = raw_fill_data.loss_map(
                ufo_datetime + pd.Timedelta(f"{post_dt}s")
            )
            non_ufo_prior_lm = raw_fill_data.loss_map(
                ufo_datetime - pd.Timedelta(f"{prior_dt}s")
            )

            if (ufo_lm.df["data"] == 0).sum() == 0:
                ufo_dataset.append(ufo_lm.df["data"].to_numpy())
                if include_meta:
                    ufo_dataset_meta.append(ufo)
            if (non_ufo_post_lm.df["data"] == 0).sum() == 0:
                non_ufo_post_dataset.append(non_ufo_post_lm.df["data"].to_numpy())
                if include_meta:
                    post_meta = ufo.copy()
                    post_meta["datetime"] = post_meta["datetime"] + pd.Timedelta(
                        f"{post_dt}s"
                    )
                    non_ufo_post_dataset_meta.append(post_meta)
            if (non_ufo_prior_lm.df["data"] == 0).sum() == 0:
                non_ufo_prior_dataset.append(non_ufo_prior_lm.df["data"].to_numpy())
                if include_meta:
                    prior_meta = ufo.copy()
                    prior_meta["datetime"] = prior_meta["datetime"] - pd.Timedelta(
                        f"{post_dt}s"
                    )
                    non_ufo_prior_dataset_meta.append(prior_meta)
    out = {
        "ufo": np.vstack(ufo_dataset),
        "non_ufo_prior": np.vstack(non_ufo_prior_dataset),
        "non_ufo_post": np.vstack(non_ufo_post_dataset),
    }
    if include_meta:
        out["ufo_meta"] = pd.DataFrame(ufo_dataset_meta)
        out["non_ufo_prior_meta"] = pd.DataFrame(non_ufo_prior_dataset_meta)
        out["non_ufo_post_meta"] = pd.DataFrame(non_ufo_post_dataset_meta)
        out["lm_meta"] = ufo_lm.df.drop(columns="data")
    return out


class DataGenerator:
    @classmethod
    def load(cls, dir_path: Path) -> "DataGenerator":
        """Create a DataGenerator instance from a json file.

        Args:
            dir_path: folder containing the saved generator

        Returns:
            The DataGenerator instance.
        """
        dir_path = Path(dir_path)
        with open(dir_path / "params.json", "r") as fp:
            param_dict = json.load(fp)
        param_dict["data"] = np.load(dir_path / "data.npy")
        param_dict["labels"] = np.load(dir_path / "labels.npy")
        param_dict["indices"] = np.load(dir_path / "indices.npy")
        param_dict["metadata"] = pd.read_hdf(dir_path / "metadata.h5")
        return cls(**param_dict)

    def __init__(
        self,
        data: np.ndarray,
        metadata: pd.DataFrame,
        labels: np.ndarray,
        indices: Optional[np.ndarray] = None,
        batch_size: int = 16,
        seed: int = 42,
        norm_method: str = "znorm",
        norm_axis: int = 0,
        ndim: int = 3,
        context: Optional[Dict[Any, Any]] = None,
    ):
        self._log = logging.getLogger(__name__)
        self.data = data
        self.metadata = metadata
        self.labels = labels
        self.batch_size = batch_size
        self.seed = seed
        self.norm_method = norm_method
        self.norm_axis = norm_axis
        self.ndim = ndim
        self.context = context
        self._rng = np.random.default_rng(self.seed)
        norm_methods = {
            "min_max": (self.norm_min_max, self.unnorm_min_max),
            "znorm": (self.norm_znorm, self.unnorm_znorm),
        }
        self._norm_func, self._unnorm_func = norm_methods[self.norm_method]
        self._norm_factors = None
        if indices is None:
            indices = np.arange(len(self.data))
        self.indices = indices

    def norm_min_max(self, data: np.ndarray) -> np.ndarray:
        if self._norm_factors is None:
            mins = data.min(axis=self.norm_axis)
            maxes = data.max(axis=self.norm_axis)
            self._norm_factors = {"mins": mins, "maxes": maxes}
        else:
            mins = self._norm_factors["mins"]
            maxes = self._norm_factors["maxes"]
        out = data.copy()
        if self.norm_axis == 1:
            mins = mins[:, None]
            maxes = maxes[:, None]
        out -= mins
        # blm 2964 can just be constant which causes issues
        div = maxes - mins
        div[div == 0] = 1
        out /= div
        return out

    def unnorm_min_max(self, data: np.ndarray) -> np.ndarray:
        out = data.copy()
        if self._norm_factors is None:
            raise ValueError("_norm_factors is None, has it been normalized?")
        out *= self._norm_factors["maxes"] - self._norm_factors["mins"]
        out += self._norm_factors["mins"]
        return out

    def norm_znorm(self, data: np.ndarray) -> np.ndarray:
        if self._norm_factors is None:
            stds = data.std(axis=self.norm_axis)
            means = data.mean(axis=self.norm_axis)
            self._norm_factors = {"stds": stds, "means": means}
        else:
            stds = self._norm_factors["stds"]
            means = self._norm_factors["means"]
        out = data.copy()
        if self.norm_axis == 1:
            means = means[:, None]
            stds = stds[:, None]
        out -= means
        out /= stds
        return out

    def unnorm_znorm(self, data: np.ndarray) -> np.ndarray:
        out = data.copy()
        if self._norm_factors is None:
            raise ValueError("_norm_factors is None, has it been normalized?")
        out *= self._norm_factors["stds"]
        out += self._norm_factors["means"]
        return out

    def _pre_norm(self, data: np.ndarray) -> np.ndarray:
        return np.log10(data)

    def pre_norm(self) -> "DataGenerator":
        self.data = self._pre_norm(self.data)
        return self

    def shuffle(self) -> "DataGenerator":
        self._log.debug("Shuffling indices, seed %s", self.seed)
        self._rng.shuffle(self.indices)
        self.data = self.data[self.indices]
        self.metadata = self.metadata.iloc[self.indices]
        self.labels = self.labels[self.indices]
        return self

    def norm(
        self, norm_factors_from: Optional["DataGenerator"] = None
    ) -> "DataGenerator":
        if norm_factors_from is not None:
            self._norm_factors = norm_factors_from._norm_factors
        self.data = self._norm_func(self.data)
        return self

    def unnorm(self) -> "DataGenerator":
        self.data = self._unnorm_func(self.data)
        return self

    def _expand(self, data: np.ndarray) -> np.ndarray:
        n_expand = max(self.ndim - data.ndim, 0)
        if n_expand > 0:
            self._log.info("Expanding the dimensions by: %i", n_expand)
            data = data[(..., *([np.newaxis] * n_expand))]
        return data

    def expand(self) -> "DataGenerator":
        self.data = self._expand(self.data)
        return self

    def split(self, ratio: float) -> Tuple["DataGenerator", "DataGenerator"]:
        """Split the generator without overlapping data samples.

        Useful for creating a training and validation generator.

        Args:
            ratio: split ratio.

        Returns:
            The 2 DataGenerator instances, the first will generate `ratio`% of
                the data. The second (1 - `ratio`)% of the data.
        """
        self._log.debug("Splitting dataset.")
        n_samples = int(ratio * len(self.data))
        self._log.debug("Split n_samples: %i", n_samples)
        indices = np.arange(len(self.data))
        split_indices = self._rng.choice(indices, n_samples, replace=False)
        remaining_indices = np.delete(indices, split_indices)
        self._log.debug("Split size: %s", split_indices.shape)
        self._log.debug("Split remaining size: %s", remaining_indices.shape)
        split_params = self.to_dict()
        split_params["data"] = self.data[split_indices]
        split_params["labels"] = self.labels[split_indices]
        split_params["metadata"] = self.metadata.iloc[split_indices]
        split_params["indices"] = None
        split_params["context"] = {"pre_split_indices": split_indices.tolist()}
        remaining_params = self.to_dict()
        remaining_params["data"] = self.data[remaining_indices]
        remaining_params["labels"] = self.labels[remaining_indices]
        remaining_params["metadata"] = self.metadata.iloc[remaining_indices]
        remaining_params["indices"] = None
        remaining_params["context"] = {"pre_split_indices": remaining_indices.tolist()}
        return DataGenerator(**split_params), DataGenerator(**remaining_params)

    def to_dict(self) -> dict:
        return dict(
            data=self.data,
            metadata=self.metadata,
            labels=self.labels,
            indices=self.indices,
            batch_size=self.batch_size,
            seed=self.seed,
            norm_method=self.norm_method,
            norm_axis=self.norm_axis,
            ndim=self.ndim,
            context=self.context,
        )

    def save(self, dir_path: Path) -> None:
        """Return a serialized json string of the arguments of the current instance.

        Args:
            dir_path: folder where to save the generator
        """
        dir_path = Path(dir_path)
        dump_kwargs = {"indent": 2}
        param_dict = self.to_dict()
        if not dir_path.is_dir():
            dir_path.mkdir(parents=True)
        del param_dict["data"]
        del param_dict["metadata"]
        del param_dict["labels"]
        del param_dict["indices"]
        with open(dir_path / "params.json", "w") as fp:
            json.dump(param_dict, fp, **dump_kwargs)
        np.save(dir_path / "data.npy", self.data)
        np.save(dir_path / "labels.npy", self.labels)
        np.save(dir_path / "indices.npy", self.indices)
        self.metadata.to_hdf(dir_path / "metadata.h5", "data")
