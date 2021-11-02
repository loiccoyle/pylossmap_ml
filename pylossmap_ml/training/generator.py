import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence
from tqdm.auto import tqdm


class DataGenerator(Sequence):
    @classmethod
    def from_json(cls, parameter_file: Path) -> "DataGenerator":
        """Create a DataGenerator instance from a json file.

        Args:
            parameter_file: file from which to read the parameters.

        Returns:
            The DataGenerator instance.
        """
        with open(parameter_file, "r") as fp:
            param_dict = json.load(fp)
        param_dict["data_file"] = Path(param_dict["data_file"])
        param_dict["BLM_dcum"] = pd.Series(param_dict["BLM_dcum"])
        if param_dict["indices"] is not None:
            param_dict["indices"] = np.array(param_dict["indices"])
        return cls(**param_dict)

    def __init__(
        self,
        data_file: Path,
        key: str = "STABLE",
        shuffle: bool = True,
        batch_size: int = 1024,
        seed: int = 42,
        norm_method: str = "min_max",
        norm_axis: int = 0,
        norm_kwargs: dict = {},
        BLM_names: Optional[List[str]] = None,
        BLM_dcum: Optional[pd.Series] = None,
        return_dataframe: bool = False,
        ndim: int = 3,
        indices: Optional[np.ndarray] = None,
    ):
        """Lossmap data hdf5 data generator.

        Args:
            data_file: path of the hdf file.
            key: key within the hdf file.
            shuffle: shuffle the order of the data samples within the datafile,
                this is ignored if `indices` is provided.
            batch_size: the number of samples to load at each iteration.
            seed: the random seed.
            norm_method: the normalization method to use.
            norm_axis: the normalization axis, 0 to normaliza each BLM across
                the entire dataset. 1 to normalize each loss map.
            norm_kwargs: passed to the normalization method.
            BLM_names: The names of the BLM columns in the `data_file`.
            BLM_dcum: BLM position data.
            return_dataframe: Return `pd.DataFrame`s.
            ndim: expand the number of dimension in the numpy array.
            indices: the indices of the samples to load from the data file.
        """
        self._log = logging.getLogger(__name__)

        self.data_file = data_file
        self._store = None
        self.key = key
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.seed = seed
        self._rng = np.random.default_rng(self.seed)
        self.norm_method = norm_method
        self.norm_axis = norm_axis
        self.norm_kwargs = norm_kwargs
        self.BLM_names = BLM_names
        self.BLM_dcum = BLM_dcum
        self.return_dataframe = return_dataframe
        self.ndim = ndim
        self._data_len = self.get_data_length()
        if indices is None:
            self._log.debug("Creating indices.")
            indices = np.arange(self._data_len)  # type: np.ndarray
            if self.shuffle:
                self._log.debug("Shuffling indices, seed %s", self.seed)
                self._rng.shuffle(indices)
        self.indices = indices  # type: np.ndarray

        self._mins_maxes = None
        self._blm_sorted = None
        # self._indices = np.arange(self._data_len)
        norm_methods = {"min_max": self.norm_min_max}
        self._norm_func = norm_methods[self.norm_method]

    def _compute_min_max(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the min and max across the entire dataset.

        Returns:
            An array of minimas and an array of maximas.
        """
        mins = []
        maxes = []
        for chunk in tqdm(
            self.store.select(self.key, chunksize=self.batch_size, iterator=True),
            total=int(self._data_len // self.batch_size),
            desc=f"Computing mins & maxes, axis={self.norm_axis}",
        ):

            maxes.append(chunk.max(axis=self.norm_axis))
            mins.append(chunk.min(axis=self.norm_axis))

        return (
            pd.concat(mins, axis=1).min(axis=1).to_numpy(),
            pd.concat(maxes, axis=1).max(axis=1).to_numpy(),
        )

    @property
    def store(self) -> pd.HDFStore:
        if self._store is None:
            self._log.debug("Opening hdf file.")
            self._store = pd.HDFStore(self.data_file, "r")
        return self._store

    @property
    def mins_maxes(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._mins_maxes is None:
            self._log.debug("Computing mins & maxes.")
            self._mins_maxes = self._compute_min_max()
        return self._mins_maxes

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
        n_samples = int(ratio * len(self.indices))
        self._log.debug("Split n_samples: %i", n_samples)
        split_indices_index = self._rng.choice(
            np.arange(len(self.indices)), n_samples, replace=False
        )
        split_indices = self.indices[split_indices_index]
        remaining_indices = np.delete(self.indices, split_indices_index)
        self._log.debug("Split size: %s", split_indices.shape)
        self._log.debug("Split remaining size: %s", remaining_indices.shape)

        split_params = self.to_dict()
        split_params["indices"] = split_indices
        remaining_params = self.to_dict()
        remaining_params["indices"] = remaining_indices

        # compute the mins & maxes across the entire dataset
        generator_split = DataGenerator(**split_params)
        generator_split._mins_maxes = self.mins_maxes
        generator_remaining = DataGenerator(**remaining_params)
        generator_remaining._mins_maxes = self.mins_maxes
        return generator_split, generator_remaining

    def get_metadata(self, entire_data_file: bool = False, **kwargs) -> np.ndarray:
        """Read the index of the data file.

        Args:
            entire_data_file: return the metadata of the entire dataset or just
                of the indices concerned by the generator.
            **kwargs: passed to the `pd.HDFStore.select` method.

        Returns:
            A numpy array containing the metadata of the dataset.
        """
        chunks = [
            chunk.index.to_numpy()
            for chunk in self.store.select(
                self.key, columns=[0], chunksize=min(int(1e6), self._data_len), **kwargs
            )
        ]
        chunks = np.hstack(chunks)
        if not entire_data_file:
            chunks = chunks[self.indices]
        return chunks

    def norm_min_max(self, data: np.ndarray) -> np.ndarray:
        """Normalize by setting the minimum to 0 and the maximum to 1.

        Args:
            data: The data to normalize.

        Returns:
            The normalized data.
        """
        mins, maxes = self.mins_maxes
        self._log.debug("mins, maxes shape: %s, %s", mins.shape, maxes.shape)
        data -= mins
        data /= maxes - mins
        return data

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """Run the chosen normalization method.

        Args:
            data: the data to normalize.

        Returns:
            The normalized data.
        """
        return self._norm_func(data, **self.norm_kwargs)

    def get_data_length(self) -> int:
        """Get the length of the dataset, i.e. the number of samples in the dataset.

        Returns:
            The number of rows in the dataset

        Raises:
            ValueError: when the number of rows could not be determined.
        """
        out = self.store.get_storer(self.key).nrows
        if out is None:
            raise ValueError("Could not determine the dataset length. Is is empty ?")
        return out

    def reorder_blms(self, data: pd.DataFrame) -> pd.DataFrame:
        """Reorder the BLMs w.r.t. the BLM dcum.

        Args:
            data: Data for which to reorder the BLMs.

        Returns:
            The data with columns sorted by dcum.
        """
        if self.BLM_dcum is not None:
            if self._blm_sorted is None:
                self._log.debug("Computing blm_sorted.")
                self._blm_sorted = (
                    self.BLM_dcum.loc[data.columns.to_list()]
                    .sort_values()
                    .index.to_list()
                )
            self._log.info("Reordering BLMs.")
            return data[self._blm_sorted]
        return data

    def _create_subset(self, index: int) -> np.ndarray:
        """Get the indices of the subset for the provided index.

        Args:
            index: the current iteration index.

        Returns:
            The indices of the current subset.
        """
        indices = self.indices[
            index
            * self.batch_size : min((index + 1) * self.batch_size, len(self.indices))
        ]
        return indices

    def to_dict(self) -> dict:
        """Return a dictionary with the arguments of the current instance.

        Returns:
            Dictionary with the arguments of the current instance.
        """
        return dict(
            data_file=self.data_file,
            key=self.key,
            shuffle=self.shuffle,
            batch_size=self.batch_size,
            seed=self.seed,
            norm_method=self.norm_method,
            norm_axis=self.norm_axis,
            norm_kwargs=self.norm_kwargs,
            BLM_names=self.BLM_names,
            BLM_dcum=self.BLM_dcum,
            return_dataframe=self.return_dataframe,
            ndim=self.ndim,
            indices=self.indices,
        )

    def to_json(self, file_path: Optional[Path] = None) -> str:
        """Return a serialized json string of the arguments of the current instance.

        Args:
            file_path: if provided, will also write the json string to file.

        Returns:
            The json string.
        """
        dump_kwargs = {"indent": 2}
        param_dict = self.to_dict()
        param_dict["data_file"] = str(param_dict["data_file"].resolve())
        param_dict["BLM_dcum"] = param_dict["BLM_dcum"].to_dict()
        if param_dict["indices"] is not None:
            param_dict["indices"] = param_dict["indices"].tolist()
        if file_path is not None:
            with open(file_path, "w") as fp:
                json.dump(param_dict, fp, **dump_kwargs)
        return json.dumps(param_dict, **dump_kwargs)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        subset = self._create_subset(index)
        subset_data = self.store.select(self.key, where=subset)
        self._log.debug("Subset shape: %s", subset_data.shape)
        subset_data = self.normalize(subset_data)

        if self.BLM_names is not None:
            self._log.info("Assigning BLM names.")
            subset_data.columns = self.BLM_names
        subset_data = self.reorder_blms(subset_data)

        if not self.return_dataframe:
            self._log.info("Converting to numpy array.")
            subset_data = subset_data.to_numpy()
            # Expand the number of dimensions in the subset array
            n_expand = max(self.ndim - subset_data.ndim, 0)
            if n_expand > 0:
                self._log.info("Expanding the dimensions by: %i", n_expand)
                subset_data = subset_data[(..., *([np.newaxis] * n_expand))]
        return subset_data, subset_data

    def __len__(self) -> int:
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __del__(self):
        if self._store is not None:
            self._store.close()
