import logging
from abc import abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from ..db import DB
from ..utils import get_fill_particle
from .utils import INTENSITY, timber_to_df


class BasePreprocessor:
    def __init__(
        self,
        blm_list: Optional[List[str]] = None,
        drop_blm_names: bool = True,
        particle_type: Optional[str] = None,
    ):
        self._log = logging.getLogger(__name__)
        self.blm_list = blm_list
        self.drop_blm_names = drop_blm_names
        self.particle_type = particle_type

    @abstractmethod
    def _preprocess(self, path_to_hdf: Path) -> Optional[pd.DataFrame]:
        pass

    def preprocess(self, path_to_hdf: Path) -> Optional[pd.DataFrame]:
        output = self._preprocess(path_to_hdf)

        if output is not None and self.drop_blm_names:
            output.columns = range(len(output.columns))
        return output

    @abstractmethod
    def normalize(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

    @staticmethod
    def path_to_fill(path: Path) -> int:
        return int(path.stem)

    def check_particle_type(self, fill_number: int) -> bool:
        if self.particle_type is None:
            return True
        return all(
            [
                beam_particle == self.particle_type
                for beam_particle in get_fill_particle(fill_number)
            ]
        )

    def blm_filter(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.blm_list is not None:
            return data[self.blm_list]
        return data

    @staticmethod
    def load_raw(path_to_hdf: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        path_to_hdf = Path(path_to_hdf).with_suffix(".h5")
        if not path_to_hdf.is_file():
            raise FileNotFoundError(f"File {path_to_hdf} not found.")

        data = pd.read_hdf(path_to_hdf, "data")
        meta = pd.read_hdf(path_to_hdf, "meta")
        header = pd.read_hdf(path_to_hdf, "header")
        header = header[0].tolist()
        # # read real columns from csv file & replace fake columns
        # with open(path_to_hdf.with_suffix('.csv'), 'r') as fp:
        #     columns = fp.readlines()
        data.columns = [c.rstrip() for c in header]
        data_blms = data.columns.to_list()
        sorted_blms = meta["dcum"].loc[data_blms].sort_values().index.to_list()
        print(sorted_blms[:5])
        data = data[sorted_blms]
        return data, meta


class NormMaxMixin:
    @staticmethod
    def normalize(data: pd.DataFrame) -> pd.DataFrame:
        """Normalize to the highest BLM signal in the data sample.

        Args:
            Data to normalize.

        Returns:
            Normalized data.
        """
        return data.divide(data.max(axis=1), axis=0)


class NormSumMixin:
    @staticmethod
    def normalize(data: pd.DataFrame) -> pd.DataFrame:
        """Normalize to the to the sum of the BLM signals.

        Args:
            Data to normalize.

        Returns:
            Normalized data.
        """
        return data.divide(data.sum(axis=1), axis=0)


class RollingWindowSum(NormMaxMixin, BasePreprocessor):
    def __init__(
        self,
        blm_list: Optional[List[str]] = None,
        drop_blm_names: bool = True,
        particle_type: Optional[str] = None,
        window_size: str = "60s",
        min_periods: int = 60,
    ):
        """This preprocessor does a rolling window sum through time.

        It normalize w.r.t. the highest BLM of each sample.
        """
        super().__init__(
            blm_list=blm_list,
            drop_blm_names=drop_blm_names,
            particle_type=particle_type,
        )
        self.window_size = window_size
        self.min_periods = min_periods

    def _preprocess(self, path_to_hdf: Path) -> pd.DataFrame:
        path_to_hdf = Path(path_to_hdf)
        fill_number = self.path_to_fill(path_to_hdf)
        if not self.check_particle_type(fill_number):
            self._log.warning(
                "Fill %i particle type is not %s", fill_number, self.particle_type
            )
            return None

        data, _ = self.load_raw(path_to_hdf)
        beam_mode = data.index.get_level_values("mode")[0]
        data = (
            data.droplevel("mode")
            .rolling(self.window_size, min_periods=self.min_periods)
            .sum()
            .dropna()
        )
        data = self.blm_filter(data)
        data = self.normalize(data)
        # readd beam_mode and fill number
        # very ugly
        data = pd.concat([data], keys=[beam_mode], names=["mode"])
        data = pd.concat([data], keys=[fill_number], names=["fill_number"])
        return data


class PassThrough(BasePreprocessor):
    """This preprocessor does nothing, it just returns the raw values."""

    def _preprocess(self, path_to_hdf: Path):
        path_to_hdf = Path(path_to_hdf)
        data, _ = self.load_raw(path_to_hdf)
        return data


class NormMax(NormMaxMixin, BasePreprocessor):
    """This preprocessor just normalizes the data to the highest BLM signal."""

    def _preprocess(self, path_to_hdf: Path) -> pd.DataFrame:
        path_to_hdf = Path(path_to_hdf)
        data, _ = self.load_raw(path_to_hdf)
        data = self.blm_filter(data)
        data = self.normalize(data)
        return data


class NoDumpDiff(BasePreprocessor):
    def __init__(
        self,
        blm_list: Optional[List[str]] = None,
        drop_blm_names: bool = True,
        particle_type: Optional[str] = None,
        intensity_threshold: float = 1500e11,
    ):
        """This preprocessor normalizes to the highest BLM signal while also
        filtering out fills where the starting intensity is below the `intensity_threshold`
        value.

        It also prunes the data to the last data point where the intensity is above
        the `intensity_threshold_dump` value.
        """
        super().__init__(
            blm_list=blm_list,
            drop_blm_names=drop_blm_names,
            particle_type=particle_type,
        )
        self.intensity_threshold = intensity_threshold

    @staticmethod
    def normalize(data: pd.DataFrame) -> pd.DataFrame:
        return data

    def _preprocess(self, path_to_hdf: Path) -> Optional[pd.DataFrame]:
        path_to_hdf = Path(path_to_hdf)
        fill_number = int(path_to_hdf.stem)
        if not self.check_particle_type(fill_number):
            self._log.warning(
                "Fill %i particle type is not %s", fill_number, self.particle_type
            )
            return None

        data, _ = self.load_raw(path_to_hdf)
        beam_mode = data.index.get_level_values("mode")[0]
        print("pre filter", data.shape)
        data = self.blm_filter(data)
        print("post filter", data.shape)
        t1 = data.index.get_level_values("timestamp")[0]
        t2 = data.index.get_level_values("timestamp")[-1]
        int_data = DB.get([INTENSITY.format(beam=1), INTENSITY.format(beam=2)], t1, t2)
        int_B1 = timber_to_df(int_data, INTENSITY.format(beam=1))
        int_B2 = timber_to_df(int_data, INTENSITY.format(beam=2))
        # if there is low intensity at the begining of the mode
        if (int_B1.iloc[0] < self.intensity_threshold).values or (
            int_B2.iloc[0] < self.intensity_threshold
        ).values:
            self._log.warning("%s start intensity too low.", path_to_hdf)
            self._log.warning("B1 start intensity: %s", int_B1.iloc[0].values)
            self._log.warning("B2 start intensity: %s", int_B2.iloc[0].values)
            self._log.warning("Intensity threshold: %e", self.intensity_threshold)
            return None

        int_df = pd.concat([int_B1, int_B2], sort=False, axis=1)
        if (int_df == 0).any().any():
            self._log.info("Pruning the dump region.")
            # prune the leftover post dump data
            # row where the big drop occurs
            t_dump_index = int_df.diff().reset_index(drop=True).idxmin().min() - 2
            print("t_dump_index", t_dump_index)
            data = data.iloc[:t_dump_index]
            print("data pruned: ", data.shape)

        data = self.normalize(data)
        print("data normalized", data.shape)
        # add the fill number to the dataframe.
        data = pd.concat([data], keys=[fill_number], names=["fill_number"])
        return data


class NoDump(BasePreprocessor):
    def __init__(
        self,
        blm_list: Optional[List[str]] = None,
        drop_blm_names: bool = True,
        particle_type: Optional[str] = None,
        intensity_threshold: float = 1500e11,
        intensity_threshold_dump: float = 1e11,
    ):
        """This preprocessor normalizes to the highest BLM signal while also
        filtering out fills where the starting intensity is below the `intensity_threshold`
        value.

        It also prunes the data to the last data point where the intensity is above
        the `intensity_threshold_dump` value.
        """
        super().__init__(
            blm_list=blm_list,
            drop_blm_names=drop_blm_names,
            particle_type=particle_type,
        )
        self.intensity_threshold = intensity_threshold
        self.intensity_threshold_dump = intensity_threshold_dump

    @staticmethod
    def normalize(data: pd.DataFrame) -> pd.DataFrame:
        return data

    def _preprocess(self, path_to_hdf: Path) -> Optional[pd.DataFrame]:
        path_to_hdf = Path(path_to_hdf)
        fill_number = int(path_to_hdf.stem)
        if not self.check_particle_type(fill_number):
            self._log.warning(
                "Fill %i particle type is not %s", fill_number, self.particle_type
            )
            return None

        data, _ = self.load_raw(path_to_hdf)
        beam_mode = data.index.get_level_values("mode")[0]
        print("pre filter", data.shape)
        data = self.blm_filter(data)
        print("post filter", data.shape)
        t1 = data.index.get_level_values("timestamp")[0]
        t2 = data.index.get_level_values("timestamp")[-1]
        int_data = DB.get([INTENSITY.format(beam=1), INTENSITY.format(beam=2)], t1, t2)
        int_B1 = timber_to_df(int_data, INTENSITY.format(beam=1))
        int_B2 = timber_to_df(int_data, INTENSITY.format(beam=2))
        # if there is low intensity at the begining of the mode
        if (int_B1.iloc[0] < self.intensity_threshold).values or (
            int_B2.iloc[0] < self.intensity_threshold
        ).values:
            self._log.warning("%s start intensity too low.", path_to_hdf)
            self._log.warning("B1 start intensity: %s", int_B1.iloc[0].values)
            self._log.warning("B2 start intensity: %s", int_B2.iloc[0].values)
            self._log.warning("Intensity threshold: %e", self.intensity_threshold)
            return None

        int_df = pd.concat([int_B1, int_B2], sort=False, axis=1)
        # get the last timestamp where the intensity is above threshold
        t_dump = (
            int_df[(int_df > self.intensity_threshold_dump).all(axis=1)]
            .dropna()
            .index[-1]
        )
        print("t_dump", t_dump)
        data = data.loc[:(beam_mode, t_dump)]
        print("data t_dumped", data.shape)
        data = self.normalize(data)
        print("data normalized", data.shape)

        # add the fill number to the dataframe.
        data = pd.concat([data], keys=[fill_number], names=["fill_number"])
        return data


class NormMaxNoDump(NormMaxMixin, NoDump):
    pass


class NormSumNoDump(NormSumMixin, NoDump):
    pass


class NormMaxNoDumpDiff(NormMaxMixin, NoDumpDiff):
    pass


class NormSumNoDumpDiff(NormSumMixin, NoDumpDiff):
    pass


# class ListSumNormNoDump(FilterNormNoDump):

#     def __init__(self, blm_list):
#         self.blm_list = blm_list

#     def blm_filter(self, data, meta):
#         return data[self.blm_list]


# class ListSumNormNoDump(MaxSumNoDump):
#     def __init__(
#         self,
#         blm_list: list,
#         drop_blm_names: bool = False,
#         intensity_threshold: float = 1500e11,
#         intensity_threshold_dump: float = 1e11,
#     ):
#         self.blm_list = blm_list
#         self.drop_blm_names = drop_blm_names
#         self.intensity_threshold = intensity_threshold
#         self.intensity_threshold_dump = intensity_threshold_dump

#     def blm_filter(self, data, meta):
#         return data[self.blm_list]

#     def normalize(self, data):
#         return data.divide(data.sum(axis=1), axis=0)

#     def preprocess(self, path_to_hdf):
#         path_to_hdf = Path(path_to_hdf)
#         fill_number = int(path_to_hdf.stem)
#         data, meta = self.load_raw(path_to_hdf)
#         # print(data.head())
#         # print(meta.head())
#         beam_mode = data.index.get_level_values("mode")[0]
#         data = self.blm_filter(data, meta)
#         t1 = data.index.get_level_values("timestamp")[0]
#         t2 = data.index.get_level_values("timestamp")[-1]
#         int_data = DB.get([INTENSITY.format(beam=1), INTENSITY.format(beam=2)], t1, t2)
#         int_B1 = timber_to_df(int_data, INTENSITY.format(beam=1))
#         int_B2 = timber_to_df(int_data, INTENSITY.format(beam=2))
#         # if there is low intensity at the begining of the mode
#         if (int_B1.iloc[0] < self.intensity_threshold).values or (
#             int_B2.iloc[0] < self.intensity_threshold
#         ).values:
#             print(path_to_hdf, "does not pass")
#             return None

#         int_df = pd.concat([int_B1, int_B2], sort=False, axis=1)
#         # get the last timestamp where the intensity is above 1e11
#         t_dump = (
#             int_df[(int_df > self.intensity_threshold_dump).all(axis=1)]
#             .dropna()
#             .index[-1]
#         )

#         data = data.loc[:(beam_mode, t_dump)]
#         data = self.normalize(data)

#         if self.drop_blm_names:
#             data.columns = range(len(data.columns))

#         # add the fill number to the dataframe.
#         data = pd.concat([data], keys=[fill_number], names=["fill_number"])
#         return data


# class ColdFilterSumNormNoDump(FilterMaxNormNoDump):
#     def blm_filter(self, data, meta):
#         filtered_blms = meta[meta["type"] != "cold"].index.tolist()
#         return data[filtered_blms]

#     def normalize(self, data):
#         return data.divide(data.sum(axis=1), axis=0)


# class NoDumpMaxNorm(BasePreprocessor):
#     def __init__(self, BLM_filter):
#         super().__init__(BLM_filter)

#     def normalize(self, data):
#         # normalize to the max BLM signal
#         return data.divide(data.max(axis=1), axis=0)

#     def preprocess(self, path_to_hdf):
#         path_to_hdf = Path(path_to_hdf)
#         data, meta = self.load_raw(path_to_hdf)
#         filtered_blms = self.blm_filter(data.columns)
#         data = data[filtered_blms]
#         # remove data where there is no beam due to dump
#         data = data[data.max(axis=1) > 0.00001]
#         data = self.normalize(data)
#         return data


# class RollingWindowSumNoDump(RollingWindowSum):
#     def __init__(self, BLM_filter, window_size="60s", min_periods=60):
#         super().__init__(BLM_filter)
#         self.window_size = window_size
#         self.min_periods = min_periods

#     def normalize(self, data):
#         # normalize to the max BLM signal
#         return data.divide(data.max(axis=1), axis=0)

#     def preprocess(self, path_to_hdf):
#         path_to_hdf = Path(path_to_hdf)
#         fill_number = int(path_to_hdf.stem)
#         # print(f'Loading {path_to_hdf}.')
#         data, meta = self.load_raw(path_to_hdf)
#         filtered_blms = self.blm_filter(data.columns)
#         data = data[filtered_blms]
#         # remove data where there is no beam due to dump
#         data = data[data.max(axis=1) > 0.0001]
#         beam_mode = data.index.get_level_values("mode")[0]
#         data = (
#             data.droplevel("mode")
#             .rolling(self.window_size, min_periods=self.min_periods)
#             .sum()
#             .dropna()
#         )

#         data = self.normalize(data)
#         # readd beam_mode and fill number
#         # very ugly
#         data = pd.concat([data], keys=[beam_mode], names=["mode"])
#         data = pd.concat([data], keys=[fill_number], names=["fill_number"])

#         return data
