import logging
import math
from abc import abstractmethod
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .preprocessor import BasePreprocessor

# TODO: add docstrings to all of these


class BaseSpooler:
    def __init__(
        self, preprocessor: BasePreprocessor, raw_files: List[Path], output: Path
    ):
        self._log = logging.getLogger(__name__)
        self.preprocessor = preprocessor
        self.raw_files = raw_files
        self.output = output
        if self.output.exists():
            self._log.warning("'%s' already exists.", self.output.resolve())

    @abstractmethod
    def spool(self) -> None:
        pass


# class MpSpooler(BaseSpooler):
#     def __init__(
#         self,
#         preprocessor: BasePreprocessor,
#         raw_files: List[Path],
#         output: Path,
#         n_jobs: int = -1,
#     ):
#         super().__init__(preprocessor, raw_files, output)
#         if n_jobs == -1:
#             n_jobs = cpu_count()
#         self.n_jobs = n_jobs

#     def spool(self):
#         for g in tqdm(
#             grouper(self.raw_files, self.n_jobs),
#             total=math.ceil(len(self.raw_files) / self.n_jobs),
#             desc="Preprocessing raw data",
#         ):
#             with Pool(self.n_jobs) as pool:
#                 out = list(
#                     pool.map(
#                         self.preprocessor.preprocess, filter(lambda x: x is not None, g)
#                     )
#                 )

#             out = [f for f in out if f is not None]
#             for o in out:
#                 o.to_hdf(
#                     self.output,
#                     o.index.get_level_values("mode")[0],
#                     format="table",
#                     append=True,
#                 )


# class SerialNumpy(BaseSpooler):
#     def __init__(
#         self, preprocessor: BasePreprocessor, raw_files: List[Path], output_dir: Path
#     ):
#         super().__init__(preprocessor, raw_files, output_dir)
#         if not self.output.is_dir():
#             self.output.mkdir(parents=True)


#     def spool(self, **_):
#         for file_path in tqdm(self.raw_files):
#             out_filepath = self.output / (file_path.stem + ".npy")
#             if out_filepath.is_file():
#                 self._log.warning("%s exists.", out_filepath)
#                 continue
#             output = self.preprocessor.preprocess(file_path)
#             if output is None:
#                 self._log.warning("%s gives none.", out_filepath)
#                 continue
#             output = output.to_numpy()
#             try:
#                 np.save(out_filepath, output)
#                 self._log.info("DONE %s", file_path)
#             except Exception as exc:
#                 self._log.error("failed: %s", file_path)
#                 self._log.error(exc)


class SerialSingleH5(BaseSpooler):
    def spool(self, **kwargs) -> None:
        for file_path in tqdm(self.raw_files):
            fill_number = file_path.stem
            with pd.HDFStore(self.output) as store:
                # print(store.keys())
                if "/" + fill_number in store.keys():
                    self._log.warning("%s already in %s", fill_number, self.output)
                    continue
                # print(fill_number)
                output = self.preprocessor.preprocess(file_path)
                if output is None:
                    continue
                try:
                    output.to_hdf(
                        self.output,
                        key=fill_number,
                        format="table",
                        append=True,
                        **kwargs,
                    )
                    self._log.info("DONE %s", file_path)
                except Exception as exc:
                    self._log.error("failed: %s", file_path)
                    self._log.error(exc)

    def concat(self, concat_path: Path, key: str = "data", **kwargs) -> None:
        with pd.HDFStore(self.output, **kwargs) as store:
            for key in tqdm(store.keys()):
                df = store[key]
                self._log.info("key: %s", key)
                self._log.info("key shape: %s", df.shape)
                df.to_hdf(concat_path, key, format="table", append=True, **kwargs)
                self._log.info("DONE %s", key)


class SerialH5(BaseSpooler):
    def __init__(
        self, preprocessor: BasePreprocessor, raw_files: List[Path], output_dir: Path
    ):
        super().__init__(preprocessor, raw_files, output_dir)
        if not self.output.is_dir():
            self.output.mkdir(parents=True)

    def spool(self, key: str = "data", **kwargs) -> None:
        for file_path in tqdm(self.raw_files):
            out_filepath = self.output / (file_path.stem + ".h5")
            if out_filepath.is_file():
                self._log.warning("%s exists.", out_filepath)
                continue
            output = self.preprocessor.preprocess(file_path)
            if output is None:
                self._log.warning("%s gives none.", out_filepath)
                continue
            try:
                output.to_hdf(out_filepath, key, format="table", **kwargs)
                self._log.info("DONE %s", file_path)
            except Exception as exc:
                self._log.error("failed: %s", file_path)
                self._log.error(exc)

    def concat(self, concat_path: Path, key: str = "data", **kwargs) -> None:
        for h5_file in tqdm(list(self.output.glob("*.h5"))):
            # TODO: check is I need to specify the compression here or does pandas figure it out?
            data = pd.read_hdf(h5_file, "data")
            try:
                data.to_hdf(concat_path, key, format="table", append=True, **kwargs)
                self._log.info("DONE %s", h5_file)
            except Exception as exc:
                self._log.error("failed: %s", h5_file)
                self._log.error(exc)


# class SerialSingleCSV(SerialSingleH5):
#     def spool(self, **_):
#         for file_path in tqdm(self.raw_files):
#             output = self.preprocessor.preprocess(file_path)
#             if output is None:
#                 self._log.warning("%s gives none.", file_path)
#                 continue
#             try:
#                 output.to_csv(self.output, mode="a", header=False)
#                 self._log.info("DONE %s", file_path)
#             except Exception as exc:
#                 self._log.error("failed: %s", file_path)
#                 self._log.error(exc)
