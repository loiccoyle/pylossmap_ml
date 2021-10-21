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


class SerialNumpy(BaseSpooler):
    def __init__(
        self, preprocessor: BasePreprocessor, raw_files: List[Path], output_dir: Path
    ):
        super().__init__(preprocessor, raw_files, output_dir)
        if not self.output.is_dir():
            self.output.mkdir(parents=True)

    def spool(self):
        for file_path in tqdm(self.raw_files):
            out_filepath = self.output / (file_path.stem + ".npy")
            if out_filepath.is_file():
                print(f"{out_filepath} exists.")
                continue
            output = self.preprocessor.preprocess(file_path).to_numpy()
            if output is None:
                print(f"{out_filepath} gives none.")
                continue
            try:
                np.save(out_filepath, output)
                print(f"DONE {file_path}")
            except Exception as exc:
                print(file_path)
                print(exc)


class SerialSingleH5(BaseSpooler):
    def __init__(
        self,
        preprocessor: BasePreprocessor,
        raw_files: List[Path],
        output: Path,
        key: Optional[str] = None,
    ):
        """For some reason when using multiple processes, `pytimber.get`
        freezes, this serial spooler works.
        """
        super().__init__(preprocessor, raw_files, output)
        self.key = key

    def spool(self, *args, **kwargs):
        for file_path in tqdm(self.raw_files):
            fill_number = file_path.stem
            with pd.HDFStore(self.output) as store:
                # print(store.keys())
                if "/" + fill_number in store.keys():
                    print(f"{fill_number} already in {self.output}")
                    continue
                # print(fill_number)
                output = self.preprocessor.preprocess(file_path)
                if output is None:
                    continue
                if self.key is not None:
                    key = self.key
                else:
                    key = fill_number

                try:
                    output.to_hdf(
                        self.output,
                        key=key,
                        format="table",
                        append=True,
                        *args,
                        **kwargs,
                    )
                    print(f"DONE {file_path}")
                except Exception as exc:
                    print(file_path)
                    print(exc)


class SerialH5(SerialNumpy):
    def spool(self, **kwargs):
        for file_path in tqdm(self.raw_files):
            out_filepath = self.output / (file_path.stem + ".h5")
            if out_filepath.is_file():
                print(f"{out_filepath} exists.")
                continue
            output = self.preprocessor.preprocess(file_path)
            if output is None:
                print(f"{out_filepath} gives none.")
                continue
            try:
                output.to_hdf(out_filepath, "data", format="table", **kwargs)
                print(f"DONE {file_path}")
            except Exception as exc:
                print(file_path)
                print(exc)


class SerialSingleCSV(SerialSingleH5):
    def spool(self):
        for file_path in tqdm(self.raw_files):
            output = self.preprocessor.preprocess(file_path)
            if output is None:
                print(f"{file_path} gives none.")
                continue
            try:
                output.to_csv(self.output, mode="a", header=False)
                print(f"DONE {file_path}")
            except Exception as exc:
                print(file_path)
                print(exc)
