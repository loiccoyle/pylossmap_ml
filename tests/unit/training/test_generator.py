import json
from unittest import TestCase
from pathlib import Path
from shutil import rmtree

import pandas as pd

from pylossmap_ml.training.generator import DataGenerator


class TestDataGenerator(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.test_data_dir = Path(__file__).parents[2] / "data"
        cls.sample_file = cls.test_data_dir / "test_sample.h5"
        cls.blm_name_file = cls.test_data_dir / "test_sample_blm_names.json"
        cls.blm_dcum_file = cls.test_data_dir / "blm_dcum.h5"
        with open(cls.blm_name_file, "r") as fp:
            cls.blm_names = json.load(fp)
        cls.blm_dcum = pd.read_hdf(cls.blm_dcum_file, "data")
        cls.test_dir = Path("test_generator")
        if not cls.test_dir.is_dir():
            cls.test_dir.mkdir(parents=True)

    def setUp(self) -> None:
        self.generator_pd = DataGenerator(
            self.sample_file,
            key="STABLE",
            batch_size=256,
            BLM_dcum=self.blm_dcum,  # type: ignore
            BLM_names=self.blm_names,
            return_dataframe=True,
        )
        self.generator_npy = DataGenerator(
            self.sample_file,
            key="STABLE",
            batch_size=256,
            BLM_dcum=self.blm_dcum,  # type: ignore
            BLM_names=self.blm_names,
            return_dataframe=False,
            ndim=3,
        )

    def test_init(self):
        DataGenerator(
            self.sample_file,
            key="STABLE",
            batch_size=256,
            BLM_dcum=self.blm_dcum,  # type: ignore
            BLM_names=self.blm_names,
        )

    def test_len(self):
        assert len(self.generator_pd) == 1024 / 256
        assert len(self.generator_npy) == 1024 / 256

    def test_to_json(self):
        json_file = self.test_dir / "generator_params.json"
        json_string = self.generator_pd.to_json(json_file)
        assert DataGenerator.from_json(json_file).to_json() == json_string

    def test_getitem_len(self):
        for data, _ in self.generator_pd:
            assert len(data) == self.generator_pd.batch_size
        for data, _ in self.generator_npy:
            assert len(data) == self.generator_npy.batch_size

    def test_getitem_sort(self):
        data, _ = self.generator_pd[0]
        assert data.columns.to_list() == self.generator_pd._blm_sorted

    def test_getitem_normalization(self):
        for data, _ in self.generator_pd:
            assert (data.max() <= 1).all()
            assert (data.min() >= 0).all()

        for data, _ in self.generator_npy:
            assert (data.max() <= 1).all()
            assert (data.min() >= 0).all()

    def test_getitem_ndim(self):
        data, _ = self.generator_npy[0]
        assert data.ndim == self.generator_npy.ndim

    def test_to_dict(self):
        self.generator_npy.to_dict()
        self.generator_pd.to_dict()

    def test_get_metadata(self):
        index = self.generator_npy.get_metadata()
        assert len(index) == self.generator_npy._data_len

    def test_split(self):
        ratio = 0.25
        generator_1, generator_2 = self.generator_npy.split(ratio)
        assert len(generator_1.indices) == int(len(self.generator_npy.indices) * ratio)
        assert len(generator_2.indices) == int(
            len(self.generator_npy.indices) * (1 - ratio)
        )
        # make sure there are no common data samples
        assert set(generator_1.indices) & set(generator_2.indices) == set()

    def tearDown(self) -> None:
        # clean closing of open files.
        del self.generator_pd
        del self.generator_npy

    @classmethod
    def tearDownClass(cls) -> None:
        rmtree(cls.test_dir)
