import json
from unittest import TestCase
from pathlib import Path

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
        cls.generator = DataGenerator(
            cls.sample_file,
            key="STABLE",
            batch_size=256,
            BLM_dcum=cls.blm_dcum,  # type: ignore
            BLM_names=cls.blm_names,
        )

    def test_init(self):
        generator = DataGenerator(
            self.sample_file,
            key="STABLE",
            batch_size=256,
            BLM_dcum=self.blm_dcum,  # type: ignore
            BLM_names=self.blm_names,
        )

    def test_len(self):
        assert len(self.generator) == 1024 / 256
