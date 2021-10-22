import json
from unittest import TestCase
from pathlib import Path

from pylossmap_ml.training.generator import DataGenerator


class TestDataGenerator(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.test_data_dir = Path(__file__).parents[2] / "data"
        cls.sample_file = cls.test_data_dir / "test_sample.h5"
        cls.blm_name_file = cls.test_data_dir / "test_sample_blm_names.json"
        with open(cls.blm_name_file, "r") as fp:
            cls.blm_names = json.load(fp)

    def test_init(self):
        generator = DataGenerator(
            self.sample_file,
            key="STABLE",
            batch_size=256,
            BLM_names=self.blm_names,
        )
