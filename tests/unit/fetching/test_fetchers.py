from unittest import TestCase
from pathlib import Path
from shutil import rmtree

import pandas as pd

from pylossmap_ml import fetching


class TestFetching(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.test_dir = Path("test_fetching")
        if not cls.test_dir.is_dir():
            cls.test_dir.mkdir(parents=True)

    def test_fetch(self) -> None:
        destination_dir = self.test_dir / "fetch"
        fetching.fetch(
            pd.to_datetime("2018-07-22 00:00:00").tz_localize("Europe/Zurich"),
            pd.to_datetime("2018-07-23 00:00:00").tz_localize("Europe/Zurich"),
            destination_dir,
        )
        assert (destination_dir / ".dataset_info.json").is_file()
        assert len(list(destination_dir.glob("*.h5"))) == 2

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.test_dir.is_dir():
            rmtree(cls.test_dir)
