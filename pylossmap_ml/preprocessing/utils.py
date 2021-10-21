from itertools import zip_longest
from typing import Any, Iterable

import numpy as np
import pandas as pd

INTENSITY = "LHC.BCTFR.A6R4.B{beam}:BEAM_INTENSITY"


def grouper(iterable: Iterable, n: int, fillvalue: Any = None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def timber_to_df(t_dict: dict, key: str):
    out = t_dict[key]
    out = pd.DataFrame(np.vstack(out).T, columns=["timestamp", key])
    out["timestamp"] = pd.to_datetime(
        out["timestamp"], unit="s", utc=True
    ).dt.tz_convert("Europe/Zurich")
    out.set_index("timestamp", inplace=True)
    return out


def dataframe_like(array, df):
    """Convert `array` to a `pd.DataFrame` using the index and columns of `df`."""
    return pd.DataFrame(array, columns=df.columns, index=df.index)
