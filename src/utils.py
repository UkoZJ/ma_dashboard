import json
import os
from configparser import BasicInterpolation, ConfigParser
from glob import glob
from pathlib import Path
from typing import List, Literal, Optional

import pandas as pd


def get_config(
    config_files: List[str] = ["config.ini", ".config.ini"], path: str = "./"
):
    """Get configuration parameters"""

    path_config = [os.path.join(path, x) for x in config_files]

    config = ConfigParser(interpolation=BasicInterpolation())
    config.read(path_config)

    assert (
        len(config.sections()) != 0 or len(config.defaults().keys()) != 0
    ), f"Configuration file is empty! Check path: {path_config}"

    return config


def get_config_filenames(env: Literal["dev", "prod"], root_dir: str = "./"):
    """
    Get all files in the config folder structure relative to production or development environment
    """
    root_files = glob(os.path.join(root_dir, "*.ini"))
    sub_files = glob(os.path.join(root_dir, env, "*.ini"))
    sub_hidden_files = glob(os.path.join(root_dir, env, ".*.ini"))
    return root_files + sub_files + sub_hidden_files


def check_null_coords(df: pd.DataFrame):
    """
    Check if there are any null values for a given coordinate and impute them
    to avoid loosing records while applying xarrays
    """
    null_coords = df.columns[df.count().values < len(df)]
    if len(null_coords) >= 1:
        print(f"Check for Null values in coordinates: {null_coords.values}")


def get_dict(config: dict, key_asint=False, value_astype=None):
    """
    Get a dictionary from a configuration argument
    """

    if key_asint:
        d = json.loads(
            config, object_pairs_hook=lambda pairs: {int(k): v for k, v in pairs}
        )
    else:
        d = json.loads(config)

    if value_astype is not None:
        return {k: value_astype(v) for k, v in d.items()}
    else:
        return d


def get_common_time(df1: pd.DataFrame, df2: Optional[pd.DataFrame] = None, freq="D"):
    """
    Get the common time range for two dataframes with the same resampling frequency
    """

    if df2 is not None:
        min_time = min(df1.min(), df2.min())
        max_time = max(df1.max(), df2.max())

        return pd.date_range(min_time, max_time, freq=freq)
    else:
        return pd.date_range(df1.min(), df1.max(), freq=freq)
