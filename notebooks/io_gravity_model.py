# %%
# Reports only relative to Spain feed as input to a gravity model

import os
import sys
from pathlib import Path

import pandas as pd

root_dir = Path(os.getcwd()).parent
sys.path.append(str(root_dir))

from src import utils

config = utils.get_config(path=root_dir)

reports = pd.read_parquet(config["paths"]["reports_transf"])
reports_codes = pd.read_parquet(config["paths"]["reports_codes"], columns=["code_gadm"])
gadm_legend = pd.read_parquet(config["paths"]["gadm_legend"])
reports_codes_spa = (
    pd.merge(
        reports_codes.reset_index(),
        gadm_legend.reset_index(),
        on="code_gadm",
        how="left",
    )
    .query("name_gadm_level0 == 'Spain'")
    .set_index("version_uuid")
)
reports_spa = reports.set_index("version_uuid").loc[reports_codes_spa.index]
reports_spa.to_csv("./reports_spa.csv", sep=",", index=True)
