# %%
import os
import sys
from pathlib import Path

import pandas as pd

root_dir = Path(os.getcwd()).parent
config_path = root_dir.joinpath("config/")
sys.path.append(str(root_dir))

from src .utils import get_config, get_config_filenames
from src.query_engine import QueryEngine

%load_ext autoreload
%autoreload 2

# Get configuration parameters and logger for the current session
config = get_config(path=config_path)
config = get_config(["config.ini"], path=config_path)
config = get_config(get_config_filenames(config["params"]["env"], config_path))


# Start DuckDB query engine
qe = QueryEngine(config)

# %%
save_dir = os.path.join(root_dir, "data/04_results")
report_types = ["", "adult", "bite"]
entities = ["Total", "Italy", "Spain", "Netherlands", "Hungary"]

for entity in entities:
    for report_type in report_types:
        if report_type != "":
            filename = f"user_retention_monthly_{report_type}_{entity}.parquet"
        else:
            filename = f"user_retention_monthly_{entity}.parquet"
        qe.user_retention(
            freq="month",
            scale="name_gadm_level0",
            entity=entity,
            report_type=report_type,
        ).to_parquet(
            os.path.join(
                save_dir,
                filename,
            ),
            index=True,
        )
    qe.overview_sampling_effort(
        freq="month",
        scale="name_gadm_level0",
        entity=entity,
    ).to_parquet(
        os.path.join(
            save_dir,
            f"human_sensor_coverage_monthly_{entity}.parquet",
        ),
        index=True,
    )
# %%
