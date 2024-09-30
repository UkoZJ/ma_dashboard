# %%
import os
import sys
from pathlib import Path

import duckdb
import pandas as pd

root_dir = Path(os.getcwd()).parent
config_path = root_dir.joinpath("config/")
sys.path.append(str(root_dir))

from src.query_engine import QueryEngine
from src.utils import get_config, get_config_filenames

%load_ext autoreload
%autoreload 2

# Get configuration parameters and logger for the current session
config = get_config(path=config_path)
config = get_config(["config.ini"], path=config_path)
config = get_config(get_config_filenames(config["params"]["env"], config_path))


# Start DuckDB query engine
qe = QueryEngine(config)

# %%
freq = "month"
df_pv = qe.filter_view(
    freq=freq, scale="name_gadm_level0", report_type="adult", entity="Spain"
)
df_pv.plot(kind="area", stacked=True)
df_pv

# %%
df = qe.filter_view_labels(scale="name_gadm_level0", report_type="adult")
df
# %%
df = qe.overview_rank(scale="name_gadm_level0")
df
# %%
df = qe.overview_view_inflow(freq="day", scale="name_gadm_level0", entity="Total")
df
# %%
report_type_tot = qe.overview_view_inflow(
    scale="name_gadm_level0", entity="Spain"
).sum()
users_tot = qe.user_activity()[["new", "active"]].sum()
kpi = pd.concat([report_type_tot, users_tot], axis=0)
kpi
# %%
df = qe.user_retention(
    freq="month", scale="name_gadm_level0", entity="Spain", report_type="adult"
)
# df = qe.user_retention(
#     freq="month", scale="name_gadm_level4", entity="L'Hospitalet de Llobregat | Bajo Llobregat | Barcelona | Cataluna | Spain", report_type="adult"
# )
df.plot()
# %%
l = []
for rt in ["", "adult", "bite", "site"]:
    df = qe.user_retention(
        freq="month", scale="name_gadm_level0", entity="Spain", report_type=rt
    )
    df = df.rename(columns={"user_retention": rt})
    l.append(df)

df_ur = pd.concat(l, axis=1)
df_ur.plot()
# %%

df = qe.user_activity()
df
# %%
df = qe.user_retention_stats(freq="month", entity="Total")
df[["new_perc", "active_perc", "user_retention"]].plot()

# %%
# df = qe.overview_sampling_effort(scale="name_gadm_level0", entity="Spain", freq="month")
df = qe.overview_sampling_effort(
    scale="name_gadm_level4",
    entity="L'Hospitalet de Llobregat | Bajo Llobregat | Barcelona | Cataluna | Spain",
)
# df = qe.overview_sampling_effort()
ax = df.plot()
# ax.set_ylim([0, 3])

# %%
users_quality = duckdb.from_parquet(config["paths"]["users_quality"])

# %%
users_quality.to_df()[["name_gadm_level0", "quality_reports"]].groupby(
    "name_gadm_level0"
).mean().sort_values("quality_reports")
# %%
