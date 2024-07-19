# %%
from pathlib import Path
import os, sys
import duckdb
from functools import partial
import prql_python

%load_ext autoreload
%autoreload 2

root_dir = Path(os.getcwd()).parent
sys.path.append(str(root_dir))

from src import ppa, utils

config = utils.get_config(path=root_dir)
qe_ppa = ppa.QueryEngine(config)

freq = "month"
min_samples = 10
report_type: str = ""

reports = duckdb.from_parquet(config["paths"]["reports_transf"])


def labels_filter(report_type: str = ""):
    if report_type != "":
        return f'filter report_type == "{report_type}"'
    else:
        return ""


prql = f"""
from reports
{labels_filter(report_type)}
derive date_base = s"date_trunc('{freq}', upload_date_utc)"
select {{user_id, version_uuid, upload_date_utc, date_base, lat, lon}}
"""

prql2sql = partial(
    prql_python.compile, options=prql_python.CompileOptions(target="sql.duckdb")
)
df_users = duckdb.sql(prql2sql(prql)).to_df().set_index("user_id")
ds_count = df_users.value_counts("user_id").sort_values()
users_id_one = ds_count[ds_count == 1].index.to_list()
users_id_high = ds_count[ds_count > 1].index.to_list()
users_id_high10 = ds_count[ds_count >= 10].index.to_list()

point_pattern_analysis_ = partial(
    ppa.point_pattern_analysis_logging, df_users=df_users, min_samples=min_samples
)

# %%
ppa.point_pattern_analysis(users_id_high10[-1], df_users, plot_fig=True)
# %%
