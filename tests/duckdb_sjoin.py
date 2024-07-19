# %%
# At the moment, there are no spatial join indexes in DuckDB and thus the execution
# is extremly slow. But watch out for new versions of DuckDB.

import duckdb

con = duckdb.connect(database=":memory:")
reports_path = "/home/uko/Dev/ma_dashboard/data/01_raw/staging/reports.parquet"
geo_path = "/home/uko/Dev/ma_dashboard/data/01_raw/geo/gadm_original/modified/gadm_410_transformed.gpkg"

# %%

con.execute(
    f"""
INSTALL spatial;
INSTALL parquet;
LOAD spatial;
LOAD parquet;

CREATE TABLE reports AS SELECT * FROM '{reports_path}';
CREATE TABLE geo AS SELECT UID, geom FROM ST_Read('{geo_path}');
CREATE TABLE latlon AS SELECT version_uuid, ST_Point(lon, lat) AS point FROM reports;

"""
)

# %%
con.execute("UPDATE geo SET geom = ST_MakeValid(geom) WHERE NOT ST_IsValid(geom);")
con.sql("SELECT * FROM geo WHERE NOT ST_IsValid(geom);")

# %%
# Reduce the number of points
con.execute("DELETE FROM latlon WHERE rowid > 100;")

# %%
con.execute(
    """
CREATE TABLE reports_geo AS 
SELECT * FROM latlon 
JOIN geo AS g 
ON ST_Within(point, g.geom);
"""
)

# %%
con.sql("select version_uuid, UID from reports_geo")

# %%
