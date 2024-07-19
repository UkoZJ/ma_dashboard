# %%
import warnings
from typing import Literal, Optional

import geopandas as gpd
import pandas as pd
import pandera as pa
from pandera.typing import DataFrame, Index, Series
from pandera.typing.geopandas import GeoSeries
from pyogrio import read_dataframe
from configparser import ConfigParser
from logging import Logger

warnings.simplefilter(action="ignore", category=UserWarning)


class GeoPoly(pa.DataFrameModel):
    """Schema for geo-polygons table."""

    code: Index[str]
    name: Optional[Series[str]]
    level: Optional[Series[int]]
    level_name: Optional[Series[str]]
    geometry: GeoSeries

    # Only for country level
    # @pa.check("code")
    # def check_code_iso2(cls, code: Index[str]) -> Series[bool]:
    #     return code.apply(lambda s: len(s) == 2)


def reverse_geo_static(countries, lat, lon, crs="EPSG:4326"):
    try:
        coordinates = gpd.points_from_xy([lon], [lat], crs=crs)

        if coordinates.is_valid:
            gdf = gpd.GeoDataFrame(geometry=coordinates)
            coord2country = gpd.sjoin_nearest(
                gdf, countries, max_distance=None, distance_col="nearest_dist"
            )
            geocode = coord2country["countryCode"][0]
        else:
            geocode = ""
    except:
        geocode = ""

    return geocode


def get_records(df):
    """
    Get records lat/lon from table
    """

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(
            df["lon"],
            df["lat"],
            crs="EPSG:4326",
        ),
        crs="EPSG:4326",
        copy=True,
    )

    return gdf


class ReportsCodes:

    def __init__(self, config: ConfigParser, logger: Optional[Logger] = None) -> None:

        self.config = config
        self.is_logger = isinstance(logger, Logger)
        if self.is_logger:
            self.logger = logger

    @pa.check_types()
    def get_geopoly(
        self,
        level: Literal["gadm", "climate_regions", "ecoregions"] = "gadm",
    ) -> DataFrame[GeoPoly]:
        """Return polygons with location labels for different geo-levels.

        Parameters
        ----------
        level, optional
            Geo-political level, by default "country"
        union, optional
            At country level merge some polygons together, by default True

        Returns
        -------
            Table with location codes, names, administrative level (optional) and polygons.
        """

        level_cfg = {
            "gadm": {
                "path": self.config["paths"]["gadm_410"],
                "read_rename_col": {"UID": "code"},
                "get_level": None,
                "select_cols": ["code", "geometry"],
            },  # https://gadm.org/download_world.html (transformed with QGIS)
            "climate_regions": {
                "path": self.config["paths"]["climate_regions"],
                "read_rename_col": {"DN": "code"},
                "get_level": None,
                "select_cols": ["code", "name", "geometry"],
            },  # https://www.gloh2o.org/koppen/
            "ecoregions": {
                "path": self.config["paths"]["ecoregions"],
                "read_rename_col": {
                    "ECO_ID": "code",
                    "ECO_NAME": "name",
                    "BIOME": "level",
                    "biomes_lab": "level_name",
                },
                "get_level": None,
                "select_cols": ["code", "name", "level", "level_name", "geometry"],
            },  # https://wwf.to/2EWMBXq
        }

        dtypes = {"code": str, "name": str, "level": int, "level_name": str}

        level_ = level_cfg[level]
        geopoly = read_dataframe(
            level_["path"], columns=list(level_["read_rename_col"].keys())
        ).rename(columns=level_["read_rename_col"])

        if level_["get_level"] is not None:
            geopoly = geopoly.query(level_["get_level"])

        isnull_geometry = geopoly["geometry"].isna()
        if isnull_geometry.sum() > 0:
            if self.is_logger:
                self.logger.warning(
                    f"Geo-layer '{level}' has {isnull_geometry.sum()} missing geometries. Dropping relative rows."
                )
            geopoly = geopoly[~isnull_geometry]

        # Hard-coded changes
        if level == "gadm":
            geopoly = geopoly.replace("", None)

        if level == "climate_regions":
            legend = pd.read_csv(
                self.config["paths"]["climate_regions_legend"], sep=";"
            )
            geopoly = geopoly.merge(legend, left_on="code", right_on="code", how="left")

        if level_["select_cols"] is not None:
            dtypes_ = {
                i: dtypes[i]
                for i in set(level_["select_cols"]).intersection(set(dtypes.keys()))
            }
            geopoly = geopoly[level_["select_cols"]].astype(dtypes_)

        geopoly = geopoly.set_index("code")

        geopoly.attrs["level"] = level
        geopoly.attrs["path"] = level_["path"]

        return geopoly

    def add_code(
        self,
        records: pd.DataFrame,
        geopoly: DataFrame[GeoPoly],
        max_distance: Optional[float] = None,
        drop_duplicates_on: Optional[str] = None,
    ) -> pd.DataFrame:
        """Assign location label to a point that falls within or close to a polygon.

        Parameters
        ----------
        records
            Table where each record has a lat/lon geometry point
        geopoly
            Table with location labels related to geo-polygons
        max_distance, optional
            Maximum distance to include a point into the nearest polygon, if zero
            then only within points are included, by default None
        drop_duplicates_on, optional
            Name of the column on which to apply drop duplicates, by default None

        Returns
        -------
            Records table with location labels
        """

        # Much faster than sjoin_nearest
        df_inside = gpd.sjoin(
            records,
            geopoly,
            how="inner",
        )

        outside = ~records.index.isin(df_inside.index)

        if outside.any() and max_distance != 0:
            # Slower, points that are not within any polygon are put in the nearest one
            df_outside = gpd.sjoin_nearest(
                records[outside], geopoly, how="left", max_distance=max_distance
            )

            df_sjoin = pd.concat([df_inside, df_outside], axis=0)

            if self.is_logger:
                self.logger.info(
                    f"Level: \n{geopoly.attrs['level']} | Number of records that fall out of polygon borders: {len(df_outside)}"
                )
        else:
            df_sjoin = df_inside

        df_sjoin = df_sjoin.rename({"index_right": "code"}, axis=1).drop(
            ["geometry"], axis=1
        )
        if drop_duplicates_on is not None:
            df_sjoin = df_sjoin.drop_duplicates(subset=drop_duplicates_on)

        return df_sjoin

    def get_codes(
        self, reports: pd.DataFrame, save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """Pipeline to add all location labels to the reports table

        Parameters
        ----------
        reports
            MA reports with 'lat' and 'lon' coordinates for each ID.
        save_path
            Save table as parquet file.

        Returns
        -------
            MA reports with location labels
        """

        latlon = get_records(reports[["lat", "lon"]])["geometry"].reset_index()

        reports_gadm = (
            self.add_code(
                latlon,
                self.get_geopoly(level="gadm"),
                drop_duplicates_on="version_uuid",
            )
            .set_index("version_uuid")
            .rename(columns={"code": "code_gadm"})
        )

        reports_climate_regions = (
            self.add_code(latlon, self.get_geopoly("climate_regions"))
            .set_index("version_uuid")
            .rename(
                columns={
                    "code": "code_climate_regions",
                    "name": "name_climate_regions",
                    "level": "level_climate_regions",
                }
            )
        )

        reports_ecoregions = (
            self.add_code(latlon, self.get_geopoly("ecoregions"))
            .set_index("version_uuid")
            .rename(
                columns={
                    "code": "code_ecoregions",
                    "name": "name_ecoregions",
                    "level": "code_biome",
                    "level_name": "name_biome",
                }
            )
        )
        layers = [
            reports_gadm,
            reports_climate_regions,
            reports_ecoregions,
        ]
        try:
            reports_codes = pd.concat(
                layers,
                join="outer",
                axis=1,
            )
        except:
            if self.is_logger:
                self.logger.warning(
                    "Dropping duplicates in geo-layers to complete the operation of concatenation."
                )
            reports_codes = pd.concat(
                [df[~df.index.duplicated()] for df in layers],
                join="outer",
                axis=1,
            )

        if save_path is not None:
            reports_codes.to_parquet(save_path)

        return reports_codes
