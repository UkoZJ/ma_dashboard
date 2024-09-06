import os
import subprocess
from urllib.parse import urlparse, quote_plus
from configparser import ConfigParser
from functools import partial
from typing import Callable, Dict, Literal, Optional, Tuple, Any
from logging import Logger

import duckdb
import numpy as np
import pandas as pd
import prql_python
import rasterio
from pyogrio import read_dataframe
from sqlalchemy import create_engine
from sshtunnel import SSHTunnelForwarder
from unidecode import unidecode

from correct_gadm import correct
from geocoding import ReportsCodes
from log import task_logging


Dataset = Dict[str, pd.DataFrame]
TableLoader = Callable[[dict], Dataset]
CONN_MODE = Literal["local", "remote"]
SQL_ENGINE = Literal["sqlalchemy", "duckdb"]

prql2sql_postgres = partial(
    prql_python.compile, options=prql_python.CompileOptions(target="sql.postgres")
)
prql2sql_duckdb = partial(
    prql_python.compile, options=prql_python.CompileOptions(target="sql.duckdb")
)


def open_conn(
    config: ConfigParser,
    conn_type: CONN_MODE = "local",
    sql_engine: SQL_ENGINE = "duckdb",
    logger: Optional[Logger] = None,
) -> Tuple[Any, Optional[SSHTunnelForwarder]]:
    """
    Open connection with the DB
    """

    def create_uri(user: str, password: str, host: str, port: int, dbname: str) -> str:
        encoded_password = quote_plus(password)
        return f"postgresql://{user}:{encoded_password}@{host}:{port}/{dbname}"

    def set_postgres_env(uri:str):
        """Setup postgres environment variables to start a DuckDB connection"""

        parsed = urlparse(uri)
        env_vars = {
            'PGUSER': parsed.username,
            'PGPASSWORD': parsed.password,
            'PGHOST': parsed.hostname,
            'PGPORT': str(parsed.port),
            'PGDATABASE': parsed.path[1:]
        }
        os.environ.update(env_vars)
        export_commands = [f"export {key}={value}" for key, value in env_vars.items()]
        command_string = "; ".join(export_commands)
        subprocess.run(command_string, shell=True, executable="/bin/bash")

    def create_engine_conn(
        uri: str, engine_type: str, is_env:bool=True) -> Any:
        if engine_type == "sqlalchemy":
            return create_engine(uri, echo=False, hide_parameters=True)
        elif engine_type == "duckdb":
            engine = duckdb.connect()
            if is_env:
                set_postgres_env(uri)
                engine.execute(f"ATTACH '' AS db (TYPE postgres, READ_ONLY);")
            else:
                engine.execute(f"ATTACH '{uri}' AS db (TYPE postgres, READ_ONLY);")
            return engine
        else:
            raise ValueError(f"Unsupported SQL engine: {engine_type}")

    if conn_type == "remote":
        try:
            if logger:
                logger.info("Connecting to the remote DB...")

            server = SSHTunnelForwarder(
                (config["ssh"]["HOST"], config["ssh"].getint("PORT")),
                ssh_username=config["ssh"]["USER"],
                ssh_password=config["ssh"]["PASSWORD"],
                local_bind_address=(
                    config["ssh"]["LOCAL_HOST"],
                    config["ssh"].getint("LOCAL_PORT"),
                ),
                remote_bind_address=(
                    config["remote_db"]["HOST"],
                    config["remote_db"].getint("PORT"),
                ),
            )

            server.start()
            if logger:
                logger.info("SSH tunnel established.")

            uri = create_uri(
                config["remote_db"]["USER"],
                config["remote_db"]["PASSWORD"],
                config["ssh"]["LOCAL_HOST"],
                config["ssh"].getint("LOCAL_PORT"),
                config["remote_db"]["NAME"],
            )

            engine = create_engine_conn(uri, sql_engine)
            logger.info("Connected to the remote DB successfully.")

            return engine, server

        except Exception as e:
            if logger:
                logger.error(f"Connection to remote DB has failed! Error: {e}")
            if server:
                server.close()
            raise

    elif conn_type == "local":
        try:
            if logger:
                logger.info("Connecting to local DB...")

            uri = create_uri(
                config["local_db"]["USER"],
                config["local_db"]["PASSWORD"],
                config["local_db"]["HOST"],
                config["local_db"].getint("PORT"),
                config["local_db"]["NAME"],
            )

            engine = create_engine_conn(uri, sql_engine)
            if logger:
                logger.info("Connected to the local DB successfully.")

            return engine, None

        except Exception as e:
            logger.error(f"Connection to local DB has failed! Error: {e}")
            raise

    else:
        raise KeyError(
            f"Connection type '{conn_type}' not available, try 'local' or 'remote'."
        )


def prql_human_presence_discr(table: str):
    return f"""
        from {table}
        derive {{
            human_presence_discr = case [
                human_presence == 0 => 'human_absence',
                (human_presence > 0 && human_presence < 50) => 'low_density',
                (human_presence >= 50 && human_presence < 300) => 'medium_density',
                human_presence >= 300 => 'high_density']
            }}
    """


class Ingest:
    def __init__(self, config: ConfigParser, logger: Optional[Logger] = None) -> None:
        self.config = config
        self.logger = logger
        self.is_logger = isinstance(logger, Logger)
        if self.is_logger:
            # Apply task_logging decorator to ETL steps
            self.task_logging = partial(task_logging, self.logger)
            self.extract = self.task_logging()(self.extract)
            self.transform = self.task_logging()(self.transform)

    def extract(
        self,
        conn_type: CONN_MODE = "remote",
        sql_engine: SQL_ENGINE = "duckdb",
    ) -> Dataset:
        """
        Get raw data from remote or local DB
        """

        # Open connection with postgres
        engine, server = open_conn(self.config, conn_type, sql_engine, logger=self.logger)
        is_prql = self.config.getboolean("params", "is_prql")
        try:
            # Get all the tables relative to their query
            for table_name in self.config["sql"]:
                sql = self.config["sql"][table_name]

                match sql_engine:
                    case "sqlalchemy":
                        sql = prql2sql_postgres(sql) if is_prql else sql
                        pd.read_sql_query(sql, engine).to_parquet(
                            self.config["paths"][table_name]
                        )
                    case "duckdb":
                        sql = prql2sql_duckdb(sql) if is_prql else sql
                        engine.execute(
                            f"""COPY (SELECT * FROM postgres_query('db', '{sql.replace("'", "''")}')) TO '{self.config["paths"][table_name]}';"""
                        )
                if self.is_logger:
                    self.logger.info(f"Table successfully ingested: {table_name}")
        except IOError:
            if self.is_logger:
                self.logger.error("Connection to DB failed.")
        finally:
            # Close connection and ssh tunnel-port
            match sql_engine:
                case "sqlalchemy":
                    engine.dispose()
                case "duckdb":
                    engine.execute("DETACH db;")
            if server is not None:
                server.close()

    def transform(
        self, refresh_static_tables: bool = False, refresh_codes: bool = True
    ) -> None:
        legends_exist = os.path.isfile(
            self.config["paths"]["gadm_legend"]
        ) and os.path.isfile(self.config["paths"]["ecoregions_legend"])

        if refresh_static_tables or not legends_exist:
            # Compute only once since tables are static
            self.logger.info("Start building static tables...")
            self.transform_legends(label_correction=True, to_ascii=True)
            self.transform_human_presence()
            self.logger.info("Completed static tables.")

        self.logger.info("Start building temporary tables...")
        self.transform_reports()
        self.transform_reports_codes(refresh_all=refresh_codes)
        self.transform_app_users()
        self.transform_sampling_effort()
        self.transform_reports_response()
        self.logger.info("Completed temporary tables.")

    def transform_reports(self) -> None:
        """Transform reports table"""

        reports = duckdb.read_parquet(self.config["paths"]["reports"])
        iqa_scores = duckdb.read_parquet(self.config["paths"]["iqa_scores"])

        # Filter out spam and inconsistent reports
        prql_raw = """
            from reports
            filter (lat != null)
            derive labels = case [
            (simplified_expert_validation_result == "nosesabe" && expert_validated == 1 && hide == false) => "not-sure",
            (simplified_expert_validation_result == "nosesabe" && expert_validated == 0 && hide == false) => "unvalidated",
            (simplified_expert_validation_result == "noseparece" && hide == false) => "other_species",
            (simplified_expert_validation_result == "conflict" && hide == false) => "other_species",
            (simplified_expert_validation_result == "site" && hide == false) => "storm_drain",
            (simplified_expert_validation_result == "site#other" && hide == false) => "other_site",
            hide == true => "spam",
            true => simplified_expert_validation_result,
            ]
            derive confidence = case [
            expert_validation_confidence == "none" => 0,
            expert_validation_confidence == 1 => 0.7,
            expert_validation_confidence == 2 => 1.0,
            true => 0,
            ]
            derive bite_severity = case [
            bite_count <= 3 => "low",
            bite_count <= 10 => "high",
            bite_count > 10 => "extreme",
            ]
            derive storm_drain_type = case [
            storm_drain_status == "storm_drain_water" => "with_water",
            storm_drain_status == "storm_drain_dry" => "dry",
            true => null
            ]
            derive {
                observation_date_utc = s"observation_date::TIMESTAMP",
                upload_date_utc = s"server_upload_time::TIMESTAMP",
                validation_date_utc = s"max_last_modified::TIMESTAMP",
                validated = s"expert_validated::BOOL",
            }
            select !{
                observation_date, server_upload_time, max_last_modified,
                expert_validated, simplified_expert_validation_result,
                storm_drain_status
                }
            sort observation_date_utc
        """

        reports_ = duckdb.query(prql2sql_duckdb(prql_raw))

        base_select = """
            version_uuid, user_id, observation_date_utc, upload_date_utc, validation_date_utc,
            validated, lat, lon, labels, report_type, confidence, best_photo_filename
            """

        prql_adult = f"""
        from reports_
        filter (report_type == "adult" && n_photos > 0)
        select {{{base_select}}}
        """

        prql_bite = f"""
        from reports_
        filter (report_type == "bite")
        derive {{labels = `bite_location`,
        validation_date_utc = null}}
        select {{{base_select}}}
        """

        prql_site = f"""
        from reports_
        filter (report_type == "site")
        derive validation_date_utc = null
        select {{{base_select}}}
        """

        prql_concat = f"""
        {prql_adult}
        append ({prql_bite})
        append ({prql_site})
                """

        reports_transf = duckdb.query(prql2sql_duckdb(prql_concat))
        reports_transf.to_parquet(self.config["paths"]["reports_transf"])
        if self.is_logger:
            self.logger.info("Completed: reports_transf")

    def transform_legends(self, label_correction=True, to_ascii=True):
        """Build legend tables"""

        name = [f"NAME_{i}" for i in range(5)]
        gadm_legend = read_dataframe(
            self.config["paths"]["gadm_410"],
            columns=["UID"] + name,
            read_geometry=False,
        )

        if label_correction:
            correct(gadm_legend, self.config)

        gadm_legend = gadm_legend.rename(columns={"UID": "code_gadm"}).set_index(
            "code_gadm"
        )
        if to_ascii:
            gadm_legend[name] = gadm_legend[name].map(unidecode)

        gadm_legend.index = gadm_legend.index.astype("str")

        for i in range(5):
            gadm_legend[f"FULL_NAME_{i}"] = gadm_legend[
                [f"NAME_{j}" for j in range(i, -1, -1)]
            ].agg(" | ".join, axis=1)

        full_name = [f"FULL_NAME_{i}" for i in range(5)]
        gadm_legend_full_name = gadm_legend[full_name].copy()
        gadm_legend_full_name.columns = name
        gadm_legend_full_name[gadm_legend[name] == ""] = None

        gadm_legend_full_name = gadm_legend_full_name.rename(
            columns={n: f"name_gadm_level{i}" for i, n in enumerate(name)}
        )
        gadm_legend_full_name.to_parquet(
            self.config["paths"]["gadm_legend"], index=True
        )
        if self.is_logger:
            self.logger.info("Completed: gadm_legend")

        # Build legend for biome
        ecoregions_legend = (
            read_dataframe(
                self.config["paths"]["ecoregions"],
                read_geometry=False,
                columns=["ECO_ID", "ECO_NAME", "BIOME", "biomes_lab"],
            )
            .rename(
                columns={
                    "ECO_ID": "code_ecoregions",
                    "ECO_NAME": "name_ecoregions",
                    "BIOME": "code_biome",
                    "biomes_lab": "name_biome",
                }
            )
            .drop_duplicates(subset="code_ecoregions")
            .sort_values("code_ecoregions")
            .astype({"code_ecoregions": "int32", "code_biome": "int32"})
        ).set_index("code_ecoregions")
        ecoregions_legend.to_parquet(
            self.config["paths"]["ecoregions_legend"], index=True
        )
        if self.is_logger:
            self.logger.info("Completed: ecoregions_legend")

    def transform_reports_codes(self, refresh_all: bool = False):
        """Transform reports table"""
        # Get geo-labels for each record
        # NOTE: We need to operate with GeoPandas since the DuckDB spatial join is still
        # inefficient. Ideally, when working with big-data, we should use DuckDB until
        # we perform aggregation operations to avoid in-memory computations.

        reports_codes = ReportsCodes(self.config, self.logger)

        latlon = pd.read_parquet(
            self.config["paths"]["reports_transf"],
            columns=["version_uuid", "lat", "lon"],
        ).set_index("version_uuid")
        reports_codes_exist = os.path.isfile(self.config["paths"]["reports_codes"])

        if not refresh_all and reports_codes_exist:
            old_reports_codes = pd.read_parquet(self.config["paths"]["reports_codes"])
            new_latlon_idx = set(latlon.index).difference(set(old_reports_codes.index))

            if len(new_latlon_idx) > 0:
                new_latlon = latlon.loc[list(new_latlon_idx)]
                new_reports_codes = reports_codes.get_codes(new_latlon)
                reports_codes = pd.concat(
                    [old_reports_codes, new_reports_codes], axis=0
                )
                reports_codes.to_parquet(self.config["paths"]["reports_codes"])
            if self.is_logger:
                self.logger.info("Location labels added to reports (update)")
        else:
            reports_codes = reports_codes.get_codes(
                latlon, save_path=self.config["paths"]["reports_codes"]
            )
            if self.is_logger:
                self.logger.info("Location labels added to reports (full refresh)")

    def transform_app_users(self) -> None:
        """Transform 'app_users' table"""

        app_users = duckdb.read_parquet(self.config["paths"]["app_users"])

        prql_raw = """
            from app_users
            derive {
                registration_time_utc = s"registration_time::TIMESTAMP",
                last_score_update_utc = s"last_score_update::TIMESTAMP",
            }
            select !{registration_time, last_score_update}
            sort registration_time_utc
        """
        app_users_transf = duckdb.query(prql2sql_duckdb(prql_raw))
        app_users_transf.to_parquet(self.config["paths"]["app_users_transf"])
        if self.is_logger:
            self.logger.info("Completed: app_users_transf")

    def transform_human_presence(self) -> None:
        """Transform human presence raster"""

        with rasterio.open(
            self.config["paths"]["human_presence_raster"],
        ) as src:
            raster = src.read()
            # Setup nodata value
            raster[0, :, :][raster[0, :, :] == 0] = -1  # climatic regions
            raster[1, :, :][raster[1, :, :] < 0] = -1  # population density
            raster[2, :, :][raster[2, :, :] == 0] = -1  # gadm
            raster[3, :, :][raster[3, :, :] == 0] = -1  # ecoregions

            n_bands = raster.shape[0]
            raster = raster.T.reshape((-1, n_bands))
            raster = raster[(raster != -1).all(axis=1)]

            raster = pd.DataFrame(
                raster,
                columns=[
                    "code_climate_regions",
                    "human_presence",
                    "code_gadm",
                    "code_ecoregions",
                ],
            )
            raster = raster[
                [
                    "human_presence",
                    "code_gadm",
                    "code_climate_regions",
                    "code_ecoregions",
                ]
            ]

        # Add biome codes
        raster = raster.merge(
            pd.read_parquet(
                self.config["paths"]["ecoregions_legend"],
                columns=["code_ecoregions", "code_biome"],
            ),
            how="left",
            left_on="code_ecoregions",
            right_on="code_ecoregions",
        )

        human_presence = duckdb.sql(
            prql2sql_duckdb(prql_human_presence_discr("raster"))
        )
        human_presence.to_parquet(self.config["paths"]["human_presence"])
        if self.is_logger:
            self.logger.info("Completed: human_presence")

    def transform_sampling_effort(self) -> None:
        """Transform 'sampling_effort' table"""

        sampling_effort = pd.read_parquet(self.config["paths"]["sampling_effort"])
        sampling_effort_unique = (
            sampling_effort[["lonlat_id", "lon", "lat"]]
            .drop_duplicates(subset="lonlat_id")
            .set_index("lonlat_id")
        )

        with rasterio.open(
            self.config["paths"]["human_presence_raster"],
        ) as src:
            raster_sample = np.array(
                list(
                    src.sample(
                        sampling_effort_unique[["lon", "lat"]].astype(float).values
                    )
                )
            )

        raster_sample = pd.DataFrame(
            raster_sample,
            index=sampling_effort_unique.index,
            columns=[
                "code_climate_regions",
                "human_presence",
                "code_gadm",
                "code_ecoregions",
            ],
        )
        # Add biome codes
        raster_sample = raster_sample.drop(
            raster_sample.query(
                "human_presence < 0 | code_gadm == 0 | code_climate_regions == 0 | code_ecoregions == 0"
            ).index
        ).reset_index()
        raster_sample = raster_sample.merge(
            pd.read_parquet(
                self.config["paths"]["ecoregions_legend"],
                columns=["code_ecoregions", "code_biome"],
            ),
            how="left",
            left_on="code_ecoregions",
            right_on="code_ecoregions",
        )

        sampling_effort = sampling_effort.merge(
            raster_sample, left_on="lonlat_id", right_on="lonlat_id", how="right"
        )

        sampling_effort_transf = duckdb.sql(
            prql2sql_duckdb(prql_human_presence_discr("sampling_effort"))
        )
        sampling_effort_transf.to_parquet(
            self.config["paths"]["sampling_effort_transf"]
        )
        if self.is_logger:
            self.logger.info("Completed: sampling_effort_transf")

    def transform_reports_response(self):
        reports_response = duckdb.from_parquet(self.config["paths"]["reports_response"])

        sql = """
        WITH cte1 AS (
            SELECT version_uuid,
                CASE
                    WHEN answer_id = 711 THEN 'thorax_albopictus'
                    WHEN answer_id = 712 THEN 'thorax_aegypti'
                    WHEN answer_id = 713 THEN 'thorax_japonicus'
                    WHEN answer_id = 714 THEN 'thorax_koreicus'
                    WHEN answer_id = 721 THEN 'abdomen_albopictus'
                    WHEN answer_id = 722 THEN 'abdomen_aegypti'
                    WHEN answer_id = 723 THEN 'abdomen_japonicus'
                    WHEN answer_id = 724 THEN 'abdomen_koreicus'
                    WHEN answer_id = 731 THEN 'leg_albopictus'
                    WHEN answer_id = 732 THEN 'leg_aegypti'
                    WHEN answer_id = 733 THEN 'leg_japonicus'
                    WHEN answer_id = 734 THEN 'leg_koreicus'
                    ELSE NULL
                END AS answer_label,
                split_part(answer_label, '_', 2) AS cs_labels,
                split_part(answer_label, '_', 1) AS body_label
            FROM reports_response
            WHERE question_id = 7
        ),
        cte2 AS (
            SELECT version_uuid,
                cs_labels,
                SUM(
                    CASE
                        WHEN body_label = 'thorax' THEN 0.6
                        WHEN body_label = 'abdomen' THEN 0.3
                        WHEN body_label = 'leg' THEN 0.1
                        ELSE 0
                    END
                ) AS weight
            FROM cte1
            GROUP BY version_uuid,
                cs_labels
        ),
        invasive AS (
            SELECT version_uuid,
                cs_labels,
                weight,
                FROM cte2 WINDOW weight_window AS (
                    PARTITION BY version_uuid
                    ORDER BY weight DESC
                ) QUALIFY ROW_NUMBER() OVER weight_window = 1
        ),
        common AS (
            SELECT version_uuid,
                CASE
                    WHEN answer_id = 62 THEN 'culex'
                    ELSE NULL
                END AS cs_labels,
                CASE
                    WHEN answer_id = 62 THEN 1.0
                    ELSE NULL
                END AS weight,
                FROM reports_response
            WHERE answer_id IN (62, 64)
        ),
        final AS (
            SELECT *
            FROM invasive
            UNION
            SELECT *
            FROM common
        )
        SELECT *
        FROM final
        """
        reports_response_transf = duckdb.sql(sql)

        reports_response_transf.to_parquet(
            self.config["paths"]["reports_response_transf"]
        )
        if self.is_logger:
            self.logger.info("Completed: reports_response_transf")
