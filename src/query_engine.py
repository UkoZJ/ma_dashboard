from functools import partial
from typing import Literal, Optional
from configparser import ConfigParser


import duckdb
import pandas as pd
import prql_python

from src import utils

prql2sql = partial(
    prql_python.compile, options=prql_python.CompileOptions(target="sql.duckdb")
)

FREQ_MAP = {
    "day": "D",
    "yearweek": "W-MON",
    "month": "MS",
    "quarter": "QS",
    "year": "YS",
}
FREQ = Literal["day", "yearweek", "month", "quarter", "year"]
SCALE = Literal[
    "name_gadm_level0",
    "name_gadm_level1",
    "name_gadm_level2",
    "name_gadm_level3",
    "name_gadm_level4",
    "name_climate",
    "name_biome",
    "nome_ecoregions",
]
REPORT_TYPE = ["adult", "bite", "site"]


def entity_filter(
    scale: str, entity: str, engine_type: Literal["prql", "sql", "sql_and"] = "prql"
) -> str:
    if entity != "Total":
        if engine_type == "prql":
            return f"""
            filter {scale} == "{entity}"
            """
        elif engine_type == "sql":
            return f"""
            where {scale} = $${entity}$$
            """
        elif engine_type == "sql_and":
            return f"""
            and {scale} = $${entity}$$
            """
        else:
            raise KeyError(engine_type)
    else:
        return ""


class QueryEngine:
    def __init__(self, config: ConfigParser):
        # Connect to the tables and query
        self.reports = duckdb.from_parquet(config["paths"]["reports_transf"])
        reports_codes_ = duckdb.from_parquet(config["paths"]["reports_codes"])
        self.gadm_legend = duckdb.from_parquet(config["paths"]["gadm_legend"])
        self.app_users = duckdb.from_parquet(config["paths"]["app_users_transf"])
        self.climate_regions_legend = duckdb.from_csv_auto(
            config["paths"]["climate_regions_legend"]
        )
        self.ecoregions_legend = duckdb.from_parquet(
            config["paths"]["ecoregions_legend"]
        )
        self.human_presence = duckdb.from_parquet(config["paths"]["human_presence"])
        self.sampling_effort = duckdb.from_parquet(
            config["paths"]["sampling_effort_transf"]
        )

        gadm_legend = self.gadm_legend
        self.reports_codes = duckdb.sql(
            "select * from reports_codes_ left join gadm_legend using(code_gadm)"
        ).to_df()

    def get_entity_opts(self, scale: SCALE, filter: Optional[int] = None):
        if filter is None:
            reports_codes = self.reports_codes
            entity_opts = (
                duckdb.sql(f"""select distinct {scale} from reports_codes""")
                .to_df()[scale]
                .sort_values()
                .tolist()
            )
        else:
            entity_opts = (
                self.overview_rank(scale=scale)
                .query("total >= @filter")
                .sort_index()
                .index.to_list()
            )
        entity_opts.insert(0, "Total")
        return entity_opts

    def filter_view_labels(
        self,
        scale: SCALE = "name_gadm_level0",
        report_type: REPORT_TYPE = "adult",
        entity: str = "Total",
    ) -> pd.DataFrame:
        filt = entity_filter(scale, entity)
        reports = self.reports
        reports_codes = self.reports_codes

        prql = f"""
        
        let reports_codes_scale = (
            from reports_codes
            select {{version_uuid, {scale}}}
            )
        let reports_filt = (
            from reports
            filter report_type == "{report_type}"
            )
        from reports_filt
        join side:left reports_codes_scale (== version_uuid)
        {filt}
        group labels (
            aggregate{{ct = count version_uuid}})
        derive perc_ct = (ct / (sum ct))*100
        sort {{-perc_ct}}
        """
        df = duckdb.sql(prql2sql(prql)).to_df().dropna()

        if not df.empty:
            return df.set_index("labels").T
        else:
            return pd.DataFrame()

    def filter_view(
        self,
        freq: FREQ = "month",
        scale: SCALE = "name_gadm_level0",
        report_type: REPORT_TYPE = "adult",
        entity: str = "Total",
    ) -> pd.DataFrame:
        entity_opts = self.get_entity_opts(scale)

        assert entity in entity_opts, f"Entity should be a viable option, got: {entity}"

        filt = entity_filter(scale, entity)
        reports = self.reports
        reports_codes = self.reports_codes

        prql = f"""    
        let reports_codes_scale = (
            from reports_codes
            select {{version_uuid, {scale}}}
            )
        let reports_filt = (
            from reports
            filter report_type == "{report_type}"
            )
        from reports_filt
        join side:left reports_codes_scale (== version_uuid)
        {filt}
        derive date_base = s"date_trunc('{freq}', observation_date_utc)"
        group {{labels, date_base}}(
            aggregate{{ct = count version_uuid}})
        sort {{date_base, labels, ct}}
        """
        df = duckdb.sql(prql2sql(prql)).to_df()

        if not df.empty:
            time = utils.get_common_time(df["date_base"], freq=FREQ_MAP[freq])
            df_pv = df.pivot(index="date_base", columns="labels", values="ct").reindex(
                time, fill_value=0
            )
            df_pv = df_pv[[col for col in df_pv.columns if isinstance(col, str)]]
            return df_pv
        else:
            return pd.DataFrame()

    def overview_rank(
        self,
        scale: SCALE = "name_gadm_level0",
    ) -> pd.DataFrame:
        reports = self.reports
        reports_codes = self.reports_codes
        prql = f"""
        let reports_codes_scale = (
        from reports_codes
        select {{version_uuid, {scale}}}
        )
        from reports
        join side:left reports_codes_scale (== version_uuid)
        group {{{scale}, report_type}}(
            aggregate{{ct = count version_uuid}})
        filter {scale} != "NULL"
        """
        df = duckdb.sql(prql2sql(prql)).to_df()
        df_pv = df.pivot(index=scale, columns="report_type", values="ct")
        df_pv["total"] = df_pv.sum(axis=1)
        return df_pv.sort_values("total", ascending=False)

    def overview_time2publish(
        self,
        scale: SCALE = "name_gadm_level0",
        entity: str = "Total",
        freq: FREQ = "month",
    ):
        reports = self.reports
        reports_codes = self.reports_codes
        report_type = "adult"

        filt = entity_filter(scale, entity)
        prql = f"""
        let reports_codes_scale = (
            from reports_codes
            select {{version_uuid, {scale}}}
            )
        let reports_filt = (
            from reports
            filter report_type == "{report_type}"
            )
        from reports_filt
        join side:left reports_codes_scale (== version_uuid)
        {filt}
        filter validation_date_utc != null
        derive {{
            time2publish = s"date_diff('minute', upload_date_utc, validation_date_utc)",
            time2publish_pos = case [
                time2publish >= 0 => time2publish],
            date_base = s"date_trunc('{freq}', upload_date_utc)"
            }}
        filter time2publish_pos != null
        """
        # BUG: time2publish could be negative for some rare cases!

        df_cte = duckdb.sql(prql2sql(prql)).to_df()
        min2days = 60 * 24
        sql = f"""
        select
        date_base,
        quantile_cont(time2publish_pos,0.25)/{min2days} as perc25,
        quantile_cont(time2publish_pos,0.5)/{min2days} as median,
        quantile_cont(time2publish_pos,0.75)/{min2days} as perc75
        from df_cte
        group by date_base
        """

        df = duckdb.sql(sql).to_df().set_index("date_base")

        if not df.empty:
            time = utils.get_common_time(df.index, freq=FREQ_MAP[freq])
            return df.reindex(time, fill_value=None)
        else:
            return pd.DataFrame()

    def overview_view_inflow(
        self,
        scale: SCALE = "name_gadm_level0",
        entity: str = "Total",
        freq: FREQ = "month",
    ) -> pd.DataFrame:
        reports = self.reports
        reports_codes = self.reports_codes
        filt = entity_filter(scale, entity)

        prql = f"""
        let reports_codes_scale = (
        from reports_codes
        select {{version_uuid, {scale}}}
        )
        from reports
        join side:left reports_codes_scale (== version_uuid)
        {filt}
        derive date_base = s"date_trunc('{freq}', upload_date_utc)"
        group {{report_type, date_base}}(
            aggregate{{ct = count version_uuid}})
        sort {{date_base, report_type, ct}}
        """
        df = duckdb.sql(prql2sql(prql)).to_df()

        if not df.empty:
            time = utils.get_common_time(df["date_base"], freq=FREQ_MAP[freq])
            return df.pivot(
                index="date_base", columns="report_type", values="ct"
            ).reindex(time, fill_value=0)
        else:
            return pd.DataFrame()

    def overview_view_accumulation(
        self,
        scale: SCALE = "name_gadm_level0",
        entity: str = "Total",
        freq: FREQ = "month",
    ):
        reports = self.reports
        reports_codes = self.reports_codes
        filt = entity_filter(scale, entity)
        report_type = "adult"

        prql_1 = f"""
        let reports_codes_scale = (
            from reports_codes
            select {{version_uuid, {scale}}}
            )
        let reports_filt = (
            from reports
            filter report_type == "{report_type}"
            )
        from reports_filt
        join side:left reports_codes_scale (== version_uuid)
        {filt}
        derive {{
            upload_date = s"date_trunc('{freq}', upload_date_utc)",
            validation_date = s"date_trunc('{freq}', validation_date_utc)"
            }}
        """
        tmp = duckdb.sql(prql2sql(prql_1))

        prql_2 = """
        let inflow = (
            from tmp
            group upload_date (aggregate{ct_inflow_tmp = count upload_date})
        )
        let outflow = (
            from tmp
            filter validated = true
            group validation_date (aggregate{ct_outflow_tmp = count validation_date})
        )
        from inflow
        join side:left outflow (upload_date == validation_date)
        derive {
            ct_outflow = case [
                ct_outflow_tmp != null => ct_outflow_tmp,
                true => 0 ],
            ct_inflow = case [
                ct_inflow_tmp != null => ct_inflow_tmp,
                true => 0 ],
            ct_diff = ct_inflow - ct_outflow
            }
        sort upload_date
        window expanding:true (
        derive {ct_acc = sum ct_diff})
        select {upload_date, ct_acc}
        """

        df = duckdb.sql(prql2sql(prql_2)).to_df().set_index("upload_date")

        if not df.empty:
            time = utils.get_common_time(df.index, freq=FREQ_MAP[freq])
            return df.reindex(time).ffill()
        else:
            return pd.DataFrame()

    def user_activity_rank(
        self,
        retention_interval: str = "1 year",
        scale: SCALE = "name_gadm_level0",
    ):
        reports = self.reports
        reports_codes = self.reports_codes
        app_users = self.app_users

        sql = f"""
        with
        reports_codes_scale as (
            select version_uuid, {scale}
            from reports_codes
            where {scale} <> 'NULL'
        ),
        tar_users_with_report as (
            select user_id, {scale}, count(*) report_count, max(upload_date_utc) last_user_upload_utc
            from reports
            right join reports_codes_scale using(version_UUID)
            group by user_id, {scale}
        ),
        new_users as (
            select {scale}, count(*) ct_new_users
            from tar_users_with_report
            group by {scale}
        ),
        active_users as (
            select {scale}, count(*) ct_active_users
            from tar_users_with_report
            where last_user_upload_utc >= (NOW()::TIMESTAMP - interval '{retention_interval}') 
            group by {scale}
        ),
        final as (
        select * from new_users
        left join active_users using({scale})
        order by ct_new_users desc
        )
        select * from final
        """
        df = (
            duckdb.sql(sql)
            .to_df()
            .set_index(scale)
            .rename(
                columns={
                    "ct_active_users": "active",
                    "ct_new_users": "new",
                    "ct_registered_users": "registered",
                }
            )
        )
        df = df.fillna(0)
        
        return df

    def user_activity(
        self,
        freq: FREQ = "month",
        retention_interval: str = "1 year",
        scale: SCALE = "name_gadm_level0",
        entity: str = "Total",
    ):
        reports = self.reports
        reports_codes = self.reports_codes
        app_users = self.app_users
        filt = entity_filter(scale, entity, engine_type="sql")

        sql = f"""
        with
        reports_codes_scale as (
            select version_uuid, {scale}
            from reports_codes
            where {scale} <> 'NULL'
        ),
        tar as (
            select user_id, count(*) report_count, max(upload_date_utc) last_user_upload_utc
            from reports
            right join reports_codes_scale using(version_UUID)
            {filt}
            group by user_id 
        ),
        tar_users as (
            select user_UUID as user_id, registration_time_utc
            from app_users au
        ),
        registered_users as (
            select date_trunc('{freq}', registration_time_utc) date_base, count(*) ct_registered_users
            from tar_users
            group by date_base
        ),
        tar_users_with_report as (
            select user_id, registration_time_utc, last_user_upload_utc
            from app_users au
            right join tar on tar.user_id = au."user_UUID" 
        ),
        new_users as (
            select date_trunc('{freq}', registration_time_utc) date_base, count(*) ct_new_users
            from tar_users_with_report
            group by date_base
        ),
        active_users as (
            select date_trunc('{freq}', registration_time_utc) date_base, count(*) ct_active_users
            from tar_users_with_report
            where last_user_upload_utc >= (NOW()::TIMESTAMP - interval '{retention_interval}') 
            group by date_base
        ),
        final as (
        select * from registered_users
        left join new_users using(date_base)
        left join active_users using(date_base)
        order by date_base
        )
        select * from final
        """

        df = (
            duckdb.sql(sql)
            .to_df()
            .set_index("date_base")
            .rename(
                columns={
                    "ct_active_users": "active",
                    "ct_new_users": "new",
                    "ct_registered_users": "registered",
                }
            )
        )
        df = df.fillna(0)
        df["inactive"] = df["new"] - df["active"]
        df["active_perc"] = df["active"] / df["new"] * 100
        df["new_perc"] = df["new"] / df["new"].max() * 100
        df = df.tail(-1)

        if not df.empty:
            time = utils.get_common_time(df.index, freq=FREQ_MAP[freq])
            return df.reindex(time, fill_value=0)
        else:
            return pd.DataFrame()

    def user_retention(
        self,
        scale: SCALE = "name_gadm_level0",
        entity: str = "Total",
        freq: FREQ = "month",
        report_type: Optional[str] = "",
        retention_interval: str = "1 year",
    ):
        reports = self.reports
        reports_codes = self.reports_codes
        app_users = self.app_users

        if report_type != "":
            filt_report_type = f"where report_type = '{report_type}'"
            filt = entity_filter(scale, entity, engine_type="sql_and")
        else:
            filt_report_type = ""
            filt = entity_filter(scale, entity, engine_type="sql")

        sql = f"""
        WITH
        retention_data AS (
            SELECT
                user_id,
                upload_date_utc,
                LAG(upload_date_utc) OVER (PARTITION BY user_id ORDER BY upload_date_utc) AS previous_upload_date_utc,
                date_trunc('{freq}', upload_date_utc) AS date_base,
            FROM
                reports
            LEFT JOIN reports_codes USING(version_uuid)
            {filt_report_type}
            {filt}
            )
        SELECT
        date_base,
        COUNT(DISTINCT CASE WHEN upload_date_utc - previous_upload_date_utc <= INTERVAL '{retention_interval}' THEN user_id END) * 100.0 / COUNT(DISTINCT user_id) AS user_retention
        FROM retention_data
        GROUP BY date_base
        ORDER BY date_base
        """

        df = duckdb.sql(sql).to_df().set_index("date_base")
        if not df.empty:
            time = utils.get_common_time(df.index, freq=FREQ_MAP[freq])
            return df.reindex(time, fill_value=0)
        else:
            return pd.DataFrame()

    def user_retention_stats(
        self,
        scale: SCALE = "name_gadm_level0",
        entity: str = "Total",
        freq: FREQ = "month",
        report_type: str = "",
        retention_interval: str = "1 year",
    ):
        return self.user_activity(
            freq=freq,
            retention_interval=retention_interval,
            scale=scale,
            entity=entity,
        ).join(
            self.user_retention(
                scale=scale,
                entity=entity,
                freq=freq,
                report_type=report_type,
                retention_interval=retention_interval,
            )
        )

    def overview_sampling_effort(
        self,
        scale: SCALE = "name_gadm_level0",
        entity: str = "Total",
        freq: FREQ = "month",
    ):
        gadm_legend = self.gadm_legend
        climate_regions_legend = self.climate_regions_legend.to_df().rename(
            columns={"code": "code_climate_regions", "name": "name_climate_regions"}
        )
        ecoregions_legend = self.ecoregions_legend
        human_presence = self.human_presence
        sampling_effort = self.sampling_effort

        filt = entity_filter(scale, entity)
        legends = {
            "name_gadm_level0": ["gadm_legend", "code_gadm"],
            "name_gadm_level1": ["gadm_legend", "code_gadm"],
            "name_gadm_level2": ["gadm_legend", "code_gadm"],
            "name_gadm_level3": ["gadm_legend", "code_gadm"],
            "name_gadm_level4": ["gadm_legend", "code_gadm"],
            "name_climate_regions": ["climate_regions_legend", "code_climate_regions"],
            "name_biome": ["ecoregions_legend", "code_biome"],
            "name_ecoregions": ["ecoregions_legend", "code_ecoregions"],
        }
        legend, code = legends[scale]
        prql = f"""
       let legend_ = (
            from {legend}
            select {{{code}, {scale}}}
            )
        let sampling_effort_presence_absence = (
            from sampling_effort
            filter n_participants >= 1
            derive date_base = s"date_trunc('{freq}', date_utc::DATE)"
            group {{lonlat_id, date_base}} (take 1)
            join side:left legend_ (=={code})
            select {{date_base, {scale}, human_presence_discr}}
            )
        let sampling_effort_filt = (
            from sampling_effort_presence_absence
            {filt}
            group {{human_presence_discr, date_base}}(
            aggregate{{ct = count sampling_effort_presence_absence.*}})
            )
        let human_presence_filt = (
            from human_presence
            join side:left legend_ (=={code})
            {filt}
            group human_presence_discr(
            aggregate{{ct_tot = count human_presence.*}})
            )
        from sampling_effort_filt
        join side:left human_presence_filt (==human_presence_discr)
        derive ct_perc = ct / ct_tot * 100
        select {{date_base, sampling_effort_filt.human_presence_discr, ct_perc}}
        sort date_base
        """

        df = duckdb.sql(prql2sql(prql)).to_df()

        if not df.empty:
            time = utils.get_common_time(df["date_base"], freq=FREQ_MAP[freq])
            return (
                df.pivot(
                    index="date_base", columns="human_presence_discr", values="ct_perc"
                )
                .reindex(time, fill_value=0)
                .fillna(0)
            )
        else:
            return pd.DataFrame()

    def ppa_view_inflow(
        self,
        users_stats: pd.DataFrame,
        user_labels: str,
        entity: str = "Total",
        freq: FREQ = "month",
    ):
        users_stats = users_stats.reset_index()
        reports = self.reports
        filt = entity_filter("name_gadm_level0", entity)

        prql = f"""
            let users_stats_ = (
                from users_stats
                {filt}
                select {{user_id, {user_labels}}}
            )
            let reports_ = (
                from reports
                filter (report_type == "adult" && labels != "not-sure" && labels != "unvalidated")
            )
            let final = (
                from reports_
                join side:inner users_stats_ (== user_id)
                derive date_base = s"date_trunc('{freq}', upload_date_utc)"
                group {{{user_labels}, date_base}}(
                    aggregate{{ct = count user_id}})
            )
            from final
            """
        df = duckdb.sql(prql2sql(prql)).to_df()

        if not df.empty:
            time = utils.get_common_time(df["date_base"], freq=FREQ_MAP[freq])
            return df.pivot(
                index="date_base", columns=user_labels, values="ct"
            ).reindex(time, fill_value=0)
        else:
            return pd.DataFrame()
