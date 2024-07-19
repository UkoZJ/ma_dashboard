import warnings
import os
from functools import partial
from typing import List, Literal, Optional, Tuple, Union
from configparser import ConfigParser
import logging

import duckdb
import gower
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import prql_python
import seaborn as sns
from loguru import logger
from matplotlib import ticker
from scipy.spatial import distance
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report
from umap import UMAP

from .log import task_logging
from .query_engine import entity_filter, QueryEngine


prql2sql = partial(
    prql_python.compile, options=prql_python.CompileOptions(target="sql.duckdb")
)

warnings.filterwarnings("ignore")

# FIG_FORMAT = "png"
FIG_FORMAT = "pdf"
REL_PATH = "./pics"
DPI = 300
CM = 1 / 2.54
WIDTH_FIG_HALFPAGE = 8.5 * CM  # inch
WIDTH_FIG_FULLPAGE = 17 * CM  # inch
FONTSIZE_FIG_MAIN = 8.0
FONTSIZE_FIG_TICK = 7.0
BAR_WIDTH_TIGHT = 0.9
CONTEXT = "notebook"
# PALETTE = "Spectral"
PALETTE = "tab10"
PALETTE_N = partial(sns.color_palette, PALETTE)
LINE_COLOR = "#726e72"

sns.set_theme(
    context=CONTEXT,
    style="white",
    palette=None,
    rc={
        "xtick.bottom": True,
        "ytick.left": True,
        "font.size": FONTSIZE_FIG_MAIN,
        "axes.labelsize": FONTSIZE_FIG_MAIN,
        "axes.titlesize": FONTSIZE_FIG_MAIN,
        "xtick.labelsize": FONTSIZE_FIG_TICK,
        "ytick.labelsize": FONTSIZE_FIG_TICK,
        "legend.fontsize": FONTSIZE_FIG_TICK,
        "legend.title_fontsize": FONTSIZE_FIG_MAIN,
    },
)


def point_pattern_analysis(
    user_id: str,
    df_users: pd.DataFrame,
    min_samples: int = 5,
    eps: float = 100,
    method: Literal["dbscan", "hdbscan", "optics"] = "optics",
    plot_fig: bool = False,
) -> dict:

    import warnings

    import geopandas as gpd
    import numpy as np
    import shapely
    from sklearn.cluster import HDBSCAN, DBSCAN, OPTICS

    import src.geocoding as geocoding

    warnings.filterwarnings("ignore")

    def minimum_bounding_rectangle(points):
        """
        Find minimum bounding rectangle of a point array.
        """
        points = np.asarray(points)
        x, y = zip(*points)
        min_x = min(x)
        min_y = min(y)
        max_x = max(x)
        max_y = max(y)
        return min_x, min_y, max_x, max_y

    def std_distance(points, m=None):
        points = np.asarray(points)
        n, p = points.shape
        if m is None:
            m = points.mean(axis=0)
        return np.sqrt(((points * points).sum(axis=0) / n - m * m).sum())

    def msd(points, m=None):
        """Mean squared displacement"""
        points = np.asarray(points)
        n, p = points.shape
        if m is None:
            m = points.mean(axis=0)
        return np.sqrt(((points - m) ** 2 / n).sum())

    def convert_wgs_to_utm(lon: float, lat: float):
        """Based on lat and lng, return best utm epsg-code"""
        utm_band = str((np.floor((lon + 180) / 6) % 60) + 1)
        if len(utm_band) == 1:
            utm_band = "0" + utm_band
        if lat >= 0:
            epsg_code = "326" + utm_band
            return epsg_code
        epsg_code = "327" + utm_band
        return epsg_code

    # median = centrography.euclidean_median(df_user[["lon", "lat"]])
    # epsg_utm = f"EPSG:{convert_wgs_to_utm(median[0], median[1])}"

    df_user = df_users.loc[[user_id], :]
    ppa_keys = [
        "n_points",
        "n_invalid_points",
        "mean_latlon",
        "std_utm",
        "msd_utm",
        "n_random_points",
        "n_clusters",
    ]
    ppa = {key: np.nan for key in ppa_keys}

    if len(df_user) <= 1:
        ppa["n_points"] = len(df_user)
        ppa["n_invalid_points"] = 0
        # mean = df_user[["lon", "lat"]].values
        return ppa
    else:
        mean = df_user[["lon", "lat"]].mean(axis=0).values
        epsg_utm = f"EPSG:{convert_wgs_to_utm(mean[0], mean[1])}"

        # lat/lon in UTM coordinates
        gdf_user = gpd.GeoDataFrame(
            df_user[["lat", "lon"]],
            geometry=gpd.points_from_xy(
                df_user["lon"], df_user["lat"], crs="EPSG:4326"
            ),
        )
        gdf_user = gdf_user.to_crs(epsg_utm)
        gdf_user_ = gdf_user[gdf_user["geometry"].is_valid]
        xy_user = shapely.get_coordinates(gdf_user_["geometry"])
        mean_utm = np.asarray(xy_user).mean(axis=0)

        ppa["n_points"] = len(xy_user)
        ppa["n_invalid_points"] = len(gdf_user) - ppa["n_points"]
        ppa["mean_latlon"] = mean
        ppa["std_utm"] = std_distance(xy_user, mean_utm)
        ppa["msd_utm"] = msd(xy_user, mean_utm)

        if len(df_user) >= min_samples:
            # Clustering
            match method:
                case "hdbscan":
                    # Invariant to feature scale
                    clusterer = HDBSCAN(
                        **{
                            "min_cluster_size": min_samples,
                            "min_samples": min_samples,
                            "cluster_selection_epsilon": eps,
                            "allow_single_cluster": True,
                        }
                    )
                case "dbscan":
                    # Lat/Lon are on the same scale
                    clusterer = DBSCAN(
                        **{
                            "min_samples": min_samples,
                            "eps": eps,
                        }
                    )
                case "optics":
                    # Not constrained by EPS since it searches over EPS=inf
                    clusterer = OPTICS(
                        **{
                            "min_samples": min_samples,
                            "metric": "euclidean",
                        }
                    )
            clusterer.fit(xy_user)
            gdf_user_["labels"] = clusterer.labels_

            labels_count = gdf_user_.value_counts("labels")
            if -1 in labels_count.index:
                if len(labels_count) > 1:
                    labels_count = labels_count.drop(-1)
                    n_clusters = len(labels_count)
                else:
                    n_clusters = 0
            else:
                n_clusters = len(labels_count)

            ppa["n_clusters"] = n_clusters
            ppa["n_random_points"] = sum(clusterer.labels_ == -1)

    if plot_fig:
        import os
        from src import utils
        from pathlib import Path

        root_dir = Path(os.getcwd()).parent
        config = utils.get_config(path=root_dir)
        rc = geocoding.ReportsCodes(config)
        world = rc.get_geopoly(level="gadm")

        mbb = minimum_bounding_rectangle(df_user[["lon", "lat"]])
        xlim = [mbb[0], mbb[2]]
        ylim = [mbb[1], mbb[3]]
        ax = world.clip([xlim[0], ylim[0], xlim[1], ylim[1]]).plot(
            color="white", edgecolor="black", alpha=0.1
        )
        sns.scatterplot(
            data=gdf_user_,
            x="lon",
            y="lat",
            hue="labels",
            marker=".",
            palette="deep",
            ax=ax,
        )

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    return ppa


def point_pattern_analysis_logging(user_id, df_users, min_samples):
    try:
        return point_pattern_analysis(user_id, df_users, min_samples)
    except Exception as e:
        logging.exception(f"Exception occurred for user_id: {user_id}")
        raise


def user_labels(
    df: pd.DataFrame, std_lim: Optional[int] = 1000, quality_lim: Optional[float] = 0.5
) -> str:
    """Hierarchical classification based on geo-location heuristic rules"""

    if std_lim is not None:
        if df["n_clusters"] == 0:
            if df["std_utm"] >= std_lim:
                label = "sparse wide"
            else:
                label = "sparse narrow"
        elif df["n_clusters"] == 1:
            if df["std_utm"] >= std_lim:
                label = "clustered wide"
            else:
                label = "clustered narrow"
        elif df["n_clusters"] > 1:
            label = "multi-clustered"
        else:
            label = "sporadic"
    else:
        if df["n_clusters"] == 0:
            label = "sparse"
        elif df["n_clusters"] == 1:
            label = "clustered"
        elif df["n_clusters"] > 1:
            label = "multi-clustered"
        else:
            label = "sporadic"

    if (quality_lim is not None) & ("sporadic" not in label):
        if df["quality"] >= quality_lim:
            label = "HQ-" + label
        else:
            label = "LQ-" + label
    return label


class QueryEngine:
    def __init__(self, config: ConfigParser):
        # Connect to the tables and query

        self.config = config
        reports_codes_ = duckdb.from_parquet(config["paths"]["reports_codes"])
        self.gadm_legend = duckdb.from_parquet(config["paths"]["gadm_legend"])
        self.reports = duckdb.from_parquet(config["paths"]["reports_transf"])

        gadm_legend = self.gadm_legend
        self.reports_codes = duckdb.sql(
            "select * from reports_codes_ join gadm_legend using(code_gadm)"
        ).to_df()

        self.scale = "name_gadm_level0"
        self.entity = "Total"
        self.filt = entity_filter(self.scale, self.entity)

    def get_users_ppa(
        self,
        freq="month",
        report_type: str = "",
        min_samples: int = 5,
        method: str = "optics",
        max_workers: int = 7,
    ):
        from concurrent.futures import ProcessPoolExecutor

        reports = self.reports

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

        df_users = duckdb.sql(prql2sql(prql)).to_df().set_index("user_id")
        ds_count = df_users.value_counts("user_id").sort_values()
        users_id_one = ds_count[ds_count == 1].index.to_list()
        users_id_high = ds_count[ds_count > 1].index.to_list()

        point_pattern_analysis_ = partial(
            point_pattern_analysis,
            df_users=df_users,
            min_samples=min_samples,
            method=method,
        )

        users_stats_high = dict()
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                executor.map(
                    point_pattern_analysis_,
                    users_id_high,
                    chunksize=len(users_id_high) // max_workers,
                )
            )
            for user_id, res in zip(users_id_high, results):
                users_stats_high[user_id] = res
        users_stats_high = pd.DataFrame.from_dict(users_stats_high, orient="index")

        users_stats_one = pd.DataFrame(
            np.tile(np.array([1, 0] + [np.nan] * 5), len(users_id_one)).reshape(
                len(users_id_one), 7
            ),
            index=users_id_one,
            columns=users_stats_high.columns,
        ).astype(dtype=users_stats_high.dtypes)

        users_ppa = pd.concat([users_stats_high, users_stats_one], axis=0).sort_values(
            "n_points", ascending=True
        )
        users_ppa.index.name = "user_id"
        users_ppa.to_parquet(self.config["paths"]["users_ppa"])
        logger.info("Completed: users_ppa")

    def get_users_quality(self, report_type=None):

        reports = self.reports
        reports_codes = self.reports_codes

        if report_type == None:
            filt_report_type = ""
        else:
            filt_report_type = f"""
                filter report_type == "{report_type}"
                """

        prql = f"""
        let reports_codes_scale = (
            from reports_codes
            select {{version_uuid, {self.scale}}}
            )
        from reports
        {filt_report_type}
        join side:left reports_codes_scale (== version_uuid)
        {self.filt}
        group {{user_id, {self.scale}}} (aggregate{{ct = count version_uuid}})
        """
        users_location_activity = duckdb.sql(prql2sql(prql))

        sql = f"""
        WITH ranked_ct AS (
        SELECT
            user_id, {self.scale}, ct,
            ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY ct DESC) AS row_num
        FROM users_location_activity
        )
        SELECT user_id, {self.scale}
        FROM ranked_ct
        WHERE row_num = 1
        """
        users_location_main_activity = duckdb.sql(sql)

        prql = f"""
        let reports_codes_scale = (
            from reports_codes
            select {{version_uuid, {self.scale}}}
            )
        from reports
        filter (report_type == "adult" && labels != "unknown")
        join side:left reports_codes_scale (== version_uuid)
        {self.filt}
        derive is_target_species = case [
            labels == "other_species" => 0,
            labels == "other" => 0,
            true => 1,
        ]
        group user_id (aggregate{{
            accum_target = sum is_target_species,
            accum_confidence = sum confidence,
            n_reports = count version_uuid
            }})
        derive {{
            quality_target = accum_target/n_reports,
            quality_reports = accum_confidence/n_reports
        }}
        join side:left users_location_main_activity (== user_id)
        sort {{-n_reports}}
        select {{user_id, {self.scale}, n_reports, quality_target, quality_reports}}
        """

        users_quality = duckdb.sql(prql2sql(prql))
        users_quality.to_parquet(self.config["paths"]["users_quality"])
        logger.info("Completed: users_quality")

    @task_logging(logger=logger)
    def run_ppa(self, refresh_users_ppa: bool = True):

        if refresh_users_ppa:
            self.get_users_ppa(method="dbscan")

        self.get_users_quality()

        users_ppa = duckdb.read_parquet(self.config["paths"]["users_ppa"])
        users_quality = duckdb.read_parquet(self.config["paths"]["users_quality"])

        prql = f"""
        from users_quality
        join side:left users_ppa (== user_id)
        {self.filt}
        select !{{users_ppa.user_id}}
        """
        users_ppa_quality = duckdb.sql(prql2sql(prql))
        users_ppa_quality.to_parquet(self.config["paths"]["users_ppa_quality"])


class DimReduction:

    def __init__(
        self,
        df: pd.DataFrame,
        metric: str = "euclidean",
        n_neighbors: list = [30],
        n_components: int = 2,
        feature_transformers: dict = (
            {
                "raw": None,
            },
        ),
        is_dimred: bool = True,
        random_state: Optional[int] = None,
    ) -> None:

        self.df = df
        self.metric = metric
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.feature_transformers = feature_transformers
        self.is_dimred = is_dimred
        self.random_state = random_state

    def screening(
        self, feature_cols: List[str], model_type: Literal["umap", "tsne"] = "umap"
    ):

        dimred = {}
        for name, ft in self.feature_transformers.items():
            if (self.metric == "gower") & (name != "raw"):
                break

            print(
                f"{name.capitalize()} feature transformation with {model_type}-{self.metric}"
            )

            if (ft is not None) & (self.metric != "gower"):
                Xft = ft.fit_transform(self.df[feature_cols])
            else:
                Xft = self.df[feature_cols].to_numpy()
            if self.metric is not None:
                if self.metric == "gower":
                    Xft_dist = gower.gower_matrix(Xft)
                else:
                    Xft_dist = distance.squareform(
                        distance.pdist(Xft, metric=self.metric)
                    )
            else:
                Xft_dist = None

            if self.is_dimred:

                df_ft = pd.DataFrame(Xft_dist, index=self.df.index)
                Xft_dimred = self.dim_reduction(
                    df_ft,
                    labels=self.df["custom_labels"],
                    model_type=model_type,
                    point_size=8,
                )
                dimred[name] = (Xft, Xft_dist, Xft_dimred)
            else:
                dimred[name] = (Xft, Xft_dist)

            return dimred

    def _model(
        self,
        model_type: Literal["umap", "tsne"],
        n_neighbors: int,
    ) -> Union[UMAP, TSNE]:
        """Instantiate dimensional reduction model with default parameters"""

        match model_type:
            case "umap":
                return UMAP(
                    n_neighbors=n_neighbors,
                    n_components=self.n_components,
                    metric="precomputed",
                    random_state=self.random_state,
                    # densmap=True,
                )
            case "tsne":
                return TSNE(
                    perplexity=n_neighbors,
                    n_components=self.n_components,
                    metric="precomputed",
                    random_state=self.random_state,
                    init="random",
                )

    def dim_reduction(
        self,
        df: pd.DataFrame,
        labels: pd.Series,
        model_type: Literal["umap", "tsne"] = "umap",
        supervised: bool = False,
        point_size: float = 0.1,
    ) -> pd.DataFrame:
        """Perform feature dimensional reduction with UMAP and display the results"""

        if isinstance(self.n_neighbors, int):
            n_neighbors = [self.n_neighbors]
        else:
            n_neighbors = self.n_neighbors

        dims = [f"Dim{i}" for i in range(1, self.n_components + 1)]
        X = df.to_numpy()
        dimred = []
        for n in n_neighbors:
            if supervised:
                model = self._model(model_type, n).fit(X, y=labels.factorize()[0])
            else:
                model = self._model(model_type, n).fit(X)
            df_dimred = pd.DataFrame(model.embedding_, columns=dims, index=df.index)
            df_dimred["label"] = labels
            df_dimred["neighbors"] = n

            dimred.append(df_dimred)

        df_dimred = pd.concat(dimred, axis=0)

        if self.n_components == 2:
            if len(n_neighbors) > 1:
                p = sns.relplot(
                    data=df_dimred,
                    x="Dim1",
                    y="Dim2",
                    col="neighbors",
                    col_wrap=2,
                    hue="label",
                    hue_order=sorted(labels.unique()),
                    kind="scatter",
                    palette=PALETTE,
                    facet_kws={"sharex": False, "sharey": False},
                    s=point_size,
                )
            else:
                p = sns.scatterplot(
                    data=df_dimred,
                    x="Dim1",
                    y="Dim2",
                    hue="label",
                    hue_order=sorted(labels.unique()),
                    palette=PALETTE,
                    s=point_size,
                )
        elif self.n_components == 3:
            for n in n_neighbors:
                p = sns.pairplot(
                    df_dimred.query(f"neighbors == {n}").drop("neighbors", axis=1),
                    hue="label",
                    hue_order=sorted(labels.unique()),
                    palette=PALETTE,
                    aspect=2,
                    s=point_size,
                )
                p.fig.suptitle(f"Number of Neighbors: {n}", y=1.08)
        else:
            raise ValueError(
                "Only 2 or 3 number of components (dimension) can be displayed."
            )
        p.figure.set_size_inches(1.5 * WIDTH_FIG_FULLPAGE, 1.5 * WIDTH_FIG_FULLPAGE)
        plt.show()
        return df_dimred


class Viz:

    def __init__(self, config: ConfigParser):
        # Connect to the tables and query

        self.config = config

    def users_stats(
        df: pd.DataFrame,
        order_col: Optional[list] = None,
        ticker_moltip: Optional[int] = None,
        colors: Optional[list] = None,
    ) -> None:
        """Display report counts given the classification labels."""

        fig = plt.figure()
        fig.set_size_inches(WIDTH_FIG_FULLPAGE, 8 * CM)
        ax1 = fig.gca()
        ax2 = ax1.twinx()

        if order_col is not None:
            emtpy_cols = list(set(order_col) - set(df.columns))
            if len(emtpy_cols) > 0:
                df[emtpy_cols] = 0
            df = df[order_col]

        df_perc = (df.div(df.sum(axis=1), axis=0) * 100).fillna(0)
        df_tot = df.sum(axis=1)
        df_tot.index = df_tot.index.astype(str)
        df_perc.index = df_perc.index.astype(str)

        if colors is None:
            df_perc.plot(
                kind="bar",
                rot=30,
                cmap=PALETTE,
                width=1,
                stacked=True,
                ax=ax1,
                legend=False,
            )
        else:
            df_perc.plot(
                kind="bar",
                rot=30,
                color=colors,
                width=1,
                stacked=True,
                ax=ax1,
                legend=False,
            )
        df_tot.plot(ax=ax2, color=LINE_COLOR, linewidth=2)

        ax1.set_ylabel("User distribution (%)")
        ax2.set_ylabel("Total reports", color=LINE_COLOR)
        ax1.legend(title=None)

        if ticker_moltip is not None:
            ax1.xaxis.set_major_locator(ticker.MultipleLocator(ticker_moltip))

    def level_users_stats(
        self,
        df: pd.DataFrame,
        level: int,
        ticker_moltip: Optional[int] = None,
        stacked: bool = True,
        is_palette: bool = True,
        save_data: bool = False,
        title: str = "Total",
    ) -> None:
        """Display report counts given the classification labels for different levels of depth."""

        if stacked:
            width = 1
        else:
            width = 0.5

        if level == 1:
            color = [
                "#669900ff",
                "#003399ff",
                "#ff6600ff",
                "#bfbfbfff",
            ]
            label = ["multi-clustered", "clustered", "sparse", "sporadic"]
        elif level == 2:
            color = [
                "#669900ff",
                "#003399ff",
                "#ff6600ff",
                "#669900ff",
                "#003399ff",
                "#ff6600ff",
                "#bfbfbfff",
            ]
            label = [
                "HQ-multi-clustered",
                "HQ-clustered",
                "HQ-sparse",
                "LQ-multi-clustered",
                "LQ-clustered",
                "LQ-sparse",
                "sporadic",
            ]
            hs, ls = slice(None, 3), slice(3, -1)
            hq, lq = label[hs], label[ls]
            hc, lc = color[hs], color[ls]
        elif level == 3:
            label = [
                "HQ-multi-clustered",
                "HQ-clustered narrow",
                "HQ-clustered wide",
                "HQ-sparse narrow",
                "HQ-sparse wide",
                "LQ-multi-clustered",
                "LQ-clustered narrow",
                "LQ-clustered wide",
                "LQ-sparse narrow",
                "LQ-sparse wide",
                "sporadic",
            ]
            color = [
                "#669900ff",
                "#003399ff",
                "#2971ffff",
                "#ff6600ff",
                "#ffcc00ff",
                "#669900ff",
                "#003399ff",
                "#2971ffff",
                "#ff6600ff",
                "#ffcc00ff",
                "#bfbfbfff",
            ]
            hs, ls = slice(None, 5), slice(5, -1)
            hq, lq = label[hs], label[ls]
            hc, lc = color[hs], color[ls]
        else:
            KeyError(level)

        l_miss = list(set(label).difference(set(df.columns)))
        if len(l_miss) > 0:
            df[l_miss] = np.nan
            df = df[label]
        df_perc = (df.div(df.sum(axis=1), axis=0) * 100).fillna(0)
        df_perc.index = df_perc.index.astype(str)
        df_tot = df.sum(axis=1)
        df_tot.index = df_tot.index.astype(str)

        if level == 1:
            fig = plt.figure()
            fig.set_size_inches(WIDTH_FIG_FULLPAGE, 8 * CM)
            ax1 = fig.gca()
            if is_palette:
                df_perc.plot(
                    kind="bar",
                    cmap=PALETTE,
                    width=width,
                    stacked=stacked,
                    rot=30,
                    ax=ax1,
                )
            else:
                df_perc.plot(
                    kind="bar",
                    color=color,
                    width=width,
                    stacked=stacked,
                    rot=30,
                    ax=ax1,
                )
            ax2 = ax1.twinx()
            df_tot.plot(ax=ax2, color=LINE_COLOR, linewidth=2)
            ax1.set_ylabel("User distribution (%)")
            ax2.set_ylabel("Total reports", color=LINE_COLOR)
            ax1.legend(title=None)

        elif level in [2, 3]:
            fig, ax = plt.subplots(2, 1, sharex=True)
            fig.set_size_inches(WIDTH_FIG_FULLPAGE, 10 * CM)
            df_perc_hq = df_perc[hq]
            if is_palette:
                df_perc_hq.plot(
                    kind="bar", cmap=PALETTE, width=width, stacked=stacked, ax=ax[0]
                )
                df_perc_lq = df_perc[lq]
                df_perc_lq.plot(
                    kind="bar",
                    cmap=PALETTE,
                    width=width,
                    stacked=stacked,
                    rot=30,
                    ax=ax[1],
                )
            else:
                df_perc_hq.plot(
                    kind="bar", color=hc, width=width, stacked=stacked, ax=ax[0]
                )
                df_perc_lq = df_perc[lq]
                df_perc_lq.plot(
                    kind="bar",
                    color=lc,
                    width=width,
                    stacked=stacked,
                    rot=30,
                    ax=ax[1],
                )

            ax0_tot = ax[0].twinx()
            ax1_tot = ax[1].twinx()
            df_tot.plot(ax=ax0_tot, color=LINE_COLOR, linewidth=2)
            df_tot.plot(ax=ax1_tot, color=LINE_COLOR, linewidth=2)
            ax[0].set_ylabel("User distribution (%)")
            ax0_tot.set_ylabel("Total reports", color=LINE_COLOR)
            ax[1].set_ylabel("User distribution (%)")
            ax1_tot.set_ylabel("Total reports", color=LINE_COLOR)
            ax1 = ax[1]
            ax[0].legend(title=None)
            ax[1].legend(title=None)

        if ticker_moltip is not None:
            ax1.xaxis.set_major_locator(ticker.MultipleLocator(ticker_moltip))

        if save_data:
            df_perc.to_parquet(
                os.path.join(
                    self.config["paths"]["results"],
                    f"level{level}_users_stats_{title}.csv",
                )
            )

    def labels_boxplot(
        users_stats_filt: pd.DataFrame, unsupervised_labels: np.array, title: str
    ) -> None:
        """Distribution of feature values against cluster classes"""

        fig, ax = plt.subplots(2, 1, sharex=True)

        for i, c in enumerate(["log_std_utm", "quality"]):
            sns.boxplot(
                users_stats_filt,
                x=unsupervised_labels,
                y=c,
                hue="custom_labels_1",
                ax=ax[i],
            )

        ax[0].legend(title=None, bbox_to_anchor=(1.0, 1.3), ncols=3)
        ax[1].set_xlabel("Cluster-class")
        ax[1].legend().set_visible(False)
        fig.set_size_inches(WIDTH_FIG_HALFPAGE, 8 * CM)

    def dimred_scatterplot(X: pd.DataFrame, hue: str) -> None:
        p = sns.scatterplot(data=X, x="Dim1", y="Dim2", hue=hue, palette=PALETTE, s=8)
        p.figure.set_size_inches(WIDTH_FIG_HALFPAGE, 8 * CM)

    def users_quality(
        self,
        users_stats: pd.DataFrame,
        metric: Literal["quality", "accuracy", "adjusted accuracy"] = "quality",
    ) -> None:

        fig, ax = plt.subplots(1, 2, sharex=True)
        fig.set_size_inches(WIDTH_FIG_FULLPAGE, 6 * CM)

        m = {
            "quality": "quality",
            "accuracy": "accuracy",
            "adjusted accuracy": "accuracy_chance_adj",
        }
        sporadic_users = users_stats.query("custom_labels_1=='sporadic'")[m[metric]]
        motivated_users = users_stats.query("custom_labels_1!='sporadic'")[m[metric]]
        sporadic_users.hist(ax=ax[0])
        motivated_users.hist(ax=ax[1])
        ax[0].set_title(f"Sporadic users (avg.{metric}: {sporadic_users.mean():.2f})")
        ax[1].set_title(f"Motivated users (avg.{metric}: {motivated_users.mean():.2f})")

    def active_users(
        self,
        qe: QueryEngine,
        retention_interval: List[str] = ["1 year", "6 months", "3 months", "1 months"],
        entity: str = "Total",
        start_date: Optional[str] = None,
        colors: Optional[str] = None,
    ):
        fig = plt.figure()
        fig.set_size_inches(WIDTH_FIG_HALFPAGE, 6 * CM)
        ax = fig.gca()

        tmp = []
        for ri in retention_interval:
            df = qe.user_activity(retention_interval=ri, entity=entity)
            ds = df["active_perc"]
            ds.name = ri
            tmp.append(ds)
        df = pd.concat(tmp, axis=1)
        if start_date is not None:
            df = df.loc[start_date:]
        if colors is not None:
            df.plot(ax=ax, color=colors, rot=30)
        else:
            df.plot(rot=30, ax=ax)
        ax.set_ylabel("Active Users (%)")
        ax.set_title(entity)

    def user_retention(
        self,
        qe: QueryEngine,
        entities: List[str],
        freq: str = "month",
        start_date: Optional[str] = None,
        colors: Optional[str] = None,
    ):

        fig = plt.figure()
        fig.set_size_inches(WIDTH_FIG_HALFPAGE, 6 * CM)
        ax = fig.gca()

        tmp = []
        for entity in entities:
            df = qe.user_retention(freq=freq, entity=entity)
            df = df.rename(columns={"user_retention": entity})
            tmp.append(df)

        df = pd.concat(tmp, axis=1)
        if start_date is not None:
            df = df.loc[start_date:]
        if colors is not None:
            df.plot(ax=ax, color=colors, rot=30)
        else:
            df.plot(ax=ax, rot=30)
        ax.set_ylabel("User Retention (%)")


def users_competence(config: ConfigParser) -> Tuple[pd.DataFrame, float, pd.DataFrame]:

    reports = pd.read_parquet(config["paths"]["reports_transf"]).set_index(
        "version_uuid"
    )
    reports_response = pd.read_parquet(
        config["paths"]["reports_response_transf"]
    ).set_index("version_uuid")

    reports_adult = reports.query("report_type=='adult'")

    species = ["aegypti", "albopictus", "culex", "japonicus", "koreicus"]
    labels = reports_response.join(
        reports_adult.query("validated==True & labels in @species")[
            ["labels", "confidence", "user_id"]
        ]
    ).dropna(subset=["cs_labels", "labels"])
    labels["coincidence"] = labels["cs_labels"] == labels["labels"]

    # Estimate global accuracy by grouping all CS in a super-CS annotator
    cs_global_competence: dict = {}
    cs_global_competence["class_report"] = pd.DataFrame.from_dict(
        classification_report(labels["labels"], labels["cs_labels"], output_dict=True)
    )

    # Scott's Pi agreement adapted for multi-class classification assessment.
    # Krippendorff's alpha and Scott's Pi are equivalent for the case of two raters.
    # Here we assume that the "second rater" provides gold-standard labels.
    Po = labels["coincidence"].sum() / len(labels)
    Pe = np.power(labels["labels"].value_counts() / len(labels), 2).sum()
    cs_global_competence["chance_adj_accuracy"] = (Po - Pe) / (1 - Pe)

    ct = reports_response.value_counts("cs_labels")
    perc_low_acc_cs_labels = (
        (
            ct.loc[["aegypti", "koreicus", "japonicus"]].sum()
            + reports_response["cs_labels"].isna().sum()
        )
        / len(reports_response)
        * 100
    )
    print(
        f"Overall CS chance adjusted accuracy: {cs_global_competence['chance_adj_accuracy']:0.2f}"
    )
    print(f"Percentage of low accuracy cs-labels: {perc_low_acc_cs_labels:0.1f}")
    print("Overall CS classification report:")
    print(cs_global_competence["class_report"])

    # Estimate individual CS accuracies
    labels_ = labels.reset_index()[["user_id", "coincidence"]]
    cs_acc = labels_.groupby("user_id").count()
    cs_acc["accuracy"] = labels_.groupby("user_id").sum() / cs_acc
    cs_acc["accuracy_chance_adj"] = (cs_acc["accuracy"] - Pe) / (1 - Pe)
    cs_acc.columns = ["accuracy_support", "accuracy", "accuracy_chance_adj"]
    cs_acc = cs_acc.sort_values("accuracy_support")

    return labels, cs_global_competence, cs_acc
