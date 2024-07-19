# %%
import gettext
from functools import partial
from io import StringIO, BytesIO
import os
import pandas as pd
import numpy as np
import panel as pn
import param
from panel.pane import ECharts
from panel.template import BootstrapTemplate
from bokeh.palettes import brewer

from src.query_engine import QueryEngine
import src.viz as viz
from src.utils import get_config

# Color-code for adults, bites and sites provided by Enric (white theme)
COLOR_ECHARTS = ["#DFD458", # adult
                 "#D28A73", # bite
                 "#7D9393"  # site
                ]

# Adjusted color-code for black theme
HEADER_COLOR = "#EDB20C"
COLOR_USERS = ["#505765", "#73c0de", "#3ba272"]
COLOR_ECHARTS = [
    "#fac858", # adults
    "#ee6666", # bites
    "#5470c6", # sites
    "#91cc75",
    "#73c0de",
    "#3ba272",
    "#fc8452",
    "#9a60b4",
    "#ea7ccc",
]


class MosquitoAlertExplorer(param.Parameterized):
    """Panel application for MA visualization"""

    # Start DuckDB query engine
    qe = QueryEngine(config=get_config([".config.session.ini"], path="./config/"))
    # Setup the default language
    lang = "en"

    # Application labels
    t = gettext.translation(
        lang, localedir="translations", languages=[lang], fallback=True
    )
    _ = t.gettext

    param_name = "### " + _("Parameters")
    info_name = "### " + _("User Guide")
    default_doc = _("General")
    title_inflow = _("Reports Inflow")
    title_time2finish = _("Monitoring Performance")
    title_retention = _("User Participation")
    title_coverage = _("Human-Sensor Coverage")
    ylabel_counts = _("Counts")
    ylabel_coverage = _("Coverage (%)")
    ylabel_retention = _("(%)")
    overview_table_names = [_("Adult Species"), _("Mosquito Bites"), _("Breeding Sites"), _("Total Reports"), _("New Users"), _("Active Users")]

    docs_button_name = _("Definitions")

    freq_label = _("Frequency")
    freq_dict = {
        # _("Daily"): "day",
        _("Weekly"): "yearweek",
        _("Monthly"): "month",
        _("Quarterly"): "quarter",
        _("Yearly"): "year",
    }
    scale_label = _("Scale")
    scale_dict = {
        _("GADM Level 0"): "name_gadm_level0",
        _("GADM Level 1"): "name_gadm_level1",
        _("GADM Level 2"): "name_gadm_level2",
        _("GADM Level 3"): "name_gadm_level3",
        _("GADM Level 4"): "name_gadm_level4",
        _("Climatic Regions"): "name_climate_regions",
        _("Biome"): "name_biome",
        _("Eco-regions"): "name_ecoregions",
    }
    report_type_label = _("Report Type")
    report_type_dict = {
        _("Adult Species"): "adult",
        _("Mosquito Bites"): "bite",
        _("Breeding Sites"): "site",
    }
    report_type_legend = {
        "adult": [
            "aegypti",
            "albopictus",
            "culex",
            "japonicus",
            "koreicus",
            "other_species",
            "not-sure",
            "spam",
            "unvalidated",
        ],
        "bite": ["Don't know", "Inside a building", "Inside a vehicle", "Outdoors"],
        "site": ["storm_drain", "other_site", "spam", "unvalidated"],
    }
    entity_label = _("Entity")
    entity_dict = dict()

    for s in [*scale_dict.values()]:
        entity_dict[s] = qe.get_entity_opts(s, filter=100)  # get all entities

    user_coverage_dict = {
        "high_density": _("Urban Cluster"),
        "medium_density": _("Rural Cluster"),
        "low_density": _("Rural Low Density"),
        "human_absence": _("Human Absence"),
    }
    user_retention_dict = {
        "new_perc": _("New users"),
        "active_perc": _("Active users"),
        "user_retention": _("User Retention"),
    }
    table_format_label = _("Table format")

    # Initialize parameters widgets
    freqs = [*freq_dict.keys()]
    scales = [*scale_dict.keys()]
    report_type = [*report_type_dict.keys()]
    entities = entity_dict[scale_dict[scales[0]]]

    report_type = param.Selector(
        default=report_type[0],
        objects=report_type,
        label=report_type_label,
        doc="Type of MA report",
    )
    freq = param.Selector(
        default=freqs[2],  # quarterly
        objects=freqs,
        label=freq_label,
        doc="Time base aggregation of data",
    )
    scale = param.Selector(
        default=scales[0],
        objects=scales,
        label=scale_label,
        doc="Base-map of analysis",
    )
    entity = param.Selector(
        default=entities[0],
        objects=entities,
        label=entity_label,
        doc="Location name relative to scale",
    )

    def __init__(self, **params):
        super().__init__(**params)

        if pn.state.location:
            pn.state.location.sync(
                self,
                {
                    "report_type": "report_type",
                    "freq": "freq",
                    "scale": "scale",
                    "entity": "entity",
                },
            )

        # Setup panel view layouts
        self.height = 500
        self.max_width = 1000
        self.width_param = 200
        self.width_settings = 350
        self.height_docs = 500
        self.margin_docs = (0, 20, 0, 20)
        self.style_docs = {
            "overflow-y": "scroll",
            "text-align": "justify",
            "text-justify": "inter-word",
            "padding-right": "20px",
        }

        # Setup plot grids
        grid_kpi = {
            "left": 50,
            "top": 70,
            "right": 50,
            "bottom": 70,
        }
        self.grid_filter = {
            "left": 50,
            "top": 70,
            "right": "5%",
            "bottom": "20%",
        }
        self.grid_inflow = grid_kpi
        self.grid_time2finish = grid_kpi
        self.grid_retention = grid_kpi
        self.grid_coverage = grid_kpi

        # Setup plot legend position
        self.legend_filter = {"left": 60, "top": "20%"}
        self.legend_kpis = {"left": 60, "top": "25%"}

        # Set app documentation
        self.get_docs()  # get markdown files and store them in docs_md
        self.docs_button = pn.widgets.MenuButton(
            name=self.docs_button_name,
            items=self.docs_md,
            button_type="default",
            sizing_mode="stretch_width",
        )
        self.docs = pn.pane.Markdown(
            self.docs_md[0][1],
            height=self.height_docs,
            margin=self.margin_docs,
            styles=self.style_docs,
            # renderer="myst",
        )
        self.docs_button.on_click(self.callback_docs)

        self.update_entity()
        self.view = self.view_init()

    def callback_docs(self, event):
        """User selects documentation interactively"""

        self.docs.object = event.new

    def get_docs(self, select_docs=None):
        """Get the list of all the md files in docs folder for a given language"""

        # Not include the following list of documents
        md_list = []

        path = "./docs/" + self.lang + "/"
        self.docs_md = []
        for dirpath, dnames, fnames in os.walk(path):
            for f in fnames:
                if f.endswith(".md") and (f not in md_list):
                    path = os.path.join(dirpath, f)
                    name = os.path.splitext(f)[0]
                    if (select_docs is None) or (name in select_docs):
                        with open(path, "r") as f:
                            self.docs_md.append((name, f.read()))

        # Sort by name
        self.docs_md = sorted(self.docs_md, key=lambda x: x[0])

    def overview_kpis(self) -> pd.DataFrame:
        report_type_tot = self.qe.overview_view_inflow().sum()
        users_tot = self.qe.user_activity()[["registered", "new", "active"]].sum()
        return pd.concat([report_type_tot, users_tot], axis=0)

    @param.depends("scale", watch=True)
    def update_entity(self):
        """Change entities list in relation to the selected level"""

        entities = [*self.entity_dict[self.scale_dict[self.scale]]]
        self.param["entity"].objects = entities
        self.entity = entities[0]

    @param.depends("report_type", "entity")
    def filter_plot_labels(self):
        report_type = self.report_type_dict[self.report_type]
        df_pv = self.qe.filter_view_labels(
            scale=self.scale_dict[self.scale],
            entity=self.entity,
            report_type=report_type,
        )
        df = df_pv[self.report_type_legend[report_type]]
        filter_labels_pie = ECharts(
            renderer="svg",
            name="Raw Data",
        )

        filter_labels_pie.object = dict(
            viz.raw_data_labels(df, lang=self.lang, color=COLOR_ECHARTS),
            responsive=True,
        )

        return filter_labels_pie

    @param.depends("report_type", "freq", "entity")
    def filter_plot(self):
        report_type = self.report_type_dict[self.report_type]
        df_pv = self.qe.filter_view(
            freq=self.freq_dict[self.freq],
            scale=self.scale_dict[self.scale],
            report_type=report_type,
            entity=self.entity,
        )

        df = df_pv[self.report_type_legend[report_type]]
        df = df.infer_objects(copy=False).fillna(0)
        # self.filt_raw_data = df
        self.filt_raw_data = df_pv  # without empty columns

        raw_data_plot = ECharts(
            renderer="svg",
            height=self.height,
            sizing_mode="stretch_width",
            name=f"Raw data filter_{report_type}",
        )

        raw_data_plot.object = dict(
            viz.raw_data(
                df,
                lang=self.lang,
                freq=self.freq_dict[self.freq],
                grid=self.grid_filter,
                legend=self.legend_filter,
                title=f"{self.report_type}\n{self.scale}\n{self.entity}",
                ylabel=self.ylabel_counts,
                color=COLOR_ECHARTS,
                is_sorted_cols=False
            ),
            responsive=True,
        )

        return raw_data_plot

    def filtered_file_csv(self):
        sio = StringIO()
        self.filt_raw_data.index.name = self.freq.lower()
        self.filt_raw_data.to_csv(sio)
        sio.seek(0)
        return sio

    def filtered_file_parquet(self):
        sio = BytesIO()
        self.filt_raw_data.index.name = self.freq.lower()
        self.filt_raw_data.to_parquet(sio)
        sio.seek(0)
        return sio

    def filtered_file_json(self):
        sio = StringIO()
        df = self.filt_raw_data.copy()
        df.index = df.index.strftime("%Y-%m-%d")
        df.to_json(sio, orient="index")
        sio.seek(0)
        return sio

    @param.depends("scale")
    def overview_plot_rank(self):
        df_reports = self.qe.overview_rank(scale=self.scale_dict[self.scale])
        df_user = self.qe.user_activity_rank(scale=self.scale_dict[self.scale])
        df = pd.concat([df_reports, df_user],axis=1)
        df.index.name = self.scale
        df.columns = self.overview_table_names
        df = df.infer_objects(copy=False).fillna(0)
        formatter_cols = dict()
        filter_cols = dict()
        for i, col in enumerate(df.columns):
            df[col] = df[col].astype(int)
            formatter_cols[col] = {
                "type": "progress",
                "max": df[col].max(),
                "color": COLOR_ECHARTS[i],
                "legend": True,
                "legendColor": "#eeeeee",
            }
            filter_cols[col] = {
                "type": "number",
                "func": ">=",
                "placeholder": "≥",
            }

        filter_cols[df.index.name] = {
            "type": "input",
            "func": "like",
            "placeholder": self.entity_label,
        }
        overview_leaders_table = pn.widgets.Tabulator(
            df,
            theme="bulma",
            pagination=None,
            # pagination="local",
            layout="fit_data_table",
            page_size=10,
            sizing_mode="stretch_both",
            disabled=True,
            stylesheets=[""".pnx-tabulator.tabulator { font-size: 10pt; }"""],
            formatters=formatter_cols,
            header_filters=filter_cols,
        )

        return overview_leaders_table

    @param.depends("freq", "entity")
    def overview_plot_inflow(self):
        df_pv = self.qe.overview_view_inflow(
            freq=self.freq_dict[self.freq],
            scale=self.scale_dict[self.scale],
            entity=self.entity,
        )
        df= df_pv[self.report_type_legend.keys()]
        df = df.replace({np.nan: 0})
        # self.filt_raw_data = df
        self.filt_raw_data = df_pv  # without empty columns

        raw_data_plot = ECharts(
            renderer="svg",
            name=self.title_inflow,
        )

        raw_data_plot.object = dict(
            viz.raw_data(
                df,
                title=self.title_inflow,
                lang=self.lang,
                freq=self.freq_dict[self.freq],
                grid=self.grid_inflow,
                legend=self.legend_kpis,
                is_dataZoom=False,
                ylabel=self.ylabel_counts,
                color=COLOR_ECHARTS,
            ),
            responsive=True,
        )

        return raw_data_plot

    @param.depends("freq", "entity")
    def overview_plot_time2finish(self):
        df_delay = self.qe.overview_time2publish(
            freq=self.freq_dict[self.freq],
            scale=self.scale_dict[self.scale],
            entity=self.entity,
        )
        df_acc = self.qe.overview_view_accumulation(
            freq=self.freq_dict[self.freq],
            scale=self.scale_dict[self.scale],
            entity=self.entity,
        )
        df = pd.concat([df_acc, df_delay], axis=1)

        raw_data_plot = ECharts(
            renderer="svg",
            name=self.title_time2finish,
        )

        raw_data_plot.object = dict(
            viz.overview_time2finish(
                df,
                title=self.title_time2finish,
                lang=self.lang,
                freq=self.freq_dict[self.freq],
                grid=self.grid_time2finish,
            ),
            responsive=True,
        )

        return raw_data_plot

    @param.depends("entity", "freq")
    def overview_plot_user_retention(self):
        df_pv = self.qe.user_retention_stats(
            scale=self.scale_dict[self.scale],
            entity=self.entity,
            freq=self.freq_dict[self.freq],
            # report_type=self.report_type_dict[self.report_type],
        )
        df_pv_perc = df_pv[["new_perc", "active_perc", "user_retention"]]
        cols = self.user_retention_dict.keys()
        df = pd.DataFrame(columns=cols, index=df_pv_perc.index)
        df[df_pv_perc.columns] = df_pv_perc
        df = df.rename(columns=self.user_retention_dict)

        # cols = df.columns
        # cols_format = {
        #     cols[0]: f"{cols[0]} (max:{df_pv['new'].max()})",
        #     cols[2]: f"{cols[2]} ({self.report_type})",
        # }
        # df = df.rename(columns=cols_format)

        raw_data_plot = ECharts(
            renderer="svg",
            name=self.title_retention,
        )

        raw_data_plot.object = dict(
            viz.raw_data(
                df.round(1),
                title=self.title_retention,
                lang=self.lang,
                freq=self.freq_dict[self.freq],
                grid=self.grid_retention,
                legend=self.legend_kpis,
                is_dataZoom=False,
                is_sorted_cols=False,
                series_opt={
                    "type": "line",
                    "step": "middle",
                    "symbol": "none",
                    "lineStyle": {"width": 2, "type": "solid"},
                },
                ylabel=self.ylabel_retention,
                ymax=100,
                color=COLOR_ECHARTS,
            ),
            responsive=True,
        )
        return raw_data_plot

    @param.depends("freq", "entity")
    def overview_plot_coverage(self):
        df_pv = self.qe.overview_sampling_effort(
            freq=self.freq_dict[self.freq],
            scale=self.scale_dict[self.scale],
            entity=self.entity,
        )

        self.filt_coveage_data = df_pv  # without empty columns
        df = pd.DataFrame(columns=self.user_coverage_dict.keys(), index=df_pv.index)
        df[df_pv.columns] = df_pv
        df = df.drop(["human_absence"], axis=1)
        df = df.rename(columns=self.user_coverage_dict)
        df = df.replace({np.nan: "-"})

        coverage_data_plot = ECharts(renderer="svg", name=self.title_coverage)

        coverage_data_plot.object = dict(
            viz.raw_data(
                df.round(3),
                title=self.title_coverage,
                lang=self.lang,
                freq=self.freq_dict[self.freq],
                grid=self.grid_coverage,
                legend=self.legend_kpis,
                is_dataZoom=False,
                is_sorted_cols=False,
                series_opt={
                    "type": "line",
                    "step": "middle",
                    "symbol": "none",
                    "lineStyle": {"width": 2, "type": "solid"},
                },
                ylabel=self.ylabel_coverage,
                color=COLOR_ECHARTS,
            ),
            responsive=True,
        )

        return coverage_data_plot

    def view_init(self, template=True):
        """Build dashboard layout"""

        pn.extension("echarts", "tabulator", "mathjax")

        with open("./docs/pics/dataset_csv.svg") as f:
            data_icon_csv = f.read()
        with open("./docs/pics/dataset_parquet.svg") as f:
            data_icon_parquet = f.read()
        with open("./docs/pics/dataset_json.svg") as f:
            data_icon_json = f.read()

        icon_size = "2em"
        margin = (20, 10, 10, 20)
        button_style = "outline"
        button_type = "light"
        file_download = partial(
            pn.widgets.FileDownload,
            icon_size=icon_size,
            margin=margin,
            button_style=button_style,
            button_type=button_type,
        )
        fd_csv = file_download(
            label="CSV",
            callback=self.filtered_file_csv,
            filename="filtered_table.csv",
            icon=data_icon_csv,
        )
        fd_parquet = file_download(
            label="Parquet",
            callback=self.filtered_file_parquet,
            filename="filtered_table.parquet",
            icon=data_icon_parquet,
        )
        fd_json = file_download(
            label="JSON",
            callback=self.filtered_file_json,
            filename="filtered_table.json",
            icon=data_icon_json,
        )

        self.param.scale.precedence = 1
        self.param.entity.precedence = 2
        self.param.freq.precedence = 3
        self.param.report_type.precedence = 4

        img_size = 120
        img_path = "./docs/pics/"

        divider_margin = (-15, 0, 0, 0)
        settings_pane = [
            self.param_name,
            pn.layout.Divider(margin=divider_margin),
            pn.Param(
                self,
                show_name=False,
                width=self.width_param,
                # widgets={
                #     "entity": {
                #         "widget_type": pn.widgets.AutocompleteInput,
                #         "case_sensitive": False,
                #     }
                # },
                widgets={
                    "entity": {
                        "widget_type": pn.widgets.AutocompleteInput,
                        "case_sensitive": False,
                        "search_strategy": "includes",
                        "restrict": True,
                        "placeholder": "Type here a name",
                    },
                },
            ),
            pn.Spacer(height=30),
            pn.Row(self.info_name, self.docs_button),
            pn.layout.Divider(margin=divider_margin),
            self.docs,
        ]

        layout = BootstrapTemplate(
            title="Mosquito Alert",
            logo="./docs/pics/LOGO_MosquitoAlert_horizontal_positivo_PNG.png",
            site_url="http://www.mosquitoalert.com/",
            favicon="http://www.mosquitoalert.com/wp-content/uploads/2017/02/favicon.png",
            sidebar_width=self.width_settings,
            header_background=HEADER_COLOR,
            theme="dark",
            sidebar=settings_pane,
        )

        kpi = self.overview_kpis()

        kpi_plot = [ECharts(renderer="svg") for i in range(0, 6)]
        kpi_plot[0].object = viz.kpi("Adults", f'{kpi["adult"]:,.0f}', COLOR_ECHARTS[0])
        kpi_plot[1].object = viz.kpi("Bites", f'{kpi["bite"]:,.0f}', COLOR_ECHARTS[1])
        kpi_plot[2].object = viz.kpi("Sites", f'{kpi["site"]:,.0f}', COLOR_ECHARTS[2])
        kpi_plot[3].object = viz.kpi(
            "Users", f'{kpi["registered"]:,.0f}', COLOR_USERS[0]
        )
        kpi_plot[4].object = viz.kpi(
            "New",
            f'{kpi["new"]/kpi["registered"]*100:,.1f}%',
            COLOR_USERS[1],
            fSize=14,
            sfSize=14,
        )
        kpi_plot[5].object = viz.kpi(
            "Active",
            f'{kpi["active"]/kpi["registered"]*100:,.01f}%',
            COLOR_USERS[2],
            fSize=14,
            sfSize=14,
        )

        adults_jpg = pn.pane.Image(
            img_path + "ic_mosquito_report.png",
            width=img_size,
        )
        bites_jpg = pn.pane.Image(
            img_path + "ic_bite_report.png",
            width=img_size,
        )
        sites_jpg = pn.pane.Image(
            img_path + "ic_breeding_report.png",
            width=img_size,
        )
        users_jpg = pn.pane.Image(
            img_path + "users.png",
            width=img_size,
        )
        summary = pn.Column(
            pn.Spacer(height=50),
            pn.Row(
                pn.Spacer(width=50),
                pn.Row(adults_jpg, kpi_plot[0]),
                pn.Row(bites_jpg, kpi_plot[1]),
                pn.Row(sites_jpg, kpi_plot[2]),
                pn.Row(
                    users_jpg,
                    pn.Column(kpi_plot[3], pn.Row(kpi_plot[4], kpi_plot[5], width=150)),
                ),
                height=150,
            ),
        )

        pane_kpis = pn.Column(
            pn.Row(
                self.overview_plot_inflow,
                self.overview_plot_time2finish,
            ),
            pn.Row(
                self.overview_plot_user_retention,
                self.overview_plot_coverage,
            ),
        )
        self.tabs = pn.Tabs(
            (
                self._("Overview"),
                pn.Column(summary, self.overview_plot_rank),
            ),
            (self._("KPIs"), pane_kpis),
            (
                self._("Report Stats"),
                pn.Row(
                    pn.Column(pn.Row(fd_csv, fd_parquet, fd_json), self.filter_plot),
                    self.filter_plot_labels,
                    max_height=600,
                ),
            ),
        )
        layout.main.append(self.tabs)

        return layout
