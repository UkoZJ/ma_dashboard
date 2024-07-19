# %%
import panel as pn
import pandas as pd
import param
import numpy as np
import src.query_engine as query_engine
from babel.dates import format_date

pn.extension("echarts")


def date_format(dates, lang="en", default=False):
    """Date formatting relative to language stardards"""

    default_format = "short"

    if lang == "en":
        f = default_format if default else "MMM-d\nyyyy"

    elif lang == "ca":
        f = default_format if default else "d MMM\nyyyy"

    elif lang == "es":
        f = default_format if default else "d 'de' MMM\nyyyy"

    return [format_date(d, format=f, locale=lang) for d in dates]


class BarPlot(param.Parameterized):
    freq = "month"
    scale = "NAME_0"
    report_type = "adult"
    entity_label = "Entity"

    # Initialize parameters widgets
    entities = query_engine.get_entity_opts(scale)
    entity = param.Selector(default=entities[0], objects=entities, label=entity_label)

    def __init__(self, **params):
        super().__init__(**params)

        self.view = pn.Tabs(
            ("Row data", pn.Row(self.param, self.plot)), ("Forecast", "lalala")
        )

        if pn.state.location:
            pn.state.location.sync(
                self,
                {"entity": "str"},
            )

    @param.depends("entity")
    def plot(self):
        df_pv = query_engine.filter_view(
            freq=self.freq,
            scale=self.scale,
            report_type=self.report_type,
            entity=self.entity,
        )

        all_cols = [
            "aegypti",
            "albopictus",
            "culex",
            "japonicus",
            "koreicus",
            "other",
            "other_species",
            "unknown",
            "spam",
        ]

        df = pd.DataFrame(columns=all_cols, index=df_pv.index)
        df[df_pv.columns] = df_pv

        xdate = date_format(dates=df.index.date, lang="en", default=False)
        cols = sorted(df.columns)
        series = []

        for col in cols:
            data = list(df.loc[:, col])
            series.append({"name": col, "stack": "1", "type": "bar", "data": data})

        echart_bar = {
            "title": {"text": "ECharts entry example"},
            "tooltip": {},
            "legend": {"data": cols},
            "xAxis": {"data": xdate},
            "yAxis": {"boundaryGap": "10%"},
            "series": series,
        }

        raw_data_plot = pn.pane.ECharts(
            renderer="svg",
            height=480,
            sizing_mode="stretch_width",
            name="Raw Data",
        )

        raw_data_plot.object = dict(echart_bar, responsive=True)
        return raw_data_plot


bar_plot = BarPlot(name="Bar Plot")

bar_plot.view.servable()

# %%
# %%
