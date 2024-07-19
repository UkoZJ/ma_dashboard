from typing import List, Optional, Literal
import numpy as np
import pandas as pd
from bokeh.palettes import brewer, tol
import gettext
from babel.dates import format_date

PALETTE = "Accent"
PALETTE_N = lambda n: brewer[PALETTE][n]
FREQ = Literal["day", "yearweek", "month", "year"]

# eCharts labels (chinese -> english)
back_zoom = "one-step back zoom"
restore_title = "view reset"
saveAsImage_title = "save image"
dataView_title = "data table"
dataView_lang = ["Table of raw data", "turn off", "refresh"]
text_lineColor = "#eeeeee"
grayColor = "#505765"
purpleColor = "#8080ff"
backgroundColor = "#212529"


def make_color_palette(cols: List[str]) -> List[str]:
    n_cols = len(cols)
    n_cols_ = 3 if n_cols < 3 else n_cols
    color = PALETTE_N(n_cols_)[:n_cols]
    return color


def date_format(dates, freq: FREQ = "day", lang="en", default=False):
    """Date formatting relative to language standards"""

    default_format = "short"
    format_dict = {
        "en": {
            "day": "MMM-d\nyyyy",
            "yearweek": "MMM-d\nyyyy",
            "month": "MMM\nyyyy",
            "quarter": "MMM\nyyyy",
            "year": "yyyy",
        },
        "ca": {
            "day": "d MMM\nyyyy",
            "yearweek": "d MMM\nyyyy",
            "month": "MMM\nyyyy",
            "quarter": "MMM\nyyyy",
            "year": "yyyy",
        },
        "es": {
            "day": "d 'de' MMM\nyyyy",
            "yearweek": "d 'de' MMM\nyyyy",
            "month": "MMM\nyyyy",
            "quarter": "MMM\nyyyy",
            "year": "yyyy",
        },
    }

    return [format_date(d, format=format_dict[lang][freq], locale=lang) for d in dates]


def no_data(lang="es"):
    # Text Translation
    t = gettext.translation(
        lang, localedir="translations", languages=[lang], fallback=True
    )
    _ = t.gettext

    echart = {
        "backgroundColor": backgroundColor,
        "title": {
            "show": "true",
            "textStyle": {"color": text_lineColor, "fontSize": 20},
            "text": "No Data",
            "left": "center",
            "top": "30%",
        },
        "xAxis": {"show": "false"},
        "yAxis": {"show": "false"},
        "series": [],
    }
    return echart


def raw_data(
    df: pd.DataFrame,
    lang: str = "en",
    freq: FREQ = "day",
    grid: dict = {},
    legend: dict = {"left": "10%", "top": "25%"},
    title: Optional[str] = None,
    is_dataZoom: bool = True,
    is_sorted_cols: bool = True,
    series_opt: dict = {
        "type": "bar",
        "step": "middle",
        "areaStyle": {},
        "symbol": "none",
        "stack": "1",
        "barCategoryGap": "0%",
        "lineStyle": {"width": 0.5, "type": "solid", "color": text_lineColor},
        # "itemStyle": {"borderWidth": 0.5, "borderColor": backgroundColor},
    },
    ylabel: str = "",
    ymax: Optional[float] = None,
):
    """Plot of raw data of MA"""

    # Text Translation
    t = gettext.translation(
        lang, localedir="translations", languages=[lang], fallback=True
    )
    _ = t.gettext

    # Plot labels
    yAxisLabel = [ylabel]

    # Build echart series with styling (xdata, series)
    xdate = date_format(dates=df.index.date, lang=lang, freq=freq, default=False)

    # Plot setup
    if is_sorted_cols:
        cols = sorted(df.columns)
    else:
        cols = df.columns
    color = make_color_palette(cols)
    colsLabel = [_(col_name.capitalize().replace("_", " ")) for col_name in cols]
    series = []
    for i, col in enumerate(cols):
        ydata = list(df.loc[:, col])

        series.append(
            {
                "name": colsLabel[i],
                "data": ydata,
                **series_opt,
            }
        )

    dataZoom = []
    if is_dataZoom:
        dataZoom = [
            {
                "type": "slider",
                "realtime": True,
                "start": 0,
                "end": 100,
                "bottom": "2%",
                "showDataShadow": False,
                "textStyle": {"color": text_lineColor, "fontSize": 8},
            },
            {
                "type": "inside",
                "realtime": True,
                "start": 0,
                "end": 100,
                "bottom": "2%",
            },
        ]
    set_title = {}
    if title is not None:
        set_title = {
            "text": title,
            "left": 90,
            "top": "7%",
            "textStyle": {
                "color": text_lineColor,
                "fontSize": 14,
                "align": "left",
            },
            "subtextStyle": {"color": purpleColor},
        }

    echart = {
        "backgroundColor": backgroundColor,
        "title": set_title,
        # "color": color,
        "toolbox": {
            "right": "15%",
            "feature": {
                "dataZoom": {
                    "title": {"zoom": "zoom", "back": _(back_zoom)},
                    "yAxisIndex": False,
                },
                "restore": {"title": _(restore_title)},
                "saveAsImage": {"title": _(saveAsImage_title)},
                "dataView": {
                    "title": _(dataView_title),
                    "lang": [_(i) for i in dataView_lang],
                },
            },
        },
        "tooltip": {
            "trigger": "axis",
            # "triggerOn": "click",
            "triggerOn": "mousemove|click",
            "axisPointer": {
                "type": "cross",
                "animation": False,
                "label": {"backgroundColor": grayColor},
            },
        },
        "legend": {
            "data": colsLabel,
            "left": legend["left"],
            "top": legend["top"],
            "orient": "vertical",
            "textStyle": {"color": text_lineColor},
        },
        # "axisPointer": {"link": [{"xAxisIndex": [0, 1]}], "label": {"precision": 0}},
        "dataZoom": dataZoom,
        "grid": grid,
        "xAxis": {
            "type": "category",
            "data": xdate,
            "axisPointer": {"type": "shadow"},
            "position": "bottom",
            "nameTextStyle": {"color": text_lineColor, "fontStyle": "normal"},
            "axisLine": {"lineStyle": {"color": text_lineColor}},
        },
        "yAxis": {
            "name": yAxisLabel[0],
            "nameLocation": "end",
            "nameGap": 10,
            "type": "value",
            "max": ymax,
            "min": 0,
            "boundaryGap": [0, "10%"],
            "splitLine": {"show": False},
            "axisLine": {"show": True, "lineStyle": {"color": text_lineColor}},
            "axisTick": {"show": True, "lineStyle": {"color": text_lineColor}},
            "nameTextStyle": {"color": text_lineColor, "fontStyle": "normal"},
        },
        "series": series,
    }

    return echart


def raw_data_labels(df, lang="es", halfpie=False):
    # Text Translation
    t = gettext.translation(
        lang, localedir="translations", languages=[lang], fallback=True
    )
    _ = t.gettext

    # Plot setup
    # cols = sorted(df.columns)
    cols = df.columns
    color = make_color_palette(cols)
    colsLabel = [
        f'{_(col_name.capitalize().replace("_", " "))} ({df.loc["perc_ct", col_name]:0.1f}%)'
        for col_name in cols
    ]
    ct_tot = df.loc["ct"].sum()

    data = []
    for i, col in enumerate(cols):
        data.append(
            {
                "value": df.loc["ct", col],
                "name": colsLabel[i],
                # "itemStyle": {"color": color[i]},
            }
        )
    if halfpie:
        data.append(
            {
                "value": df.loc["ct", :].sum(),
                "itemStyle": {"color": "none", "decal": {"symbol": "none"}},
                "label": {"show": False},
            }
        )

    echart = {
        "tooltip": {"trigger": "item"},
        "series": [
            {
                "type": "pie",
                "radius": ["40%", "70%"],
                # "center": ["50%", "50%"],
                "startAngle": 180,
                "label": {"show": True, "color": text_lineColor},
                "labelLine": {"lineStyle": {"color": text_lineColor}},
                "itemStyle": {
                    "borderRadius": 10,
                    "borderColor": backgroundColor,
                    "borderWidth": 2,
                },
                "data": data,
            },
        ],
        "graphic": {
            "elements": [
                {
                    "type": "text",
                    "left": "center",
                    "top": "45%",
                    "style": {
                        "text": "Overall",
                        "fontSize": 14,
                        "fill": text_lineColor,
                        "fontWeight": "normal",
                    },
                },
                {
                    "type": "text",
                    "left": "center",
                    "top": "center",
                    "style": {
                        "text": f"{ct_tot:,.0f}",
                        "fontSize": 18,
                        "fill": text_lineColor,
                        "fontWeight": "bold",
                    },
                },
            ]
        },
    }

    return echart


def overview_time2finish(
    df: pd.DataFrame,
    title: Optional[dict] = {},
    lang: str = "en",
    freq: FREQ = "day",
    grid: dict = {},
):
    # Text Translation
    t = gettext.translation(
        lang, localedir="translations", languages=[lang], fallback=True
    )
    _ = t.gettext

    df["iqr"] = df["perc75"] - df["perc25"]
    cols_delay = ["median", "perc25", "iqr"]

    # Plot setup
    colsLabel = [_("Delay (days)"), _("Accumulation")]
    ciLabel = [_("Median"), _("Q1"), _("IQR")]

    # Select variables (columns) to visualize
    df[cols_delay] = df[cols_delay].round(2)
    df[cols_delay] = df[cols_delay].replace({np.nan: "-"})

    # Build echart series with styling (xdata, series)
    xdate = date_format(dates=df.index.date, lang=lang, freq=freq, default=False)

    data_low = df["perc25"].tolist()
    data_range = df["iqr"].tolist()
    data_median = df["median"].tolist()
    data_acc = df["ct_acc"].tolist()

    yAxisColor = [text_lineColor, purpleColor]
    yAxis = []
    for i in range(len(colsLabel)):
        yAxis.append(
            {
                "name": colsLabel[i],
                "nameLocation": "end",
                "nameGap": 10,
                "type": "log",
                "logBase": 10,
                "splitNumber": 3,
                "splitLine": {"show": False},
                "axisLine": {"show": True},
                "axisTick": {"show": True},
                "splitLine": {"show": False},
                "axisLine": {"show": True, "lineStyle": {"color": yAxisColor[i]}},
                "nameTextStyle": {"color": yAxisColor[i], "fontStyle": "normal"},
            }
        )

    if title is not None:
        set_title = {
            "text": title,
            "left": "10%",
            "top": "5%",
            "textStyle": {
                "color": text_lineColor,
                "fontSize": 14,
                "align": "left",
            },
        }
    echart = {
        "title": set_title,
        "xAxis": {
            #'name': xLabel,
            "type": "category",
            "data": xdate,
            "boundaryGap": False,
            "splitLine": {"show": False},
            "axisLine": {"show": True, "lineStyle": {"color": text_lineColor}},
            "axisTick": {"show": True, "lineStyle": {"color": text_lineColor}},
            "nameTextStyle": {"color": text_lineColor, "fontStyle": "normal"},
        },
        "yAxis": yAxis,
        "grid": grid,
        "axisPointer": {"label": {"precision": 0}},
        "tooltip": {
            "trigger": "axis",
            # "triggerOn": "click",
            "triggerOn": "mousemove|click",
            "formatter": "{b0}<br />{a3}: <b>{c3}</b><br />{a2}: <b>{c2}</b><br />{a1}: <b>{c1}</b><br />{a0}: <b>{c0}</b>",
            "axisPointer": {
                "axis": "x",
                "type": "shadow",
                "animation": False,
                "label": {
                    "backgroundColor": grayColor,
                },
            },
        },
        "toolbox": {
            "right": "15%",
            "feature": {
                "dataZoom": {
                    "title": {"zoom": "zoom", "back": _(back_zoom)},
                    "yAxisIndex": False,
                },
                "restore": {"title": _(restore_title)},
                "saveAsImage": {"title": _(saveAsImage_title)},
                "dataView": {
                    "title": _(dataView_title),
                    "lang": [_(i) for i in dataView_lang],
                },
            },
        },
        "series": [
            {
                "name": f"{colsLabel[0]} {ciLabel[1]}",
                "type": "line",
                "step": "middle",
                "data": data_low,
                "lineStyle": {"color": "gray", "opacity": 0.3},
                "stack": "CI",
                "symbol": "none",
                "yAxisIndex": 0,
            },
            {
                "name": f"{colsLabel[0]} {ciLabel[2]}",
                "type": "line",
                "step": "middle",
                "data": data_range,
                "lineStyle": {"color": "gray", "opacity": 0.3},
                "areaStyle": {"color": "gray", "opacity": 0.5},
                "stack": "CI",
                "symbol": "none",
                "yAxisIndex": 0,
            },
            {
                "name": f"{colsLabel[0]} {ciLabel[0]}",
                "type": "line",
                "step": "middle",
                "data": data_median,
                "lineStyle": {"width": 2, "color": text_lineColor},
                "symbol": "none",
                "yAxisIndex": 0,
            },
            {
                "name": colsLabel[1],
                "type": "line",
                "step": "middle",
                "data": data_acc,
                "lineStyle": {"width": 2, "color": purpleColor},
                "symbol": "none",
                "yAxisIndex": 1,
            },
        ],
    }

    return echart


def kpi(
    title: str,
    value: str,
    valueColor: str = text_lineColor,
    fSize: int = 24,
    sfSize: int = 18,
) -> dict:

    echart = {
        "responsive": True,
        "title": {
            "text": title,
            "subtext": value,
            "textStyle": {
                "color": text_lineColor,
                "fontWeight": "bold",
                "fontSize": fSize,
            },
            "subtextStyle": {
                "color": valueColor,
                "fontWeight": "bold",
                "fontSize": sfSize,
            },
        },
    }

    return echart
