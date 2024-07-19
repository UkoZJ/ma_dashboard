# %%
import panel as pn
import param
import numpy as np


class BarPlot(param.Parameterized):
    bar0 = param.Integer(bounds=(1, 5), default=2)
    bar1 = param.Integer(bounds=(1, 100), default=50)
    bar2 = param.Integer(bounds=(1, 10), default=5)

    def __init__(self, **params):
        super().__init__(**params)

        if pn.state.location:
            pn.state.location.sync(self, {"bar1": "int", "bar2": "int"})

        self.view = pn.Tabs(
            ("Forecast", pn.Row(self.param)),
            (
                "Raw Data",
                pn.Row(self.param, self.plot),
            ),
        )

    @param.depends("bar0", "bar1", "bar2")
    def plot(self):
        data = list(np.arange(0, self.bar2) * self.bar1)
        xdata = list(range(0, len(data)))
        series = []
        legend_names = []
        for i in range(0, self.bar0):
            series.append(
                {"name": f"series_{i}", "stack": "1", "type": "bar", "data": data}
            )
            legend_names.append(f"series_{i}")
        echart_bar = {
            "title": {"text": "ECharts entry example"},
            "tooltip": {},
            "legend": {"data": legend_names},
            "xAxis": {"data": xdata},
            "yAxis": {},
            "series": series,
        }

        self.raw_data_plot = pn.pane.ECharts(
            renderer="svg",
            height=480,
            sizing_mode="stretch_width",
            name="Raw Data",
        )

        self.raw_data_plot.object = dict(echart_bar, responsive=True)
        return self.raw_data_plot


bar_plot = BarPlot(name="Bar Plot")

bar_plot.view.show()

# %%
# %%
