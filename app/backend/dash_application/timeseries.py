"""Instantiate a Dash app."""
from dash.dependencies import Output, Input, State
import numpy as np
import pandas as pd
import dash
import dash_html_components as html
import dash_core_components as dcc
from .data import create_dataframe
from .layout import timeseries_layout
from app import dash_app1
import plotly.express as px


# Data untuk grafik perbandingan kategori semua tahun
df = create_dataframe()
# tempat_kejadian = df['tempat_kejadian']
# df['tanggal'] = pd.to_datetime(df['tanggal'])
# df = df.groupby(['Tahun','kategori','tempat_kejadian'], as_index=False)['ID'].count()

df = df.groupby(["Tahun", "kategori", "tempat_kejadian"], as_index=False)[
    "Frekuensi"
].count()
# df = df.set_index('tanggal')
# df = df.loc['2014-01-01':'2019-12-30']
# df = df.groupby([pd.Grouper(freq="M"), 'kategori'])['ID'].count().reset_index()

# Data untuk grafik seasonal per tahun
# dt = df.groupby([pd.Grouper(freq="M"), 'Tahun'])['ID'].count().reset_index()


# dt = create_dataframe()
# tempat_kejadian = dt['tempat_kejadian']
# dt['tanggal'] = pd.to_datetime(df['tanggal'])
# dt = dt.groupby(['tanggal','kategori'], as_index=False)['ID'].count()
# dt = dt.set_index('tanggal')
# dt = dt.loc['2014-01-01':'2019-12-30']
# dt = dt.groupby([pd.Grouper(freq="M"), 'kategori'])['ID'].count().reset_index()


def show():
    return render_template("kategori.html", kategori=df)


print(df)
# Create Layout
timeseries = html.Div(
    [
        html.Div(
            [
                html.Div(
                    [
                        html.H6(
                            """Tempat Pelanggaran""",
                            style={"margin-right": "2em"},
                        ),
                        dcc.Dropdown(
                            id="crossfilter-tempat",
                            options=[
                                {"label": y, "value": y}
                                for y in df.sort_values("tempat_kejadian")[
                                    "tempat_kejadian"
                                ].unique()
                            ],
                            multi=False,
                            persistence="string",
                            persistence_type="local",
                            placeholder="Pilih tempat pelanggaran",
                        ),
                    ],
                    style={"width": "49%", "display": "inline-block"},
                ),
                html.Div(
                    [
                        html.H6(
                            """Kategori Pelanggaran 1""",
                            style={"margin-right": "2em, font-weight : bold"},
                        ),
                        dcc.Dropdown(
                            id="crossfilter-kategori",
                            options=[
                                {"label": i, "value": i}
                                for i in df.sort_values("kategori")["kategori"].unique()
                            ],
                            clearable=True,
                            className="form-dropdown",
                            placeholder="Pilih kategori pelanggaran pertama",
                        ),
                        html.H6(
                            """Kategori Pelanggaran 2""",
                            style={"margin-right": "2em, font-weight : bolder"},
                        ),
                        dcc.Dropdown(
                            id="crossfilter-kategori-2",
                            options=[
                                {"label": i, "value": i}
                                for i in df.sort_values("kategori")["kategori"].unique()
                            ],
                            clearable=True,
                            className="form-dropdown",
                            placeholder="Pilih kategori pelanggaran kedua",
                        ),
                    ],
                    style={
                        "width": "49%",
                        "float": "left",
                        # "display": "inline-block",
                    },
                ),
                html.Div(
                    [
                        html.H6(
                            """Geser Waktu Pelanggaran""",
                            style={"margin-right": "2em"},
                        ),
                        dcc.RangeSlider(
                            id="select_year",
                            min=2014,
                            max=2019,
                            dots=True,
                            value=[2015, 2019],
                            marks={str(yr): str(yr) for yr in range(2014, 2019)},
                        ),
                    ],
                    style={
                        "width": "100%",
                        "padding": "80px 80px 80px 80px",
                        "color": "black",
                    },
                ),
                html.Div(
                    [
                        dcc.Graph(
                            id="crossfilter-indicator-scatter"
                            # hoverData={"points": [{"customdata": "Laut Halmahera"}]},
                        ),
                    ],
                    style={
                        "width": "49%",
                        "display": "inline-block",
                        "padding": "0 20",
                    },
                ),
                html.Div(
                    [
                        dcc.Graph(
                            id="seasonality"
                            # hoverData={"points": [{"customdata": "Laut Halmahera"}]},
                        ),
                        dcc.Slider(
                            id="PerTahun",
                            min=2014,
                            max=2019,
                            value=2019,
                            step=None,
                            marks={
                                2014: {
                                    "label": "2014",
                                    "style": {
                                        "color": "#77b0b1",
                                        "top-padding": "10px",
                                    },
                                },
                                2015: {"label": "2015"},
                                2016: {"label": "2016"},
                                2017: {"label": "2017"},
                                2018: {
                                    "label": "2018",
                                },
                                2019: {"label": "2019", "style": {"color": "#f50"}},
                                # str(year): str(year) for year in df["Tahun"].unique()
                            },
                        ),
                    ],
                    style={
                        "width": "49%",
                        "display": "inline-block",
                        "padding": "0 20",
                    },
                ),
            ],
            style={
                "borderBottom": "thin lightgrey solid",
                "backgroundColor": "rgb(250, 250, 250)",
                "padding": "10px 5px",
            },
        ),
    ]
)


@dash_app1.callback(
    Output("crossfilter-indicator-scatter", "figure"),
    [
        Input("crossfilter-kategori", "value"),
        Input("crossfilter-kategori-2", "value"),
        Input("crossfilter-tempat", "value"),
        Input("select_year", "value"),
    ],
)
def build_graph(kategori, kategori2, tempatkejadian, year):
    # dff = df.groupby()
    dff = df[
        ((df["kategori"] == kategori) | (df["kategori"] == kategori2))
        & ((df["tempat_kejadian"] == tempatkejadian))
        & ((df["Tahun"] >= year[0]) & (df["Tahun"] <= year[1]))
    ]
    fig = px.line(dff, x="Tahun", y="Frekuensi", color="kategori")
    fig.update_layout(
        yaxis={"title": "Frekuensi"},
        xaxis={"title": "Waktu"},
        title={
            "text": "Grafik Perbandingan Trend",
            "font": {"size": 28},
            "x": 0.5,
            "xanchor": "center",
        },
    )
    print(dff)
    return fig


@dash_app1.callback(
    Output("seasonality", "figure"),
    [Input("crossfilter-kategori", "value"), Input("crossfilter-kategori-2", "value")],
)
# df['kategori'] == kategori) |
def build_graph2(kategori, kategori2):
    dff = df[(df["kategori"] == kategori) | (df["kategori"] == kategori2)]
    fig2 = px.line(dff, x="Tahun", y="ID", color="kategori")
    fig2.update_layout(
        yaxis={"title": "Frekuensi"},
        xaxis={"title": "Waktu"},
        title={
            "text": "Grafik Seasonal Perbandingan Trend",
            "font": {"size": 28},
            "x": 0.5,
            "xanchor": "center",
        },
    )
    return fig2


# # @dash_app1.callback(
# #     dash.dependencies.Output("crossfilter-indicator-scatter", "figure"),
# #     [
# #         dash.dependencies.Input("crossfilter-kategori", "value"),
# #         dash.dependencies.Input("crossfilter-kategori-2", "value"),
# #         # dash.dependencies.Input("crossfilter-tempat", "value"),
# #         dash.dependencies.Input("crossfilter-year--slider", "value"),
# #     ],
# # )
# # def update_graph(xaxis_column_name, yaxis_column_name, year_value):
# #     dff = df[df["Tahun"] == year_value]
# #     fig = px.line(
# #         x=dff[dff["kategori"] == xaxis_column_name],
# #         y=dff[dff["kategori"] == yaxis_column_name],
# #         # hover_name=dff[dff["kategori"] == yaxis_column_name]["tempat_kejadian"],
# #     )

# #     fig.update_traces(
# #         customdata=dff[dff["kategori"] == yaxis_column_name]["tempat_kejadian"]
# #     )

# #     fig.update_xaxes(title=xaxis_column_name, type="linear")

# #     fig.update_yaxes(title=yaxis_column_name, type="linear")

# #     fig.update_layout(margin={"l": 40, "b": 40, "t": 10, "r": 0}, hovermode="closest")

# #     return fig


# # def create_time_series(dff, title):

# #     fig = px.line(dff, x="Tahun", y="kategori", labels={
# #                      "kategori": "Frekuensi",
# #                  })

# #     fig.update_traces(mode="lines+markers")

# #     fig.update_xaxes(showgrid=False)

# #     fig.update_yaxes(type="linear")

# #     fig.add_annotation(
# #         x=0,
# #         y=0.85,
# #         xanchor="left",
# #         yanchor="bottom",
# #         xref="paper",
# #         yref="paper",
# #         showarrow=False,
# #         align="left",
# #         bgcolor="rgba(255, 255, 255, 0.5)",
# #         text=title,
# #     )

# #     fig.update_layout(height=225, margin={"l": 20, "b": 30, "r": 10, "t": 10})

# #     return fig


# # @dash_app1.callback(
# #     dash.dependencies.Output("x-time-series", "figure"),
# #     [
# #         dash.dependencies.Input("crossfilter-indicator-scatter", "hoverData"),
# #         dash.dependencies.Input("crossfilter-kategori", "value"),
# #     ],
# # )
# # def update_y_timeseries(hoverData, xaxis_column_name):
# #     nama_tempat = hoverData["points"][0]["customdata"]
# #     dff = df[df["tempat_kejadian"] == nama_tempat]
# #     dff = dff[dff["kategori"] == xaxis_column_name]
# #     title = "<b>{}</b><br>{}".format(nama_tempat, xaxis_column_name)
# #     return create_time_series(dff, title)


# # @dash_app1.callback(
# #     dash.dependencies.Output("y-time-series", "figure"),
# #     [
# #         dash.dependencies.Input("crossfilter-indicator-scatter", "hoverData"),
# #         dash.dependencies.Input("crossfilter-kategori-2", "value"),
# #     ],
# # )
# # def update_x_timeseries(hoverData, yaxis_column_name):
# #     dff = df[df["tempat_kejadian"] == hoverData["points"][0]["customdata"]]
# #     dff = dff[dff["kategori"] == yaxis_column_name]
# #     return create_time_series(dff, yaxis_column_name)
