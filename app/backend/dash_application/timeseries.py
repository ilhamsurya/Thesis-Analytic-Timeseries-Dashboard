"""Instantiate a Dash app."""
import numpy as np
import pandas as pd
import dash
import dash_html_components as html
import dash_core_components as dcc
from .data import create_dataframe
from .layout import timeseries_layout
from app import dash_app1
import plotly.express as px

# Load DataFrame
df = create_dataframe()

available_indicators = df["kategori"].unique()
available_locations = df["tempat_kejadian"].unique()

# Create Layout
timeseries = html.Div(
    [
        html.Div(
            [
                html.Div(
                    [
                        html.H6(
                            """Pilih Kategori Pelanggaran""",
                            style={"margin-right": "2em"},
                        ),
                        dcc.Dropdown(
                            id="crossfilter-kategori",
                            options=[
                                {"label": i, "value": i} for i in available_indicators
                            ],
                            value="Pelanggaran laut kategori pertama",
                            placeholder="Pilih kategori pelanggaran pertama",
                        ),
                    ],
                    style={"width": "49%", "display": "inline-block"},
                ),
                html.Div(
                    [
                        html.H6(
                            """Pilih Lokasi Pelanggaran""",
                            style={"margin-right": "2em"},
                        ),
                        dcc.Dropdown(
                            id="crossfilter-tempat",
                            options=[
                                {"label": y, "value": y} for y in available_locations
                            ],
                            value="Tempat Kejadian Pelanggaran Laut)",
                            placeholder="Pilih tempat pelanggaran pertama",
                        ),
                    ],
                    style={
                        "width": "49%",
                        "float": "right",
                        "display": "inline-block",
                    },
                ),
                html.Div(
                    [
                        dcc.Dropdown(
                            id="crossfilter-kategori-2",
                            options=[
                                {"label": i, "value": i} for i in available_indicators
                            ],
                            value="Pelanggaran laut kategori kedua",
                            placeholder="Pilih kategori pelanggaran kedua",
                        ),
                    ],
                    style={
                        "width": "49%",
                        "float": "left",
                        "display": "inline-block",
                    },
                ),
                html.Div(
                    [
                        html.H6(
                            """Geser Waktu Pelanggaran""",
                            style={"margin-right": "2em"},
                        ),
                        dcc.Slider(
                            id="crossfilter-year--slider",
                            min=df["Tahun"].min(),
                            max=df["Tahun"].max(),
                            value=df["Tahun"].max(),
                            step=None,
                            marks={
                                str(year): str(year) for year in df["Tahun"].unique()
                            },
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
                        html.H6(
                            """Grafik Perbandingan Kategori""",
                            style={"margin-right": "2em"},
                        ),
                        dcc.Graph(
                            id="crossfilter-indicator-scatter",
                            hoverData={"points": [{"customdata": "Laut Halmahera"}]},
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
                        html.H6(
                            """Grafik Trend Kategori""",
                            style={"margin-right": "2em"},
                        ),
                        dcc.Graph(id="x-time-series"),
                        html.H6(
                            """Grafik Season Kategori""",
                            style={"margin-right": "2em"},
                        ),
                        dcc.Graph(id="y-time-series"),
                    ],
                    style={"display": "inline-block", "width": "49%"},
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
    dash.dependencies.Output("crossfilter-indicator-scatter", "figure"),
    [
        dash.dependencies.Input("crossfilter-kategori", "value"),
        dash.dependencies.Input("crossfilter-kategori-2", "value"),
        # dash.dependencies.Input("crossfilter-tempat", "value"),
        dash.dependencies.Input("crossfilter-year--slider", "value"),
    ],
)
def update_graph(xaxis_column_name, yaxis_column_name, year_value):
    dff = df[df["Tahun"] == year_value]
    fig = px.line(
        x=dff[dff["kategori"] == xaxis_column_name]["tanggal"],
        y=dff[dff["kategori"] == yaxis_column_name]["tanggal"],
        # hover_name=dff[dff["kategori"] == yaxis_column_name]["tempat_kejadian"],
    )

    fig.update_traces(
        customdata=dff[dff["kategori"] == yaxis_column_name]["tempat_kejadian"]
    )

    fig.update_xaxes(title=xaxis_column_name, type="linear")

    fig.update_yaxes(title=yaxis_column_name, type="linear")

    fig.update_layout(margin={"l": 40, "b": 40, "t": 10, "r": 0}, hovermode="closest")

    return fig


def create_time_series(dff, title):

    fig = px.line(dff, x="Tahun", y="kategori")

    fig.update_traces(mode="lines+markers")

    fig.update_xaxes(showgrid=False)

    fig.update_yaxes(type="linear")

    fig.add_annotation(
        x=0,
        y=0.85,
        xanchor="left",
        yanchor="bottom",
        xref="paper",
        yref="paper",
        showarrow=False,
        align="left",
        bgcolor="rgba(255, 255, 255, 0.5)",
        text=title,
    )

    fig.update_layout(height=225, margin={"l": 20, "b": 30, "r": 10, "t": 10})

    return fig


@dash_app1.callback(
    dash.dependencies.Output("x-time-series", "figure"),
    [
        dash.dependencies.Input("crossfilter-indicator-scatter", "hoverData"),
        dash.dependencies.Input("crossfilter-kategori", "value"),
    ],
)
def update_y_timeseries(hoverData, xaxis_column_name):
    nama_tempat = hoverData["points"][0]["customdata"]
    dff = df[df["tempat_kejadian"] == nama_tempat]
    dff = dff[dff["kategori"] == xaxis_column_name]
    title = "<b>{}</b><br>{}".format(nama_tempat, xaxis_column_name)
    return create_time_series(dff, title)


@dash_app1.callback(
    dash.dependencies.Output("y-time-series", "figure"),
    [
        dash.dependencies.Input("crossfilter-indicator-scatter", "hoverData"),
        dash.dependencies.Input("crossfilter-kategori-2", "value"),
    ],
)
def update_x_timeseries(hoverData, yaxis_column_name):
    dff = df[df["tempat_kejadian"] == hoverData["points"][0]["customdata"]]
    dff = dff[dff["kategori"] == yaxis_column_name]
    return create_time_series(dff, yaxis_column_name)
