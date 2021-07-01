"""Instantiate a Dash app."""
from dash.dependencies import Output, Input, State
import numpy as np
import pandas as pd
import dash
import plotly.graph_objs as go
import dash_html_components as html
import dash_core_components as dcc
from statsmodels.tsa.seasonal import seasonal_decompose
from plotly.missing_ipywidgets import FigureWidget
from .data import create_dataframe, create_dataframe_time_series
from .layout import timeseries_layout
from app import dash_app1
import plotly.express as px


# Data untuk grafik perbandingan kategori semua tahun
df = create_dataframe()

data_timeseries = create_dataframe_time_series()

timeseries = html.Div(
    [
        html.Div(
            [
                dcc.RangeSlider(
                    id="years",
                    min=2014,
                    max=2019,
                    dots=True,
                    value=[2014, 2019],
                    marks={str(yr): "'" + str(yr)[2:] for yr in range(2014, 2019)},
                ),
                html.Br(),
                html.Br(),
            ],
            style={"width": "75%", "margin-left": "12%", "background-color": "#eeeeee"},
        ),
        html.Div(
            [
                dcc.Dropdown(
                    id="kategori",
                    multi=True,
                    value=[""],
                    placeholder="Select kategori",
                    options=[
                        {"label": c, "value": c}
                        for c in sorted(df["kategori"].unique().astype(str))
                    ],
                ),
                dcc.Dropdown(
                    id="lokasi",
                    multi=True,
                    value=[""],
                    placeholder="Select lokasi",
                    options=[
                        {"label": c, "value": c}
                        for c in sorted(df["tempat_kejadian"].unique().astype(str))
                    ],
                ),
            ],
            style={"width": "50%", "margin-left": "25%", "background-color": "#eeeeee"},
        ),
        dcc.Graph(id="by_year_country_world", config={"displayModeBar": False}),
        html.Hr(),
        html.Br(),
        html.Content(
            "Time Series Component", style={"margin-left": "25rem", "font-size": 25}
        ),
        html.Div(
            [
                html.Div(
                    [
                        dcc.Graph(
                            id="trend_analysis",
                            figure={"layout": {"margin": {"r": 10, "t": 50}}},
                            config={"displayModeBar": False},
                        ),
                    ],
                    style={"width": "48%", "display": "inline-block"},
                ),
                html.Div(
                    [
                        dcc.Graph(
                            id="seaonality_analysis",
                            config={"displayModeBar": False},
                            figure={"layout": {"margin": {"l": 10, "t": 50}}},
                        ),
                    ],
                    style={"width": "48%", "display": "inline-block", "float": "right"},
                ),
            ]
        ),
    ]
)


@dash_app1.callback(
    Output("trend_analysis", "figure"),
    [Input("kategori", "value"), Input("years", "value"), Input("lokasi", "value")],
)
def trend(kategori, years, lokasi):
    trend_data = data_timeseries[
        data_timeseries["kategori"].isin(kategori)
        & data_timeseries["tempat_kejadian"].isin(lokasi)
        & data_timeseries["Tahun"].between(years[0], years[1])
    ]
    decomposition = seasonal_decompose(
        trend_data["Frekuensi"], model="additive", period=12
    )
    trend = decomposition.trend
    trace1 = {
        "line": {"color": "rgb(178,34,34)", "width": 3},
        "mode": "lines",
        "name": "Observed",
        "type": "scatter",
        "x": trend_data.index,
        "y": trend,
    }

    return {
        "data": [trace1],
        "layout": {
            "title": "Trend Analysis " + ", ".join(kategori),
            "xaxis": {
                "type": "date",
                "title": "Time",
                "autorange": True,
            },
            "yaxis": {
                "type": "linear",
                "title": "Jumlah Kasus",
                "autorange": True,
            },
            "autosize": True,
        },
    }


@dash_app1.callback(
    Output("seaonality_analysis", "figure"),
    [Input("kategori", "value"), Input("years", "value"), Input("lokasi", "value")],
)
def seasonality(kategori, years, lokasi):
    season_data = data_timeseries[
        data_timeseries["kategori"].isin(kategori)
        & data_timeseries["tempat_kejadian"].isin(lokasi)
        & data_timeseries["Tahun"].between(years[0], years[1])
    ]
    decomposition = seasonal_decompose(
        season_data["Frekuensi"], model="additive", period=12
    )
    seasonality = decomposition.seasonal
    trace1 = {
        "line": {"color": "rgb(255,140,0)", "width": 3},
        "mode": "lines",
        "name": "Observed",
        "type": "scatter",
        "x": season_data.index,
        "y": seasonality,
    }

    return {
        "data": [trace1],
        "layout": {
            "title": "Seasonality Analysis " + ", ".join(kategori),
            "xaxis": {
                "type": "date",
                "title": "Time",
                "autorange": True,
            },
            "yaxis": {
                "type": "linear",
                "title": "Jumlah Kasus",
                "autorange": True,
            },
            "autosize": True,
        },
    }


@dash_app1.callback(
    Output("by_year_country_world", "figure"),
    [Input("kategori", "value"), Input("years", "value"), Input("lokasi", "value")],
)
def annual_by_country_barchart(kategori, years, lokasi):
    data_map = df[
        df["kategori"].isin(kategori)
        & df["tempat_kejadian"].isin(lokasi)
        & df["Tahun"].between(years[0], years[1])
    ]
    data_map = data_map.groupby(["tanggal", "kategori"], as_index=False)[
        "Frekuensi"
    ].count()

    return {
        "data": [
            go.Scatter(
                x=data_map[data_map["kategori"] == c]["tanggal"],
                y=data_map[data_map["kategori"] == c]["Frekuensi"],
                name=c,
            )
            for c in kategori
        ],
        "layout": go.Layout(
            title="Observed Data "
            + ", ".join(kategori)
            + "  "
            + " - ".join([str(y) for y in years]),
            plot_bgcolor="#eeeeee",
            paper_bgcolor="#eeeeee",
            font={"family": "Roboto"},
            xaxis=dict(
                rangeselector=dict(
                    buttons=list(
                        [
                            dict(
                                count=1, label="1m", step="month", stepmode="backward"
                            ),
                            dict(
                                count=6, label="6m", step="month", stepmode="backward"
                            ),
                            dict(count=1, label="YTD", step="year", stepmode="todate"),
                            dict(count=1, label="1y", step="year", stepmode="backward"),
                            dict(step="all"),
                        ]
                    )
                ),
                rangeslider=dict(visible=True),
                type="date",
            ),
        ),
    }
