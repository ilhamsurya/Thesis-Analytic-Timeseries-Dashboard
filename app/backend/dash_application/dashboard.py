"""Instantiate a Dash app."""
import numpy as np
import pandas as pd
import dash
import dash_table
import dash_html_components as html
import dash_core_components as dcc
from .data import create_dataframe
from .layout import timeseries_layout


def init_dashboard(server):
    """Create a Plotly Dash dashboard."""
    dash_app = dash.Dash(
        server=server,
        routes_pathname_prefix="/dashapp/",
        external_stylesheets=[
            "/static/dist/css/styles.css",
            "https://fonts.googleapis.com/css?family=Lato",
        ],
    )

    # Load DataFrame
    df = create_dataframe()

    available_indicators = df["kategori"].unique()

    # Custom HTML layout
    dash_app.index_string = html_layout

    # Create Layout
    dash_app.layout = html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            dcc.Dropdown(
                                id="crossfilter-xaxis-column",
                                options=[
                                    {"label": i, "value": i}
                                    for i in available_indicators
                                ],
                                value="Fertility rate, total (births per woman)",
                            ),
                            dcc.RadioItems(
                                id="crossfilter-xaxis-type",
                                options=[
                                    {"label": i, "value": i} for i in ["Linear", "Log"]
                                ],
                                value="Linear",
                                labelStyle={"display": "inline-block"},
                            ),
                        ],
                        style={"width": "49%", "display": "inline-block"},
                    ),
                    html.Div(
                        [
                            dcc.Dropdown(
                                id="crossfilter-yaxis-column",
                                options=[
                                    {"label": i, "value": i}
                                    for i in available_indicators
                                ],
                                value="Life expectancy at birth, total (years)",
                            ),
                            dcc.RadioItems(
                                id="crossfilter-yaxis-type",
                                options=[
                                    {"label": i, "value": i} for i in ["Linear", "Log"]
                                ],
                                value="Linear",
                                labelStyle={"display": "inline-block"},
                            ),
                        ],
                        style={
                            "width": "49%",
                            "float": "right",
                            "display": "inline-block",
                        },
                    ),
                    html.Div(
                        dcc.Slider(
                            id="crossfilter-year--slider",
                            min=df["Tahun"].min(),
                            max=df["Tahun"].max(),
                            value=df["Tahun"].max(),
                            marks={
                                str(tahun): str(tahun) for tahun in df["Tahun"].unique()
                            },
                            step=None,
                        ),
                        style={
                            "width": "100%",
                            "padding": "0px 20px 20px 20px",
                            "color": "black",
                        },
                    ),
                    html.Div(
                        [
                            dcc.Graph(
                                id="crossfilter-indicator-scatter",
                                hoverData={"points": [{"customdata": "Japan"}]},
                            )
                        ],
                        style={
                            "width": "49%",
                            "display": "inline-block",
                            "padding": "0 20",
                        },
                    ),
                    html.Div(
                        [
                            dcc.Graph(id="x-time-series"),
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

    return dash_app.server


#    children=[
#             dcc.Graph(
#                 id="histogram-graph",
#                 figure={
#                     "data": [
#                         {
#                             "x": df["kategori"],
#                             "text": df["kategori"],
#                             "customdata": df["tanggal"],
#                             "name": "311 Calls by region.",
#                             "type": "histogram",
#                         }
#                     ],
#                     "layout": {
#                         "title": "Pelanggaran Laut",
#                         "height": 500,
#                         "padding": 150,
#                     },
#                 },
#             ),
#         ],
#         id="dash-container",
