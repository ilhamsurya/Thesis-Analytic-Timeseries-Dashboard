import dash
import dash_html_components as html
import dash_html_components as html
import dash_core_components as dcc
from .data import create_dataframe
from .layout import forecasting_layout
from app import dash_app2
import plotly.express as px

# Load DataFrame
df = create_dataframe()

# Create Layout
forecasting = html.Div(
    [
        html.Div(
            [
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
                                2014: {"label": "2014", "style": {"color": "#77b0b1"}},
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
                        "width": "100%",
                        "padding": "80px 80px 80px 80px",
                        "color": "black",
                    },
                ),
                html.Div(
                    [
                        html.H6(
                            """Grafik PForecasting""",
                            style={"margin-right": "8em"},
                        ),
                        dcc.Graph(
                            id="crossfilter-indicator-scatter",
                            hoverData={"points": [{"customdata": "Laut Halmahera"}]},
                        ),
                    ],
                    style={
                        "width": "49%",
                        "display": "inline-block",
                        "padding": "0 80",
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
