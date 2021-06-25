import dash
import dash_html_components as html
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Output, Input, State
import dash_bootstrap_components as dbc
from .data import create_dataframe
from .layout import forecasting_layout
from app import dash_app2
from app import dash_app1
import plotly.express as px


#kebutuhan ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 10, 6
from sklearn.metrics import mean_squared_error

# Load DataFrame
df = create_dataframe()


df = df.groupby(["Tahun", "kategori", "tempat_kejadian"], as_index=False)[
    "Frekuensi"
].count()

# Create Layout
forecasting = html.Div(
    [
        html.Div(
            [
                html.Div(
                    [
                        html.H6(
                            """Kategori Pelanggaran 2""",
                            style={"margin-right": "2em"},
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
                    style={"width": "49%", "padding-bottom":"50px"},
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
                dbc.Card(
                dbc.CardBody(
                    [
                    html.H4("Title", className="card-title"),
                    html.H6("Card subtitle", className="card-subtitle"),
                    html.P(
                        "Some quick example text to build on the card title and make "
                        "up the bulk of the card's content.",
                            className="card-text",
                        ),
                        dbc.CardLink("Card link", href="#"),
                        dbc.CardLink("External link", href="https://google.com"),
                    ]
                ),
                style={"width": "18rem"},
                )
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
        Input("crossfilter-kategori-2", "value"),
    ],
)
def build_graph(kategori):
    # dff = df.groupby()
    dff = df[
        ((df["kategori"] == kategori))
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
    # print(dff)
    return fig 
