import dash
from flask import config
import plotly.graph_objects as go
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Output, Input, State
import dash_bootstrap_components as dbc
from .data import create_dataframe
from .layout import forecasting_layout
from app import dash_app2
import math as mt
import itertools
import numpy as np
import plotly.express as px
from dash_table import DataTable, FormatTemplate
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARMA

# import pmdarima as pm
import math
import pandas as pd

# <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.6.0/dist/css/bootstrap.min.css" integrity="sha384-B0vP5xmATw1+K9KRQjQERJvTumQW0nPEzvF6L/Z6nronJ3oUOFUFpCjEUQouq2+l" crossorigin="anonymous">

# kebutuhan ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from matplotlib.pylab import rcParams
from sklearn.metrics import mean_squared_error

rcParams["figure.figsize"] = 10, 6


df = pd.read_csv("dataset/gabungan.csv", encoding="unicode_escape")
dff = df.groupby(["kategori", "Tahun", "Bulan"], as_index=False)["Frekuensi"].count()


def differencing(series):
    temp = []
    for x in range(1, len(series)):
        temp.append((series.iloc[x] - series.iloc[x - 1]))
    temp_df = pd.DataFrame(temp, columns=["Frekuensi"])
    return temp_df


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# identifikasi
def Stasionarity_test(series):
    results = adfuller(series)
    return results


def hitungMAPE(actuall, forecast):
    # MAPE
    temp = 0
    for i in range(1, len(forecast) + 1):
        temp = temp + (
            abs(actuall.iloc[i - len(forecast) + 1] - forecast.iloc[i - 1])
            / actuall.iloc[i - len(forecast) + 1]
        )
    print("MAPE: %.2f" % ((temp / len(forecast)) * 100), "%")
    return (temp / len(forecast)) * 100


def hitungMSE(actuall, forecast):
    temp = 0
    # MSE
    for i in range(1, len(forecast) + 1):
        temp = temp + (actuall.iloc[i - len(forecast) + 1] - forecast.iloc[i - 1]) ** 2
    print("MSE : %.2f" % (temp / len(forecast)))
    return temp / len(forecast)


def hitungRMSE(actuall, forecast):
    temp = hitungMSE(actuall, forecast)
    # RMSE
    print("RMSE: %.2f" % (math.sqrt(temp)))
    return math.sqrt(temp)


def hitungMAE(actuall, forecast):
    temp = 0
    # MAE (Mean absoluter error)
    for i in range(1, len(forecast) + 1):
        temp = temp + (abs(actuall.iloc[i - len(forecast) + 1] - forecast.iloc[i - 1]))
    return temp / len(forecast)


def MODEL(model):
    fore = model.get_prediction(start=-10)
    forecast = fore.predicted_mean
    return forecast


def production(model):
    # production
    ramal = model.get_forecast(steps=5)
    ramal_akhir = ramal.predicted_mean
    return ramal_akhir


# estimasi(jml["Frekuensi"])
card_content_1 = [
    dbc.CardHeader("MAPE"),
    dbc.CardBody(
        [
            html.H2(id="MAPE", className="card-title"),
            html.P(
                "10-20 Good forecasting",
                className="card-text",
            ),
        ]
    ),
]

card_content_2 = [
    dbc.CardHeader("RMSE"),
    dbc.CardBody(
        [
            html.H2(id="RMSE", className="card-title"),
            html.P("Lower values are better", className="card-text"),
        ]
    ),
]
card_content_3 = [
    dbc.CardHeader("MAE"),
    dbc.CardBody(
        [
            html.H2(id="MAE", className="card-title"),
            html.P("Lower values are better", className="card-text"),
        ]
    ),
]
card_content_4 = [
    dbc.CardHeader("MSE"),
    dbc.CardBody(
        [
            html.H2(id="MSE", className="card-title"),
            html.P("Lower values are better", className="card-text"),
        ]
    ),
]

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
                                {"label": "Piracy", "value": "Piracy"},
                                {"label": "IUU Fishing", "value": "IUU Fishing"},
                                {
                                    "label": "Perdagangan Manusia",
                                    "value": "Perdagangan Manusia",
                                },
                                {"label": "Imigran Ilegal", "value": "Imigran Ilegal"},
                                {
                                    "label": "Survei Hidros Ilegal",
                                    "value": "Survei Hidros Ilegal",
                                },
                                # for i in jml.sort_values("kategori")["kategori"].unique()
                            ],
                            clearable=True,
                            className="form-dropdown",
                            placeholder="Pilih kategori pelanggaran kedua",
                        ),
                    ],
                    style={"width": "49%", "padding-bottom": "50px"},
                ),
                html.Div(
                    [
                        html.H6(
                            """Grafik Forecasting""",
                            style={"margin-right": "8em"},
                        ),
                        dcc.Graph(
                            id="crossfilter-indicator-scatter",
                            hoverData={"points": [{"customdata": "Laut Halmahera"}]},
                        ),
                    ],
                    style={
                        "width": "100%",
                        "display": "inline-block",
                        "padding": "0 80",
                    },
                ),
                dbc.Row(
                    [
                        dbc.Col(dbc.Card(card_content_1, color="danger", inverse=True)),
                        dbc.Col(
                            dbc.Card(card_content_2, color="secondary", inverse=True)
                        ),
                        dbc.Col(dbc.Card(card_content_3, color="info", inverse=True)),
                        dbc.Col(
                            dbc.Card(card_content_4, color="warning", inverse=True)
                        ),
                    ],
                    style={
                        "width": "100%",
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


@dash_app2.callback(
    Output("MAPE", "children"),
    Output("RMSE", "children"),
    Output("MSE", "children"),
    Output("MAE", "children"),
    Output("crossfilter-indicator-scatter", "figure"),
    [
        Input("crossfilter-kategori-2", "value"),
    ],
)
def build_graph(kategori):
    jml = dff[(dff["kategori"] == kategori)]
    # jml.head()
    waktu = []
    # #proses penambahan variabel series
    for i in range(len(jml["Frekuensi"])):
        waktu.append(i + 1)
    jml["series"] = waktu.copy()
    # #proses ploting
    jml.head()

    fix = []
    for x in range(len(jml)):
        fix.append([jml["series"].iloc[x], jml["Frekuensi"].iloc[x]])

    df = pd.DataFrame(fix, columns=["series", "Frekuensi"])
    df = df.set_index("series")
    # df.head()

    # pengukuran nilai p data asli
    # adf_test_asli = Stasionarity_test(df)
    # if(adf_test_asli[1] > 0.05):
    #     df= differencing(df)

    # slicing data to train and test data
    train_flag = mt.floor(len(df) * 0.8)
    train = df[0:train_flag]
    test = df[train_flag:]

    # penentuan nilai pdq
    p = d = q = range(0, 5)
    pdq = list(itertools.product(p, d, q))
    # order_aic_bic=[]
    # for param in pdq:
    #     try:
    #         model_arima = SARIMAX(train,order=(param))
    #         model_arima_fit = model_arima.fit()
    #         order_aic_bic.append((param,model_arima_fit.aic))
    #     except:
    #         continue

    # #sorting berdasarkan nilai aic terkecil
    # order_df=pd.DataFrame(order_aic_bic, columns=['param','aic'])
    # sem = order_df.sort_values('aic').iloc[0]

    mape = []
    # hitungMAPE(test,prediction)
    # buat model ARIMA
    for y in pdq:
        try:
            model_arima = SARIMAX(test, order=y)
            model_arima_fit = model_arima.fit()
            # prediction
            prediction = model_arima_fit.forecast(steps=len(test))
            mape.append((y, mean_absolute_percentage_error(test, prediction)))
        except:
            continue

    terpilih = pd.DataFrame(mape, columns=["param", "mape"])
    fixpdq = terpilih.sort_values("mape").iloc[0]

    # #model terpilih
    model_arima = SARIMAX(test, order=(fixpdq["param"]))
    model_arima_fit = model_arima.fit()
    # prediction
    prediction = model_arima_fit.forecast(steps=len(test))

    # Create traces
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=train.index, y=train["Frekuensi"], mode="lines", name="Training")
    )
    fig.add_trace(
        go.Scatter(x=test.index, y=test["Frekuensi"], mode="lines", name="Testing")
    )
    fig.add_trace(
        go.Scatter(x=test.index, y=prediction, mode="lines", name="Forecasting")
    )

    fig.update_layout(
        yaxis={"title": "Frekuensi"},
        xaxis={"title": "Bulan"},
        title={
            "text": "Grafik Forecast",
            "font": {"size": 28},
            "x": 0.5,
            "xanchor": "center",
        },
    )

    # plt.plot(test)
    # plt.plot(test.index, prediction)
    rmse = "{:.2f}".format(hitungRMSE(test, prediction))
    mse = "{:.2f}".format(mean_squared_error(test, prediction))
    # mae = "{:.2f}".format(hitungMAE(test, prediction))
    MAEEE = hitungMAE(test, prediction)
    MAPEEE = "{:.2f}".format(mean_absolute_percentage_error(test, prediction))
    # MAPEEE =
    return MAPEEE + "%", rmse, mse, MAEEE, fig
