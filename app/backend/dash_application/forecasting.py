import dash
import plotly.graph_objects as go
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Output, Input, State
import dash_bootstrap_components as dbc
from .data import create_dataframe
from .layout import forecasting_layout
from app import dash_app2
import plotly.express as px
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARMA

# import pmdarima as pm
import math
import pandas as pd


# kebutuhan ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from matplotlib.pylab import rcParams

rcParams["figure.figsize"] = 10, 6
from sklearn.metrics import mean_squared_error

df = pd.read_csv("dataset/Piracy.csv", encoding="unicode_escape")
# proses counting
jml = df.groupby(["kategori", "Tahun", "Bulan"], as_index=False)["Frekuensi"].count()
# proses ploting
# Load DataFrame
# df = create_dataframe()
# manual differencing'
d = 0

card_content_1 = [
    dbc.CardHeader("Card header"),
    dbc.CardBody(
        [
            html.H5("Card title", className="card-title"),
            html.P(
                "This is some card content that we'll reuse",
                className="card-text",
            ),
        ]
    ),
]


def differencing(series):
    temp = []
    for x in range(1, len(series)):
        temp.append((series.iloc[x] - series.iloc[x - 1]))
    temp_df = pd.DataFrame(temp, columns=["Frekuensi"])
    #     temp_df.plot()
    #     plt.show()
    #     print(temp_df)
    return temp_df


# identifikasi
def Stasionarity_test(series):
    results = adfuller(series)
    #     print('ADF Statistic: %f' % results[0])
    #     print('p-value: %f'% results[1])
    #     print('Critical Values:')
    return results


# estimasi
def estimasi(series):
    global p
    global q
    order_aic_bic = []
    for p in range(len(series.iloc[-3:])):
        for q in range(len(series.iloc[-3:])):
            try:
                model = SARIMAX(series, order=(p, 0, q))
                results = model.fit()
                order_aic_bic.append((p, q, results.aic, results.bic))
            except:
                print(p, q, None, None)
    order_df = pd.DataFrame(order_aic_bic, columns=["p", "q", "aic", "bic"])
    sem = order_df.sort_values("aic").iloc[0]
    p = sem["p"]
    q = sem["q"]


# Evaluasi
def evaluasi(actuall, forecast):
    # Evaluasi model
    temp = 0
    # MAPE
    for i in range(1, len(forecast) + 1):
        temp = temp + (
            abs(actuall.iloc[i - len(forecast) + 1] - forecast.iloc[i - 1])
            / actuall.iloc[i - len(forecast) + 1]
        )
    print("MAPE: %.2f" % ((temp / len(forecast)) * 100), "%")

    temp = 0
    # MSE
    for x in range(1, len(forecast) + 1):
        temp = temp + (actuall.iloc[i - len(forecast) + 1] - forecast.iloc[i - 1]) ** 2
    print("MSE : %.2f" % (temp / len(forecast)))
    # RMSE
    print("RMSE: %.2f" % (math.sqrt(temp / len(forecast))))
    temp = 0

    # MAE (Mean absoluter error)
    for x in range(1, len(forecast) + 1):
        temp = temp + (abs(actuall.iloc[i - len(forecast) + 1] - forecast.iloc[i - 1]))
    print("MAE : %.2f" % (temp / len(forecast)))


# production
def production(data):
    model = SARIMAX(data, order=(p, 0, q))
    hasil = model.fit()
    return hasil
    # plotting
    # plt.figure()
    # plt.plot( jml['series'],data,color='blue',label='Actual')
    # plt.plot( jml['series'].iloc[-10:],forecast.values,color='red',label='Forecast')
    # ramal_akhir.plot(color="Orange")
    # plt.fill_between(,mean_forecast, confidence_interval['lower Frekuensi'], confidence_interval['upper Frekuensi'], color='pink')
    # plt.show()
    # print("Actuall")
    # print(data)
    # print("Testing")
    # print(forecast)
    # print("Forecast")
    # print(ramal_akhir)
    # evaluasi(data, forecast)


# estimasi(jml["Frekuensi"])


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
                                for i in jml.sort_values("kategori")[
                                    "kategori"
                                ].unique()
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
                            dbc.Card(card_content_1, color="secondary", inverse=True)
                        ),
                        dbc.Col(dbc.Card(card_content_1, color="info", inverse=True)),
                        dbc.Col(
                            dbc.Card(card_content_1, color="warning", inverse=True)
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


@dash_app2.callback(Output("my_bar", "figure"), [Input("user_choice", "value")])
def update_graph(value):
    fig = px.scatter(
        df.query("year=={}".format(str(value))),
        x="gdpPercap",
        y="lifeExp",
        size="pop",
        color="continent",
        title=str(value),
        hover_name="country",
        log_x=True,
        size_max=60,
    ).update_layout(showlegend=True, title_x=0.5)
    return fig


@dash_app2.callback(
    Output("crossfilter-indicator-scatter", "figure"),
    [
        Input("crossfilter-kategori-2", "value"),
    ],
)
def build_graph(kategori):

    # jml.head()
    waktu = []
    # proses penambahan variabel series
    for i in range(len(jml["Frekuensi"])):
        waktu.append(i + 1)
    jml["series"] = waktu
    # dff = df.groupby()
    # identifikasi
    hasil = Stasionarity_test(jml["Frekuensi"])
    print("P-value %f" % hasil[1])
    p_value = "%f" % hasil[1]
    if float(p_value) > 0.05:
        df_jml = differencing(jml["Frekuensi"])
        estimasi(df_jml)
        hasil = production(jml["Frekuensi"])
        fore = hasil.get_prediction(start=-10)
        forecast = fore.predicted_mean
        ramal = hasil.get_forecast(steps=5)
        ramal_akhir = ramal.predicted_mean

    # estimasi
    else:
        estimasi(jml["Frekuensi"])
        hasil = production(jml["Frekuensi"])
        fore = hasil.get_prediction(start=-10)
        forecast = fore.predicted_mean
        ramal = hasil.get_forecast(steps=5)
        ramal_akhir = ramal.predicted_mean

    # Create traces

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=jml["series"], y=jml["Frekuensi"], mode="lines", name="Training")
    )
    fig.add_trace(
        go.Scatter(
            x=jml["series"].iloc[-10:], y=forecast.values, mode="lines", name="Testing"
        )
    )
    fig.add_trace(
        go.Scatter(x=ramal_akhir.index, y=ramal_akhir, mode="lines", name="Forecasting")
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
    return fig
