import dash
from flask import config
from mysql.connector import Connect
import plotly.graph_objects as go
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Output, Input
import dash_bootstrap_components as dbc
from app import dash_app2
import math as mt
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
import pandas as pd

from app.backend.database.conn import connect
# import pmdarima as pm
# kebutuhan ARIMA
from statsmodels.tsa.stattools import adfuller
from matplotlib.pylab import rcParams
from sklearn.metrics import mean_squared_error

# <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.6.0/dist/css/bootstrap.min.css" integrity="sha384-B0vP5xmATw1+K9KRQjQERJvTumQW0nPEzvF6L/Z6nronJ3oUOFUFpCjEUQouq2+l" crossorigin="anonymous">
 
 #database
 


rcParams["figure.figsize"] = 10, 6
df = pd.read_csv("dataset/gabungan.csv", encoding="unicode_escape")
dff = df.groupby(["kategori", "Tahun", "Bulan"], as_index=False)["Frekuensi"].count()


# one-step sarima forecast
def sarima_forecast(history, config):
    order, sorder, trend = config
    # define model
    model = SARIMAX(
        history,
        order=order,
        seasonal_order=sorder,
        trend=trend,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    # fit model
    model_fit = model.fit(disp=False)
    # make one step forecast
    yhat = model_fit.predict(len(history), len(history))
    return yhat[0]


# root mean squared error or rmse
def measure_mse(actual, predicted):
    return mean_squared_error(actual, predicted)


# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
    return data[:-n_test], data[-n_test:]


# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg):
    predictions = list()
    # split dataset
    train, test = train_test_split(data, n_test)
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set
    for i in range(len(test)):
        # fit model and make forecast for history
        yhat = sarima_forecast(history, cfg)
        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        history.append(test[i])
    # estimate prediction error
    error = measure_mse(test, predictions)
    return error


# score a model, return None on failure
def score_model(data, n_test, cfg, debug=False):
    result = None
    # convert config to a key
    key = str(cfg)
    # show all warnings and fail on exception if debugging
    if debug:
        result = walk_forward_validation(data, n_test, cfg)
    else:
        # one failure during model validation suggests an unstable config
        try:
            # never show warnings when grid searching, too noisy
            with catch_warnings():
                filterwarnings("ignore")
                result = walk_forward_validation(data, n_test, cfg)
        except:
            error = None
    # check for an interesting result
    if result is not None:
        print(" > Model[%s] %.3f" % (key, result))
    return (key, result)


# grid search configs
def grid_search(data, cfg_list, n_test, parallel=True):
    scores = None
    if parallel:
        # execute configs in parallel
        executor = Parallel(n_jobs=cpu_count(), backend="multiprocessing")
        tasks = (delayed(score_model)(data, n_test, cfg) for cfg in cfg_list)
        scores = executor(tasks)
    else:
        scores = [score_model(data, n_test, cfg) for cfg in cfg_list]
    #! remove empty results
    scores = [r for r in scores if r[1] != None]
    # sort configs by error, asc
    scores.sort(key=lambda tup: tup[1])
    return scores


# create a set of sarima configs to try
def sarima_configs(seasonal=[0]):
    models = list()
    # define config lists
    p_params = [0, 1, 2]
    d_params = [0, 1]
    q_params = [0, 1, 2]
    t_params = ["n", "c", "t", "ct"]
    P_params = [0, 1, 2]
    D_params = [0, 1]
    Q_params = [0, 1, 2]
    m_params = seasonal
    # create config instances
    for p in p_params:
        for d in d_params:
            for q in q_params:
                for t in t_params:
                    for P in P_params:
                        for D in D_params:
                            for Q in Q_params:
                                for m in m_params:
                                    if (p != 0) and (d != 0) and (q != 0):
                                        cfg = [(p, d, q), (P, D, Q, m), t]
                                        models.append(cfg)
    return models


# parse dates
def custom_parser(x):
    return datetime.strptime("195" + x, "%Y-%m")


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
    dbc.CardHeader("MSE"),
    dbc.CardBody(
        [
            html.H2(id="MSE", className="card-title"),
            html.P(
                # "This is some card content that we'll reuse",
                className="card-text",
            ),
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
    dbc.CardHeader("RMSE"),
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
                            """Kategori Pelanggaran""",
                            style={"margin-right": "2em"},
                        ),
                        dcc.Dropdown(
                            id="crossfilter-kategori-2",
                            # options=[
                            #     {"label": "Piracy", "value": "Piracy"},
                            #     {"label": "IUU Fishing", "value": "IUU Fishing"},
                            #     {"label": "Perdagangan Manusia", "value": "Perdagangan Manusia"},
                            #     {"label": "Imigran Ilegal", "value": "Imigran Ilegal"},
                            #     {"label": "Survei Hidros Ilegal", "value": "Survei Hidros Ilegal"},
                            #     # for i in jml.sort_values("kategori")["kategori"].unique()
                            # ],
                            options=[
                                {"label": c, "value": c}
                                for c in sorted(df["kategori"].unique().astype(str))
                            ],
                            clearable=True,
                            className="form-dropdown",
                            placeholder="Pilih kategori pelanggaran",
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
                        dbc.Col(dbc.Card(card_content_1, color="info", inverse=True)),
                        dbc.Col(
                            dbc.Card(card_content_2, color="secondary", inverse=True)
                        ),
                        # dbc.Col(dbc.Card(card_content_3, color="info", inverse=True)),
                        # dbc.Col(dbc.Card(card_content_4, color="warning", inverse=True)),
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
    Output("MSE", "children"),
    Output("crossfilter-indicator-scatter", "figure"),
    [
        Input("crossfilter-kategori-2", "value"),
    ],
)
def ARIMA_model(kategori):
    data_frame = dff[(dff["kategori"] == kategori)]
    waktu = []
    for i in range(len(data_frame["Frekuensi"])):
        waktu.append(i + 1)

    data_frame["series"] = waktu.copy()
    fix = []
    for x in range(len(data_frame)):
        fix.append([data_frame["series"].iloc[x], data_frame["Frekuensi"].iloc[x]])

    akhir = pd.DataFrame(fix, columns=[ "series","Frekuensi"])
    akhir = akhir.set_index('series')
	
	# data = akhir.values
    # data = akhir.values
    # # data split
    # n_test = mt.floor(len(data)*0.8)
    # cfg_list = sarima_configs()
    # grid search
    # scores = grid_search(data, cfg_list, n_test)
    # for cfg, error in scores[:3]:
    # 	print(cfg, error)

    
    koneksi = connect()
    mycursor = koneksi.cursor()
    mycursor.execute("select * from parameter")
    result = mycursor.fetchall()

    print("Yang dicari")
    print(kategori)
    for i in result:
        if i[8] == kategori:
            print("Hasil ngambil dari database")
            print(i[8])
            p = i[1]
            d = i[2]
            q = i[3]
            P = i[4]
            D = i[5]
            Q = i[6]
            S = i[7]

    # print(result[0])
	# split into train and test sets
    X=[]
    X = akhir.values
    size = int(len(X) * 0.66)
    train, test = X[0:size], X[size : len(X)]
    test_index = akhir[size : len(X)]
    train_index = akhir[0:size]
    history = [x for x in train]
    predictions = list()
    # # walk-forward validation
    for t in range(len(test)):
    	if S == 0:
    		model = SARIMAX(history, order=(p,d,q))
    	else:
    		model = SARIMAX(history, order=(p,d,q), seasonal_order=(P,D,Q,S))
		
    	model_fit = model.fit()
    	output = model_fit.forecast()
    	yhat = output[0]
    	predictions.append(yhat)
    	obs = test[t]
    	history.append(obs)
    	# print('predicted=%f, expected=%f' % (yhat, obs))
    
    # Create traces
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=train_index.index,
            y=train_index["Frekuensi"],
            mode="lines",
            name="Training",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=test_index.index, y=test_index["Frekuensi"], mode="lines", name="Testing"
        )
    )
    fig.add_trace(
        go.Scatter(x=test_index.index, y=predictions, mode="lines", name="Forecasting")
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

    # rmse  = "{:.2f}".format(hitungRMSE(test,prediction))
    mse = "{:.2f}".format(measure_mse(test, predictions))
    # MAEEE =hitungMAE(test,prediction)
    MAPEEE = "{:.2f}".format(mean_absolute_percentage_error(test, predictions))
    # MAPEEE =
    print("Nilai test")
    # print(test[0])
    print("Nilai train")
    # print(train[0])
    print("Nilai Predictions")
    print(predictions)
    print("Nilai pdq PDQS")
    # print(scores[0])
    return MAPEEE+"%",mse,fig
