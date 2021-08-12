import dash
from flask import config
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
<<<<<<< HEAD

# <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.6.0/dist/css/bootstrap.min.css" integrity="sha384-B0vP5xmATw1+K9KRQjQERJvTumQW0nPEzvF6L/Z6nronJ3oUOFUFpCjEUQouq2+l" crossorigin="anonymous">

=======
# import pmdarima as pm
>>>>>>> 2bd337c22d47cbb49f3d0ffa1af99d475c10946a
# kebutuhan ARIMA
from statsmodels.tsa.stattools import adfuller
from matplotlib.pylab import rcParams
from sklearn.metrics import mean_squared_error
# <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.6.0/dist/css/bootstrap.min.css" integrity="sha384-B0vP5xmATw1+K9KRQjQERJvTumQW0nPEzvF6L/Z6nronJ3oUOFUFpCjEUQouq2+l" crossorigin="anonymous">
 


<<<<<<< HEAD
df = pd.read_csv("dataset/gabungan.csv", encoding="unicode_escape")
dff = df.groupby(["kategori", "Tahun", "Bulan"], as_index=False)["Frekuensi"].count()
=======
rcParams["figure.figsize"] = 10, 6
df = pd.read_csv("dataset/gabungan.csv", encoding='unicode_escape')
dff = df.groupby(["kategori","Tahun","Bulan"], as_index=False)["Frekuensi"].count()
>>>>>>> 2bd337c22d47cbb49f3d0ffa1af99d475c10946a



# one-step sarima forecast
def sarima_forecast(history, config):
	order, sorder, trend = config
	# define model
	model = SARIMAX(history, order=order, seasonal_order=sorder, trend=trend, enforce_stationarity=False, enforce_invertibility=False)
	# fit model
	model_fit = model.fit(disp=False)
	# make one step forecast
	yhat = model_fit.predict(len(history), len(history))
	return yhat[0]

# root mean squared error or rmse
def measure_rmse(actual, predicted):
	return mt.sqrt(mean_squared_error(actual, predicted))

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
	error = measure_rmse(test, predictions)
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
		print(' > Model[%s] %.3f' % (key, result))
	return (key, result)

# grid search configs
def grid_search(data, cfg_list, n_test, parallel=True):
	scores = None
	if parallel:
		# execute configs in parallel
		executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
		tasks = (delayed(score_model)(data, n_test, cfg) for cfg in cfg_list)
		scores = executor(tasks)
	else:
		scores = [score_model(data, n_test, cfg) for cfg in cfg_list]
	# remove empty results
	scores = [r for r in scores if r[1] != None ]
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
	t_params = ['n','c','t','ct']
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
									if ((p != 0) and (d != 0) and (q != 0)):  
										cfg = [(p,d,q), (P,D,Q,m), t]
										models.append(cfg)
	return models

# parse dates
def custom_parser(x):
	return datetime.strptime('195'+x, '%Y-%m')

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


<<<<<<< HEAD
=======

>>>>>>> 2bd337c22d47cbb49f3d0ffa1af99d475c10946a
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
<<<<<<< HEAD
    # RMSE
    print("RMSE: %.2f" % (math.sqrt(temp)))
    return math.sqrt(temp)


=======
    #RMSE
    print("RMSE: %.2f"%(mt.sqrt(temp)))
    return (mt.sqrt(temp))
    
>>>>>>> 2bd337c22d47cbb49f3d0ffa1af99d475c10946a
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
<<<<<<< HEAD
            html.H2(id="RMSE", className="card-title"),
            html.P("Lower values are better", className="card-text"),
=======
             html.H2(
                id="MSE",className="card-title"),
            html.P(
                # "This is some card content that we'll reuse",
                className="card-text",
            ),
>>>>>>> 2bd337c22d47cbb49f3d0ffa1af99d475c10946a
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
                            """Kategori Pelanggaran 2""",
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
<<<<<<< HEAD
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
=======
                                {"label": c, "value": c}
                                for c in sorted(df["kategori"].unique().astype(str))
>>>>>>> 2bd337c22d47cbb49f3d0ffa1af99d475c10946a
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
<<<<<<< HEAD
                        dbc.Col(dbc.Card(card_content_1, color="danger", inverse=True)),
                        dbc.Col(
                            dbc.Card(card_content_2, color="secondary", inverse=True)
                        ),
                        dbc.Col(dbc.Card(card_content_3, color="info", inverse=True)),
                        dbc.Col(
                            dbc.Card(card_content_4, color="warning", inverse=True)
                        ),
=======
                        dbc.Col(dbc.Card(card_content_1, color="info", inverse=True)),
                        dbc.Col(dbc.Card(card_content_2, color="secondary", inverse=True)),
                        # dbc.Col(dbc.Card(card_content_3, color="info", inverse=True)),
                        # dbc.Col(dbc.Card(card_content_4, color="warning", inverse=True)),
>>>>>>> 2bd337c22d47cbb49f3d0ffa1af99d475c10946a
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
<<<<<<< HEAD
    Output("MAPE", "children"),
    Output("RMSE", "children"),
    Output("MSE", "children"),
    Output("MAE", "children"),
=======
    Output('MAPE','children'),
    Output('MSE','children'),
>>>>>>> 2bd337c22d47cbb49f3d0ffa1af99d475c10946a
    Output("crossfilter-indicator-scatter", "figure"),
    [
        Input("crossfilter-kategori-2", "value"),
    ],
)
<<<<<<< HEAD
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
=======

def ARIMA_model(kategori):
    data_frame = dff[(dff["kategori"] ==kategori)]
    waktu= []
    for i in range(len(data_frame["Frekuensi"])):
    	waktu.append(i+1) 

    data_frame["series"] = waktu.copy()    
    fix=[]
    for x in range(len(data_frame)):
    	fix.append([data_frame["series"].iloc[x],data_frame["Frekuensi"].iloc[x]])

    akhir = pd.DataFrame(fix, columns=[ "series","Frekuensi"])
    akhir = akhir.set_index('series')
	
	# data = akhir.values
    data = akhir.values
    # # data split
    n_test = mt.floor(len(data)*0.8)
    # # model configs
    cfg_list = sarima_configs()
    # grid search
    scores = grid_search(data, cfg_list, n_test)
    # for cfg, error in scores[:3]:
    # 	print(cfg, error)

    p = int(scores[0][0][2])
    d = int(scores[0][0][5])
    q = int(scores[0][0][8])

    P = int(scores[0][0][13])
    D = int(scores[0][0][16])
    Q = int(scores[0][0][19])
    S = int(scores[0][0][22])

    # p = 1
    # d = 1
    # q = 2

    # P = 0
    # D = 0
    # Q = 0
    # S = 0

	# split into train and test sets
    X=[]
    X = akhir.values
    size = int(len(X) * 0.66)
    train, test = X[0:size], X[size:len(X)]
    test_index = akhir[size:len(X)]
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
    	print('predicted=%f, expected=%f' % (yhat, obs))
	
	# evaluate forecasts
    # mape = (mean_absolute_percentage_error(test, predictions))
    # print('Test MAPE: %.3f' % mape+ '%')
	# # plot forecasts against actual outcomes
    # plt.plot(test)
    # plt.plot(predictions, color='red')
    # plt.show()
    
    # Create traces
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_index.index, y=train_index["Frekuensi"],
                        mode='lines',
                        name='Training'))
    fig.add_trace(go.Scatter(x=test_index.index, y= test_index["Frekuensi"],
                        mode='lines',
                        name='Testing'))
    fig.add_trace(go.Scatter(x=test_index.index, y=predictions,
                        mode='lines', name='Forecasting'))
>>>>>>> 2bd337c22d47cbb49f3d0ffa1af99d475c10946a

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

<<<<<<< HEAD
    # plt.plot(test)
    # plt.plot(test.index, prediction)
    rmse = "{:.2f}".format(hitungRMSE(test, prediction))
    mse = "{:.2f}".format(mean_squared_error(test, prediction))
    # mae = "{:.2f}".format(hitungMAE(test, prediction))
    MAEEE = hitungMAE(test, prediction)
    MAPEEE = "{:.2f}".format(mean_absolute_percentage_error(test, prediction))
    # MAPEEE =
    return MAPEEE + "%", rmse, mse, MAEEE, fig
=======
    # rmse  = "{:.2f}".format(hitungRMSE(test,prediction))
    mse= "{:.2f}".format(measure_rmse(test,predictions))
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
    print(p,d,q,P,D,Q,S)
    return MAPEEE+"%",mse,fig
>>>>>>> 2bd337c22d47cbb49f3d0ffa1af99d475c10946a
