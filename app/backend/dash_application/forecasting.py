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
import plotly.express as px
from dash_table import DataTable, FormatTemplate
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

rcParams["figure.figsize"] = 10, 6
from sklearn.metrics import mean_squared_error

df = pd.read_csv("dataset/gabungan.csv", encoding="unicode_escape")
jml = df.groupby(["kategori", "Tahun", "Bulan"], as_index=False)["Frekuensi"].count()


def differencing(series):
    temp=[]
    for x in range(1,len(series)):
        temp.append((series.iloc[x] - series.iloc[x-1]))
    temp_df = pd.DataFrame(temp, columns=['Frekuensi'])
    return temp_df

#identifikasi
def Stasionarity_test(series):
    results = adfuller(series)
    return results

def hitungMAPE(actuall, forecast):
    #MAPE
    temp=0
    for i in range(1,len(forecast)+1):
        temp = temp + (abs(actuall.iloc[i-len(forecast)+1] - forecast.iloc[i-1])/actuall.iloc[i-len(forecast)+1])
    print("MAPE: %.2f"%((temp/len(forecast))*100),"%")
    return (temp/len(forecast))*100

def hitungMSE(actuall, forecast):
    temp=0
    #MSE
    for i in range(1,len(forecast)+1):
         temp = temp + (actuall.iloc[i-len(forecast)+1] - forecast.iloc[i-1])**2
    print("MSE : %.2f"%(temp/len(forecast)))
    return (temp/len(forecast))

def hitungRMSE(actuall, forecast):
    temp = hitungMSE(actuall, forecast)
    #RMSE
    print("RMSE: %.2f"%(math.sqrt(temp)))
    return (math.sqrt(temp))
    
def hitungMAE(actuall, forecast):
    temp = 0
    #MAE (Mean absoluter error)
    for i in range(1,len(forecast)+1):
         temp = temp + (abs(actuall.iloc[i-len(forecast)+1] - forecast.iloc[i-1]))
    print("MAE : %.2f"%(temp/len(forecast)))
    return (temp/len(forecast))

def MODEL(model):
    fore=model.get_prediction(start=-10)
    forecast = fore.predicted_mean
    return forecast

def production(model):
    #production
    ramal = model.get_forecast(steps=5)
    ramal_akhir = ramal.predicted_mean
    return ramal_akhir
    

# estimasi(jml["Frekuensi"])
card_content_1 = [
    dbc.CardHeader("MAPE"),
    dbc.CardBody(
        [
            html.H2(
                id="MAPE",className="card-title"),
            html.P(
                "This is some card content that we'll reuse",
                className="card-text",
            ),
        ]
    ),
]

card_content_2 = [
    dbc.CardHeader("RMSE"),
    dbc.CardBody(
        [
             html.H2(
                id="RMSE",className="card-title"),
            html.P(
                "This is some card content that we'll reuse",
                className="card-text",
            ),
        ]
    ),
]
card_content_3 = [
    dbc.CardHeader("MAE"),
    dbc.CardBody(
        [
             html.H2(
                id="MAE",className="card-title"),
            html.P(
                "This is some card content that we'll reuse",
                className="card-text",
            ),
        ]
    ),
]
card_content_4 = [
    dbc.CardHeader("MSE"),
    dbc.CardBody(
        [
             html.H2(
                id="MSE",className="card-title"),
            html.P(
                "This is some card content that we'll reuse",
                className="card-text",
            ),
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
                                {"label": "Perdagangan Manusia", "value": "Perdagangan Manusia"},
                                {"label": "Imigran Ilegal", "value": "Imigran Ilegal"},
                                {"label": "Survei Hidros Ilegal", "value": "Survei Hidros Ilegal"},
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
                        dbc.Col(dbc.Card(card_content_2, color="secondary", inverse=True)),
                        dbc.Col(dbc.Card(card_content_3, color="info", inverse=True)),
                        dbc.Col(dbc.Card(card_content_4, color="warning", inverse=True)),
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
    Output('MAPE','children'),
    Output('RMSE','children'),
    Output('MSE','children'),
    Output('MAE','children'),
    Output("crossfilter-indicator-scatter", "figure"),
    [
        Input("crossfilter-kategori-2", "value"),
    ],
)

def build_graph(kategori):
    dff=jml[(jml["kategori"]==kategori)]
    waktu=[]
    #identifikasi
    #proses penambahan variabel series
    for i in range(len(dff["Frekuensi"])):
        waktu.append(i+1) 
    dff['series'] = waktu

    #identifikasi
    hasil = Stasionarity_test(dff["Frekuensi"])
    print("P-value %f" % hasil[1])
    p_value = "%f"%hasil[1]
    if float(p_value) > 0.05:
        df_jml = differencing(dff["Frekuensi"])
        
        #estimasi
        order_aic_bic=[]
        for p in range(len(df_jml.iloc[-3:])):
            for q in range(len(df_jml.iloc[-3:])):
                try:
                    model = SARIMAX(df_jml, order=(p,0,q))
                    results = model.fit()
                    print((p,q,results.aic, results.bic))
                    order_aic_bic.append((p,q,results.aic, results.bic))
                except:
                    print(p,q,None,None)
        order_df=pd.DataFrame(order_aic_bic, columns=['p','q','aic','bic'])
        sem = order_df.sort_values('aic').iloc[0]
        p = sem['p']
        q = sem['q']
        model = SARIMAX(df_jml, order=(p,0,q))
        hasil = model.fit()
        #production
        hasil_model = MODEL(hasil)
        hasil_forecast = production(hasil)
        
    

    #estimasi
    else:
        order_aic_bic=[]
        for p in range(len(dff["Frekuensi"].iloc[-3:])):
            for q in range(len(dff["Frekuensi"].iloc[-3:])):
                try:
                    model = SARIMAX(dff["Frekuensi"], order=(p,0,q))
                    results = model.fit()
                    print((p,q,results.aic, results.bic))
                    order_aic_bic.append((p,q,results.aic, results.bic))
                except:
                    print(p,q,None,None)
        order_df=pd.DataFrame(order_aic_bic, columns=['p','q','aic','bic'])
        sem = order_df.sort_values('aic').iloc[0]
        p = sem['p']
        q = sem['q']
        model = SARIMAX(dff["Frekuensi"], order=(p,0,q))
        hasil = model.fit()
        
        #production
        hasil_model = MODEL(hasil)
        hasil_forecast = production(hasil)

    #evaluasi
    # %.2f"%(temp/len(forecast))
    mape = "%.2f"%(hitungMAPE(dff["Frekuensi"],hasil_model))
    mse = "%.2f"%hitungMSE(dff["Frekuensi"],hasil_model)
    rmse = "%.2f"%hitungRMSE(dff["Frekuensi"],hasil_model)
    mae = "%.2f"%hitungMAE(dff["Frekuensi"],hasil_model)

    # Create traces
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=waktu, y=dff["Frekuensi"],
                        mode='lines',
                        name='Training'))
    fig.add_trace(go.Scatter(x=dff['series'].iloc[-10:], y=hasil_model.values,
                        mode='lines',
                        name='Testing'))
    fig.add_trace(go.Scatter(x=hasil_forecast.index, y=hasil_forecast,
                        mode='lines', name='Forecasting'))

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
    return mape+"%",rmse,mse,mae,fig

