import os
from flask import Flask
import dash
import dash_html_components as html
import dash_core_components as dcc

app = Flask(__name__)

dash_app1 = dash.Dash(
    __name__,
    server=app,
    url_base_pathname="/timeseries/",
    external_stylesheets=[
        "/static/dist/css/styles.css",
        "https://fonts.googleapis.com/css?family=Lato",
    ],
)

from app.backend.dash_application.timeseries import timeseries
from app.backend.dash_application.timeseries import timeseries_layout

dash_app1.index_string = timeseries_layout
dash_app1.layout = html.Div(
    children=[
        html.Div([html.H1("I am built using Dash!")]),
        timeseries,  # this is the component we imported.
    ]
)
# Import routes here
from app.routes import *

# importing model here
from app.backend.model import *

# app = Flask(__name__, instance_relative_config=False)
# app = init_dashboard(app)
