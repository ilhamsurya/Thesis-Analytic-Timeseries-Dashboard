import dash
import dash_html_components as html
import flask

server = flask.Flask(__name__)

dash_app = dash.Dash(
    __name__,
    server=server,
    url_base_pathname="/forecasting/",
    requests_pathname_prefix="/forecasting/",
)

app.layout = html.Div("Forecasting")
