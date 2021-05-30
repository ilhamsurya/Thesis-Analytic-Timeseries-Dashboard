import dash
import dash_html_components as html
import dash_html_components as html
import dash_core_components as dcc
from .data import create_dataframe
import plotly.express as px
import plotly.graph_objects as go

# Load DataFrame
df = create_dataframe()
df["text"] = df["kategori"] + ", " + df["tempat_kejadian"]

fig = go.Figure(
    data=go.Scattergeo(
        lon=df["latitude"],
        lat=df["longitude"],
        text=df["text"],
        mode="markers",
    )
)

fig.update_layout(geo_scope="usa")

# Create Layout
mapbox = html.Div(
    children=[
        html.H1(children="Identified Geothermal Systems of the Western USA"),
        html.Div(
            children="""
        This data was provided by the USGS.
    """
        ),
        dcc.Graph(id="example-map", figure=fig),
    ]
)
