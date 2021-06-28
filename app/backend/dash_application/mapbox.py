import os
import pandas as pd
import numpy as np
import dash  # (version 1.0.0)
import dash_table
from .data import create_dataframe
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import plotly.offline as py  # (version 4.4.1)
import plotly.graph_objs as go
import plotly.express as px

from app import dash_app3


mapbox_access_token = os.environ.get("MAPBOX_ACCESS_KEY")


# Load DataFrame
df = create_dataframe()
candidates = df.sort_values("kategori")["kategori"].unique()

mapbox = html.Div(
    [
        html.P("Candidate:"),
        dcc.RadioItems(
            id="candidate",
            options=[{"value": x, "label": x} for x in candidates],
            value=candidates[0],
            labelStyle={"display": "inline-block", "margin-left": "1rem"},
        ),
        dcc.Graph(id="choropleth"),
    ]
)


@dash_app3.callback(Output("choropleth", "figure"), [Input("candidate", "value")])
def display_choropleth(candidate):
    fig = px.density_mapbox(
        df,
        lat="longitude",
        lon="latitude",
        hover_name="kategori",
        hover_data=["tanggal", "jam"],
        zoom=10,
        height=600,
    )
    fig.update_layout(
        mapbox_style="white-bg",
        mapbox_layers=[
            {
                "below": "traces",
                "sourcetype": "raster",
                "sourceattribution": "Sea Crime 2014 - 2019",
                "source": [
                    "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"
                ],
            },
            {
                "sourcetype": "raster",
                "sourceattribution": "Indonesia Spatial Geographic Visualization by Kota 103",
                "source": [
                    "https://geo.weather.gc.ca/geomet/?"
                    "SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&BBOX={bbox-epsg-3857}&CRS=EPSG:3857"
                    "&WIDTH=1000&HEIGHT=1000&LAYERS=RADAR_1KM_RDBR&TILED=true&FORMAT=image/png"
                ],
            },
        ],
    )
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

    return fig
