import os
import random
import textwrap
import datetime as dt
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
candidates = [dict(label=t, value=t) for t in df["kategori"].unique()]


mapbox = html.Div(
    [
        html.Div(
            [
                dcc.RangeSlider(
                    id="years",
                    min=2014,
                    max=2019,
                    dots=True,
                    value=[2014, 2019],
                    marks={str(yr): "'" + str(yr)[2:] for yr in range(2014, 2019)},
                ),
                html.Br(),
                html.Br(),
            ],
            style={"width": "75%", "margin-left": "12%", "background-color": "#eeeeee"},
        ),
        html.Div(
            [
                dcc.Dropdown(
                    id="kategori",
                    multi=True,
                    value=[""],
                    placeholder="Select kategori",
                    options=[
                        {"label": c, "value": c}
                        for c in sorted(df["kategori"].unique().astype(str))
                    ],
                )
            ],
            style={"width": "50%", "margin-left": "25%", "background-color": "#eeeeee"},
        ),
        dcc.Graph(id="map_world", config={"displayModeBar": False}),
        # dcc.Graph(id="choropleth"),
        # dcc.Dropdown(
        #     id="select_type",
        #     options=candidates,
        #     multi=True,
        #     value="Total",
        #     placeholder="Pilih Kategori",
        #     style={
        #         "margin-bottom": "2rem",
        #         "margin-top": "2rem",
        #     },
        # ),
        dcc.Graph(id="by_year_country_world", config={"displayModeBar": False}),
        html.Hr(),
        html.Br(),
        html.Content(
            "Statistik Pelanggaran Laut", style={"margin-left": "25%", "font-size": 25}
        ),
        html.Br(),
        html.Br(),
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                dcc.RangeSlider(
                                    id="years_attacks",
                                    min=2014,
                                    max=2019,
                                    dots=True,
                                    value=[2014, 2019],
                                    marks={
                                        str(yr): str(yr) for yr in range(2014, 2019, 5)
                                    },
                                ),
                                html.Br(),
                            ],
                            style={"margin-left": "5%", "margin-right": "5%"},
                        ),
                        dcc.Graph(
                            id="top_countries_attacks",
                            figure={"layout": {"margin": {"r": 10, "t": 50}}},
                            config={"displayModeBar": False},
                        ),
                    ],
                    style={"width": "48%", "display": "inline-block"},
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                dcc.RangeSlider(
                                    id="years_deaths",
                                    min=2014,
                                    max=2019,
                                    dots=True,
                                    value=[2014, 2019],
                                    marks={
                                        str(yr): str(yr) for yr in range(2014, 2019, 5)
                                    },
                                ),
                                html.Br(),
                            ],
                            style={"margin-left": "5%", "margin-right": "5%"},
                        ),
                        dcc.Graph(
                            id="top_countries_deaths",
                            config={"displayModeBar": False},
                            figure={"layout": {"margin": {"l": 10, "t": 50}}},
                        ),
                    ],
                    style={"width": "48%", "display": "inline-block", "float": "right"},
                ),
            ]
        ),
    ]
)


@dash_app3.callback(
    Output("by_year_country_world", "figure"),
    [Input("kategori", "value"), Input("years", "value")],
)
def annual_by_country_barchart(kategori, years):
    data_map = df[
        df["kategori"].isin(kategori) & df["Tahun"].between(years[0], years[1])
    ]
    data_map = data_map.groupby(["Tahun", "kategori"], as_index=False)[
        "tanggal"
    ].count()

    return {
        "data": [
            go.Bar(
                x=data_map[data_map["kategori"] == c]["Tahun"],
                y=data_map[data_map["kategori"] == c]["tanggal"],
                name=c,
            )
            for c in kategori
        ],
        "layout": go.Layout(
            title="Kasus Pelanggaran Tahunan "
            + ", ".join(kategori)
            + "  "
            + " - ".join([str(y) for y in years]),
            plot_bgcolor="#eeeeee",
            paper_bgcolor="#eeeeee",
            font={"family": "Roboto"},
        ),
    }


@dash_app3.callback(
    Output("map_world", "figure"),
    [Input("kategori", "value"), Input("years", "value")],
)
def countries_on_map(kategori, years):
    data_map = df[
        df["kategori"].isin(kategori) & df["Tahun"].between(years[0], years[1])
    ]
    fig = go.Figure(
        data=[
            go.Scattergeo(
                lon=[
                    x + random.gauss(0.04, 0.03)
                    for x in data_map[data_map["kategori"] == c]["latitude"]
                ],
                lat=[
                    x + random.gauss(0.04, 0.03)
                    for x in data_map[data_map["kategori"] == c]["longitude"]
                ],
                name=c,
                hoverinfo="text",
                marker={
                    "size": 9,
                    "opacity": 0.65,
                    "line": {"width": 0.2, "color": "#cccccc"},
                },
                hovertext=data_map[data_map["kategori"] == c]["tanggal"].astype(str)
                + ", "
                + data_map[data_map["kategori"] == c]["kategori"].astype(str)
                + "<br>"
                + [
                    dt.datetime.strftime(d, "%d %b, %Y")
                    for d in data_map[data_map["kategori"] == c]["tanggal"]
                ]
                + "<br>"
                + "Ditangkap Di:"
                + data_map[data_map["kategori"] == c]["tempat_kejadian"].astype(str)
                + "<br>"
                + "Nama Kapal:"
                + data_map[data_map["kategori"] == c]["nama kapal"].astype(str)
                + "<br>"
                + "Asal Negara: "
                + data_map[data_map["kategori"] == c]["bendera kapal"].astype(str)
                + "<br>"
                + "Penangkap: "
                + data_map[data_map["kategori"] == c]["pemeriksa"].astype(str)
                + "<br>"
                + "Kapal Penangkap: "
                + data_map[data_map["kategori"] == c]["kapal penangkap"].astype(str)
                + "<br><br>"
                + [
                    "<br>".join(textwrap.wrap(x, 40))
                    if not isinstance(x, float)
                    else ""
                    for x in data_map[data_map["kategori"] == c][
                        "sumber_klasifikasi_berita"
                    ]
                ],
            )
            for c in kategori
        ],
        layout=go.Layout(
            title="Terrorist Attacks "
            + ", ".join(kategori)
            + "  "
            + " - ".join([str(y) for y in years]),
            font={"family": "Palatino"},
            titlefont={"size": 22},
        ),
    )

    # fig = px.density_mapbox(
    #     df,
    #     lat="longitude",
    #     lon="latitude",
    #     hover_name="kategori",
    #     hover_data=["tanggal", "jam", "tempat_kejadian"],
    #     zoom=10,
    #     height=600,
    # )
    fig.update_layout(mapbox_style="stamen-terrain", mapbox_center_lon=180)
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

    return fig


# @dash_app3.callback(Output("choropleth", "figure"), [Input("kategori", "value")])
# def display_choropleth(kategori):
#     fig = px.density_mapbox(
#         df,
#         lat="longitude",
#         lon="latitude",
#         hover_name="kategori",
#         hover_data=["tanggal", "jam", "tempat_kejadian"],
#         zoom=10,
#         height=600,
#     )
#     fig.update_layout(
#         mapbox_style="white-bg",
#         mapbox_layers=[
#             {
#                 "below": "traces",
#                 "sourcetype": "raster",
#                 "sourceattribution": "Sea Crime 2014 - 2019",
#                 "source": [
#                     "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"
#                 ],
#             },
#             {
#                 "sourcetype": "raster",
#                 "sourceattribution": "Indonesia Spatial Geographic Visualization by Kota 103",
#                 "source": [
#                     "https://geo.weather.gc.ca/geomet/?"
#                     "SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&BBOX={bbox-epsg-3857}&CRS=EPSG:3857"
#                     "&WIDTH=1000&HEIGHT=1000&LAYERS=RADAR_1KM_RDBR&TILED=true&FORMAT=image/png"
#                 ],
#             },
#         ],
#     )
#     fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

#     return fig


@dash_app3.callback(
    Output("top_countries_attacks", "figure"), [Input("years_attacks", "value")]
)
def top_countries_count(years):
    df_top_countries = df[df["Tahun"].between(years[0], years[1])]
    df_top_countries = df_top_countries.groupby(["kategori"], as_index=False)[
        "tempat_kejadian"
    ].agg(["count"])
    data_kejadian = df_top_countries.sort_values(["count"]).tail(20)
    return {
        "data": [
            go.Bar(
                x=data_kejadian["count"],
                y=data_kejadian.index,
                orientation="h",
                constraintext="none",
                text=df_top_countries.sort_values(["count"]).tail(20).index,
                textposition="outside",
            )
        ],
        "layout": go.Layout(
            title="Akumulasi Jenis Pelanggaran "
            + "  "
            + " - ".join([str(y) for y in years]),
            plot_bgcolor="#eeeeee",
            paper_bgcolor="#eeeeee",
            font={"family": "Palatino"},
            height=700,
            yaxis={"visible": False},
        ),
    }


@dash_app3.callback(
    Output("top_countries_deaths", "figure"), [Input("years_deaths", "value")]
)
def top_countries_deaths(years):
    df_top_countries = df[df["Tahun"].between(years[0], years[1])]
    df_top_countries = df_top_countries.groupby(["tempat_kejadian"], as_index=False)[
        "kategori"
    ].agg(["count"])

    return {
        "data": [
            go.Bar(
                x=df_top_countries.sort_values(["count"]).tail(20)["count"],
                y=df_top_countries.sort_values(["count"]).tail(20).index,
                orientation="h",
                constraintext="none",
                showlegend=False,
                text=df_top_countries.sort_values(["count"]).tail(20).index,
                textposition="outside",
            )
        ],
        "layout": go.Layout(
            title="Lokasi Tempat Pelanggaran "
            + "  "
            + " - ".join([str(y) for y in years]),
            plot_bgcolor="#eeeeee",
            font={"family": "Palatino"},
            paper_bgcolor="#eeeeee",
            height=700,
            yaxis={"visible": False},
        ),
    }
