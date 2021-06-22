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
blackbold = {"color": "black", "font-weight": "bold"}

# Load DataFrame
df = create_dataframe()

mapbox = html.Div(
    [
        # ---------------------------------------------------------------
        # Map_legen + Borough_checklist + Recycling_type_checklist + Web_link + Map
        html.Div(
            [
                html.Div(
                    [
                        # # Map-legend
                        # html.Ul(
                        #     [
                        #         html.Li(
                        #             "Compost",
                        #             className="circle",
                        #             style={
                        #                 "background": "#ff00ff",
                        #                 "color": "black",
                        #                 "list-style": "none",
                        #                 "text-indent": "17px",
                        #             },
                        #         ),
                        #         html.Li(
                        #             "Electronics",
                        #             className="circle",
                        #             style={
                        #                 "background": "#0000ff",
                        #                 "color": "white",
                        #                 "list-style": "none",
                        #                 "text-indent": "17px",
                        #                 "white-space": "nowrap",
                        #             },
                        #         ),
                        #         html.Li(
                        #             "Hazardous_waste",
                        #             className="circle",
                        #             style={
                        #                 "background": "#FF0000",
                        #                 "color": "black",
                        #                 "list-style": "none",
                        #                 "text-indent": "17px",
                        #             },
                        #         ),
                        #         html.Li(
                        #             "Plastic_bags",
                        #             className="circle",
                        #             style={
                        #                 "background": "#00ff00",
                        #                 "color": "black",
                        #                 "list-style": "none",
                        #                 "text-indent": "17px",
                        #             },
                        #         ),
                        #         html.Li(
                        #             "Recycling_bins",
                        #             className="circle",
                        #             style={
                        #                 "background": "#824100",
                        #                 "color": "black",
                        #                 "list-style": "none",
                        #                 "text-indent": "17px",
                        #             },
                        #         ),
                        #     ],
                        #     style={
                        #         "border-bottom": "solid 3px",
                        #         "border-color": "#00FC87",
                        #         "padding-top": "6px",
                        #     },
                        # ),
                        # Borough_checklist
                        html.Label(
                            children=["Filter Tempat Kejadian: "], style=blackbold
                        ),
                        dcc.Checklist(
                            id="input_tempat",
                            options=[
                                {
                                    "label": str(b),
                                    "value": b,
                                }
                                for b in sorted(
                                    df["tempat_kejadian"].unique().astype(str)
                                )
                            ],
                            value=[
                                b
                                for b in sorted(
                                    df["tempat_kejadian"].unique().astype(str)
                                )
                            ],
                        ),
                        # Recycling_type_checklist
                        html.Label(children=["Filter Kategori:  "], style=blackbold),
                        dcc.Checklist(
                            id="input_kategori",
                            options=[
                                {"label": str(b), "value": b}
                                for b in sorted(df["kategori"].unique().astype(str))
                            ],
                            value=[
                                b for b in sorted(df["kategori"].unique().astype(str))
                            ],
                        ),
                        # Web_link
                        html.Br(),
                        html.Label(["Peta Pelanggaran Laut:"], style=blackbold),
                        html.Pre(
                            id="web_link",
                            children=[],
                            style={
                                "white-space": "pre-wrap",
                                "word-break": "break-all",
                                "border": "1px solid black",
                                "text-align": "center",
                                "padding": "12px 12px 12px 12px",
                                "color": "blue",
                                "margin-top": "3px",
                            },
                        ),
                    ],
                    className="three columns",
                ),
                # Map
                html.Div(
                    [
                        dcc.Graph(
                            id="graph",
                            config={"displayModeBar": False, "scrollZoom": True},
                            style={
                                "background": "#00FC87",
                                "padding-bottom": "2px",
                                "padding-left": "2px",
                                "height": "100vh",
                            },
                        )
                    ],
                    className="nine columns",
                ),
            ],
            className="row",
        ),
    ],
    className="ten columns offset-by-one",
)

# ---------------------------------------------------------------
# Output of Graph
@dash_app3.callback(
    Output("graph", "figure"),
    [Input("input_tempat", "value"), Input("input_kategori", "value")],
)
def update_figure(tempat_terpilih, kategori_terpilih):
    df_sub = df[
        (df["tempat_kejadian"].isin(tempat_terpilih))
        & (df["kategori"].isin(kategori_terpilih))
    ]

    # Create figure
    locations = [
        go.Scattermapbox(
            lon=df["latitude"],
            lat=df["longitude"],
            mode="markers",
            unselected={"marker": {"opacity": 1}},
            selected={"marker": {"opacity": 0.5, "size": 25}},
            hoverinfo="text",
        )
    ]

    # Return figure
    return {
        "data": locations,
        "layout": go.Layout(
            uirevision="foo",  # preserves state of figure/map after callback activated
            clickmode="event+select",
            hovermode="closest",
            hoverdistance=2,
            title=dict(
                text="Sebaran Pelanggaran Laut", font=dict(size=50, color="green")
            ),
            mapbox=dict(
                accesstoken=mapbox_access_token,
                bearing=25,
                style="light",
                center=dict(lat=40.80105, lon=-73.945155),
                pitch=40,
                zoom=11.5,
            ),
        ),
    }


# ---------------------------------------------------------------
# callback for Web_link
@dash_app3.callback(Output("web_link", "1"), [Input("graph", "clickData")])
def display_click_data(clickData):
    if clickData is None:
        return "Click on any bubble"
    else:
        # print (clickData)
        the_link = clickData["points"][0]["customdata"]
        if the_link is None:
            return "No Website Available"
        else:
            return html.A(the_link, href=the_link, target="_blank")
