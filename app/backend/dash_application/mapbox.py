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


def process_pelanggaran_data(df):

    # Columns renaming
    df.columns = [col.lower() for col in df.columns]

    # Saving countries positions (latitude and longitude per subzones)
    posisi_kejadian = (
        df[["latitude", "longitude"]]
        .drop_duplicates(["tempat_kejadian"])
        .set_index(["tempat_kejadian"])
    )

    # Pivoting per category
    df = pd.pivot_table(
        df, values="count", index=["tanggal", "tempat_kejadian"], columns=["kategori"]
    )
    df.columns = ["Imigran Ilegal", "Pembalakan Liar"]

    # Merging locations after pivoting
    df = df.join(posisi_kejadian)

    # Filling nan values with 0
    df = df.fillna(0)

    # Compute bubble sizes
    df["size"] = (
        df["Imigran Ilegal"]
        .apply(lambda x: (np.sqrt(x / 100) + 1) if x > 500 else (np.log(x) / 2 + 1))
        .replace(np.NINF, 0)
    )

    # Compute bubble color
    df["color"] = (df["recovered"] / df["confirmed"]).fillna(0).replace(np.inf, 0)

    return df


blackbold = {"color": "black", "font-weight": "bold"}

mapbox = html.Div(
    [
        # ---------------------------------------------------------------
        # Map_legen + Borough_checklist + Recycling_type_checklist + Web_link + Map
        html.Div(
            [
                html.Div(
                    [
                        # Web_link
                        html.Br(),
                        html.Label(["Website:"], style=blackbold),
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
