"""Prepare data for Plotly Dash."""
import numpy as np
import pandas as pd


def create_dataframe():
    """Create Pandas DataFrame from local CSV."""
    df = pd.read_csv("dataset/gabungan.csv", parse_dates=["tanggal"])
    # df["tanggal"] = df["tanggal"].dt.date
    # num_complaints = df["kategori"].value_counts()
    # to_remove = num_complaints[num_complaints <= 30].index
    # df.replace(to_remove, np.nan, inplace=True)
    return df
