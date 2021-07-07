"""Prepare data for Plotly Dash."""
import numpy as np
import pandas as pd


def create_dataframe():
    """Create Pandas DataFrame from local CSV."""
    df = pd.read_csv("dataset/gabungan.csv")
    df["tanggal"] = pd.to_datetime(df["tanggal"])
    num_complaints = df["kategori"].value_counts()
    to_remove = num_complaints[num_complaints <= 30].index
    df.replace(to_remove, np.nan, inplace=True)
    df["Tahun"] = df["tanggal"].dt.year
    df["Bulan"] = df["tanggal"].dt.month
    df["Hari"] = df["tanggal"].dt.day
    df["tanggal"] = df["tanggal"].dt.strftime("%Y-%m")
    # df["tanggal"] = df["tanggal"].dt.date
    # num_complaints = df["kategori"].value_counts()
    # to_remove = num_complaints[num_complaints <= 30].index
    # df.replace(to_remove, np.nan, inplace=True)
    return df
    
def create_dataframe_time_series():
    """Create Pandas DataFrame from local CSV."""
    dataset = pd.read_csv("dataset/gabungan.csv")
    dataset["tanggal"] = pd.to_datetime(dataset["tanggal"])
    dataset.set_index("tanggal", inplace=True)
    dataset.sort_index(inplace=True)
    # df["tanggal"] = df["tanggal"].dt.date
    # num_complaints = df["kategori"].value_counts()
    # to_remove = num_complaints[num_complaints <= 30].index
    # df.replace(to_remove, np.nan, inplace=True)
    return dataset

def create_dataframe_map():
    """Create Pandas DataFrame from local CSV."""
    dff = pd.read_csv("dataset/gabungan.csv")
    dff["tanggal"] = pd.to_datetime(dff["tanggal"])
    dff["Tahun"] = dff["tanggal"].dt.year
    dff["Bulan"] = dff["tanggal"].dt.month
    dff["Hari"] = dff["tanggal"].dt.day
    # df["tanggal"] = df["tanggal"].dt.date
    # num_complaints = df["kategori"].value_counts()
    # to_remove = num_complaints[num_complaints <= 30].index
    # df.replace(to_remove, np.nan, inplace=True)
    return dff
