import pandas as pd
import streamlit as st

@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dayofweek"] = df["date"].dt.dayofweek
    return df
