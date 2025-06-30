import joblib
import streamlit as st

@st.cache_resource
def load_model():
    return joblib.load("random_forest_model.pkl")
