import streamlit as st
from app.pages import dashboard, exploration, prediction, rapport
from app.data_loader import load_data
from app.model_loader import load_model

# Titre principal
st.set_page_config(page_title="PrediStore", layout="wide")
st.title("📈 PREDI STORE - Prédiction des ventes")

# Chargement des données et du modèle
df = load_data()
model = load_model()

# Menu de navigation
menu = ["📊 Dashboard", "🔍 Exploration", "🔮 Prédiction", "📝 Rapport"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "📊 Dashboard":
    dashboard.show(df)
elif choice == "🔍 Exploration":
    exploration.show(df)
elif choice == "🔮 Prédiction":
    prediction.show(df, model)
elif choice == "📝 Rapport":
    rapport.show()