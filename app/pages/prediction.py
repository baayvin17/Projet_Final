import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def show(df, model):
    st.header("🔮 Prédiction des ventes futures")

    store = st.number_input("Magasin (store)", min_value=int(df["store"].min()), max_value=int(df["store"].max()), value=int(df["store"].min()))
    item = st.number_input("Produit (item)", min_value=int(df["item"].min()), max_value=int(df["item"].max()), value=int(df["item"].min()))
    date_input = st.date_input("Date de la prédiction")

    def prepare_input(date, store, item):
        date = pd.to_datetime(date)
        return pd.DataFrame({
            "store": [store],
            "item": [item],
            "year": [date.year],
            "month": [date.month],
            "day": [date.day],
            "dayofweek": [date.dayofweek],
        })

    if st.button("Faire la prédiction"):
        input_df = prepare_input(date_input, store, item)
        try:
            prediction = model.predict(input_df)[0]
            st.success(f"🔮 Prédiction des ventes : {prediction:.2f}")

            df_filtered = df[(df["store"] == store) & (df["item"] == item)].copy().sort_values("date")
            fig, ax = plt.subplots(figsize=(10,5))
            ax.plot(df_filtered["date"], df_filtered["sales"], label="Historique des ventes")
            ax.scatter([pd.to_datetime(date_input)], [prediction], color="red", label="Prédiction", zorder=5)
            ax.set_title(f"Évolution des ventes - Magasin {store}, Produit {item}")
            ax.set_xlabel("Date")
            ax.set_ylabel("Ventes")
            ax.legend()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")
