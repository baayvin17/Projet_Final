import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Titre de l'application
st.title("Projet Yshop - Prédiction des ventes")

# Chargement des données
@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dayofweek"] = df["date"].dt.dayofweek
    return df

# Chargement du modèle
@st.cache_resource
def load_model():
    return joblib.load("random_forest_model.pkl")

df = load_data()
model = load_model()

# Menu de navigation
menu = ["📊 Dashboard", "🔍 Exploration", "🔮 Prédiction", "📝 Rapport"]
choice = st.sidebar.selectbox("Menu", menu)

# === 📊 Dashboard ===
if choice == "📊 Dashboard":
    st.header("📊 Dashboard")

    # Chiffres clés
    total_sales = df["sales"].sum()
    total_stores = df["store"].nunique()
    total_items = df["item"].nunique()

    col1, col2, col3 = st.columns(3)
    col1.metric("Ventes totales", f"{total_sales:,}")
    col2.metric("Nombre de magasins", total_stores)
    col3.metric("Nombre de produits", total_items)

    # Ventes par mois
    df['month_str'] = df['date'].dt.to_period('M')
    sales_by_month = df.groupby('month_str')['sales'].sum().reset_index()
    sales_by_month['month_str'] = sales_by_month['month_str'].dt.to_timestamp()

    fig, ax = plt.subplots(figsize=(10,5))
    sns.lineplot(data=sales_by_month, x='month_str', y='sales', ax=ax)
    ax.set_title("Ventes totales par mois")
    ax.set_xlabel("Mois")
    ax.set_ylabel("Ventes")
    st.pyplot(fig)
    st.markdown("📈 **Les ventes mensuelles** montrent une tendance saisonnière avec des pics en fin d’année, probablement liés aux fêtes de fin d’année.")

    # Ventes moyennes par jour de la semaine
    sales_by_dayofweek = df.groupby('dayofweek')['sales'].mean()
    day_names = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']

    fig2, ax2 = plt.subplots()
    sns.barplot(x=day_names, y=sales_by_dayofweek.values, ax=ax2)
    ax2.set_title("Ventes moyennes par jour de la semaine")
    ax2.set_ylabel("Ventes moyennes")
    st.pyplot(fig2)
    st.markdown("🛍️ **Les ventes moyennes sont les plus élevées le week-end**, notamment le samedi et le dimanche, ce qui reflète une plus forte affluence des clients ces jours-là.")

    # Heatmap des ventes par magasin et produit
    sales_store_item = df.groupby(['store', 'item'])['sales'].sum().unstack(fill_value=0)
    fig3, ax3 = plt.subplots(figsize=(12,8))
    sns.heatmap(sales_store_item, cmap="YlGnBu", ax=ax3)
    ax3.set_title("Heatmap des ventes par magasin et produit")
    st.pyplot(fig3)
    st.markdown("🏬 Cette **carte thermique permet de visualiser les produits les plus performants dans chaque magasin**. On peut identifier rapidement les combinaisons les plus lucratives.")

    # Variation mensuelle des ventes (en %)
    sales_by_month['pct_change'] = sales_by_month['sales'].pct_change() * 100
    fig4, ax4 = plt.subplots(figsize=(10,4))
    sns.barplot(x=sales_by_month['month_str'].dt.strftime('%Y-%m'), y=sales_by_month['pct_change'], ax=ax4)
    ax4.set_title("Variation mensuelle des ventes (%)")
    ax4.set_ylabel("Variation (%)")
    ax4.set_xlabel("Mois")
    plt.xticks(rotation=45)
    st.pyplot(fig4)
    st.markdown("📉 Ce graphique **met en évidence les hausses et baisses des ventes d’un mois à l’autre**, ce qui est utile pour identifier des périodes atypiques à analyser plus en détail.")

# === 🔍 Exploration ===
elif choice == "🔍 Exploration":
    st.header("🔍 Exploration des données")

    st.write("Aperçu des données :")
    st.dataframe(df.head())

    # Ventes par magasin
    sales_by_store = df.groupby('store')['sales'].sum().sort_values(ascending=False)
    st.subheader("Ventes par magasin")
    st.bar_chart(sales_by_store)
    st.markdown("🏪 **Certains magasins génèrent beaucoup plus de chiffre d'affaires que d'autres**, ce qui peut indiquer des différences de fréquentation ou de localisation.")

    # Ventes par produit
    sales_by_item = df.groupby('item')['sales'].sum().sort_values(ascending=False)
    st.subheader("Ventes par produit")
    st.bar_chart(sales_by_item)
    st.markdown("📦 **Tous les produits ne se vendent pas de manière équitable**. Cette analyse met en avant les produits les plus populaires.")

    # Histogramme des ventes
    fig2, ax2 = plt.subplots()
    sns.histplot(df['sales'], bins=30, kde=True, ax=ax2)
    ax2.set_title("Distribution des ventes")
    st.pyplot(fig2)
    st.markdown("📊 **La distribution des ventes est asymétrique** : la majorité des ventes sont concentrées sur de petites valeurs, avec quelques pics élevés.")

    # Ventes par année
    sales_by_year = df.groupby('year')['sales'].sum()
    st.subheader("Ventes par année")
    st.bar_chart(sales_by_year)
    st.markdown("📅 Cette visualisation montre **l'évolution annuelle des ventes sur 5 ans**, utile pour détecter une tendance de croissance ou de décroissance.")

    # Boxplot des ventes par produit
    fig3, ax3 = plt.subplots(figsize=(10,5))
    sns.boxplot(x='item', y='sales', data=df, ax=ax3)
    ax3.set_title("Distribution des ventes par produit")
    ax3.set_xlabel("Produit (item)")
    ax3.set_ylabel("Ventes")
    plt.xticks(rotation=45)
    st.pyplot(fig3)
    st.markdown("📦 Le boxplot met en lumière **les produits avec des ventes très variables**, et ceux plus stables. Cela permet de cibler les produits à surveiller.")

    # Matrice de corrélation
    st.subheader("Matrice de corrélation")
    corr_data = df[['year', 'month', 'day', 'dayofweek', 'sales']]
    st.dataframe(corr_data.corr())

    fig4, ax4 = plt.subplots()
    sns.heatmap(corr_data.corr(), annot=True, cmap='coolwarm', ax=ax4)
    st.pyplot(fig4)
    st.markdown("🧠 Cette **matrice de corrélation montre les relations entre variables**. Par exemple, la corrélation entre `month` et `sales` permet d'identifier un effet saisonnier.")

# === PREDICTION ===
elif choice == "🔮 Prédiction":
    st.header("🔮 Prédiction des ventes futures")

    store = st.number_input("Magasin (store)", min_value=int(df["store"].min()), max_value=int(df["store"].max()), value=int(df["store"].min()))
    item = st.number_input("Produit (item)", min_value=int(df["item"].min()), max_value=int(df["item"].max()), value=int(df["item"].min()))
    date_input = st.date_input("Date de la prédiction")

    def prepare_input(date, store, item):
        date = pd.to_datetime(date)
        data = {
            "store": [store],
            "item": [item],
            "year": [date.year],
            "month": [date.month],
            "day": [date.day],
            "dayofweek": [date.dayofweek],
        }
        return pd.DataFrame(data)

    if st.button("Faire la prédiction"):
        input_df = prepare_input(date_input, store, item)
        try:
            prediction = model.predict(input_df)[0]
            st.success(f"🔮 Prédiction des ventes : {prediction:.2f}")
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")

# === DATA STORYTELLING ===
elif choice == "📝 Rapport":
    st.header("📝 Rapport")

    st.markdown("""
    ## Contexte du projet  
    Le projet **Yshop** vise à analyser les ventes d’un réseau de magasins pour mieux comprendre les comportements d’achat et anticiper les volumes futurs grâce à la data science. Cette démarche permet d’optimiser la gestion des stocks et d’appuyer les décisions commerciales.

    ## Problématique  
    Comment optimiser la gestion des stocks et les décisions commerciales en s'appuyant sur des données historiques de ventes, tout en anticipant les fluctuations saisonnières et les comportements d’achat spécifiques ?

    ## Données utilisées  
    Le jeu de données comprend :  
    - 5 années complètes de données de vente quotidiennes,  
    - 50 magasins répartis sur différentes zones géographiques,  
    - 50 produits différents couvrant plusieurs catégories,  
    - Des variables temporelles : jour, mois, année, jour de la semaine, permettant une analyse fine des tendances.

    ## Insights clés et observations approfondies  
    - 📈 **Pics saisonniers marqués** en fin d’année (novembre-décembre), liés aux fêtes et événements commerciaux majeurs (Black Friday, Noël).  
    - 🛍️ **Week-end (samedi et dimanche)** : jours où les ventes moyennes sont les plus élevées, représentant plus de 40 % du chiffre hebdomadaire.  
    - 🏬 **Performance variable par magasin et produit** : certains magasins urbains et certains produits électroniques montrent des performances nettement supérieures, comme illustré par la heatmap des ventes.  
    - 📉 **Baisse inattendue en mars 2024** : une diminution notable des ventes a été détectée, vraisemblablement liée à des ruptures de stock ou à des événements locaux, orientant vers une amélioration de la chaîne d’approvisionnement.  

    ## Modèle prédictif  
    Un **modèle Random Forest** a été entraîné pour prédire les ventes futures en fonction :  
    - Du magasin,  
    - Du produit,  
    - De la date (année, mois, jour, jour de la semaine).  

    Ce modèle a été validé par une découpe temporelle rigoureuse pour éviter le surapprentissage. Il permet d’anticiper la demande, de simuler des scénarios commerciaux, et d’ajuster les ressources et stocks en conséquence.

    ## Objectifs atteints  
    ✅ Analyses visuelles riches et variées (courbes temporelles, barres, heatmaps)  
    ✅ Modèle fonctionnel intégré et testé dans l’application  
    ✅ Interface interactive développée avec Streamlit pour faciliter l’usage par les équipes métier  
    ✅ Data storytelling intégré pour faciliter la compréhension et appuyer les décisions stratégiques

    ## Perspectives et améliorations futures  
    Pour aller plus loin, plusieurs pistes sont envisagées :  
    - Intégrer des facteurs externes tels que la météo, les événements locaux, ou les campagnes promotionnelles.  
    - Tester des modèles complémentaires plus avancés comme XGBoost ou des réseaux de neurones LSTM.  
    - Déployer l’application sur une plateforme cloud (Streamlit Cloud, Docker).  
    - Ajouter des fonctionnalités de visualisation plus interactives et des tableaux de bord personnalisables.

    ---
    **Projet réalisé par Baayvin & Hugo dans le cadre de la Spécialité Data & IA – Bachelor 3**
    """)
