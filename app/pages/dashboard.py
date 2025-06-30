import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def show(df):
    st.header("📊 Dashboard")

    total_sales = df["sales"].sum()
    total_stores = df["store"].nunique()
    total_items = df["item"].nunique()

    col1, col2, col3 = st.columns(3)
    col1.metric("Ventes totales", f"{total_sales:,}")
    col2.metric("Nombre de magasins", total_stores)
    col3.metric("Nombre de produits", total_items)

    df['month_str'] = df['date'].dt.to_period('M')
    sales_by_month = df.groupby('month_str')['sales'].sum().reset_index()
    sales_by_month['month_str'] = sales_by_month['month_str'].dt.to_timestamp()

    fig, ax = plt.subplots(figsize=(10,5))
    sns.lineplot(data=sales_by_month, x='month_str', y='sales', ax=ax)
    ax.set_title("Ventes totales par mois")
    st.pyplot(fig)

    sales_by_dayofweek = df.groupby('dayofweek')['sales'].mean()
    day_names = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
    fig2, ax2 = plt.subplots()
    sns.barplot(x=day_names, y=sales_by_dayofweek.values, ax=ax2)
    ax2.set_title("Ventes moyennes par jour de la semaine")
    st.pyplot(fig2)

    sales_store_item = df.groupby(['store', 'item'])['sales'].sum().unstack(fill_value=0)
    fig3, ax3 = plt.subplots(figsize=(12,8))
    sns.heatmap(sales_store_item, cmap="YlGnBu", ax=ax3)
    ax3.set_title("Heatmap des ventes par magasin et produit")
    st.pyplot(fig3)

    sales_by_month['pct_change'] = sales_by_month['sales'].pct_change() * 100
    fig4, ax4 = plt.subplots(figsize=(10,4))
    sns.barplot(x=sales_by_month['month_str'].dt.strftime('%Y-%m'), y=sales_by_month['pct_change'], ax=ax4)
    ax4.set_title("Variation mensuelle des ventes (%)")
    plt.xticks(rotation=45)
    st.pyplot(fig4)
