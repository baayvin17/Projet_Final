import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

def show(df):
    st.header("🔍 Exploration des données")

    st.dataframe(df.head())

    sales_by_store = df.groupby('store')['sales'].sum().sort_values(ascending=False)
    st.subheader("Ventes par magasin")
    st.bar_chart(sales_by_store)

    sales_by_item = df.groupby('item')['sales'].sum().sort_values(ascending=False)
    st.subheader("Ventes par produit")
    st.bar_chart(sales_by_item)

    fig, ax = plt.subplots()
    sns.histplot(df['sales'], bins=30, kde=True, ax=ax)
    ax.set_title("Distribution des ventes")
    st.pyplot(fig)

    sales_by_year = df.groupby('year')['sales'].sum()
    st.subheader("Ventes par année")
    st.bar_chart(sales_by_year)

    fig2, ax2 = plt.subplots(figsize=(10,5))
    sns.boxplot(x='item', y='sales', data=df, ax=ax2)
    ax2.set_title("Distribution des ventes par produit")
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    st.subheader("Matrice de corrélation")
    corr_data = df[['year', 'month', 'day', 'dayofweek', 'sales']]
    fig3, ax3 = plt.subplots()
    sns.heatmap(corr_data.corr(), annot=True, cmap='coolwarm', ax=ax3)
    st.pyplot(fig3)
