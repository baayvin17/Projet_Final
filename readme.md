# 📊 Projet Yshop - Prédiction des Ventes

Projet final de 3ᵉ année Bachelor Ynov – Spécialité Data & IA  
Réalisé en binôme

## 🧠 Objectif

Créer une **application interactive** en Python permettant :
- D’explorer les données de ventes (analyse exploratoire)
- D’appliquer un modèle de **Machine Learning** pour prédire les ventes futures
- De visualiser les résultats via un **dashboard dynamique** (Streamlit)

---

## ⚙️ Technologies utilisées

- **Python 3**
- **Pandas** & **NumPy** – Manipulation des données
- **Matplotlib** & **Seaborn** – Visualisations statiques
- **Scikit-learn** – Modélisation (Random Forest)
- **Streamlit** – Création de l’application
- **Joblib** – Sauvegarde du modèle

---

## 📁 Structure du projet

├── app.py # Application principale Streamlit
├── train.csv # Données de ventes historiques
├── random_forest_model.pkl # Modèle ML sauvegardé
└── README.md # Présentation du projet


---

## 🧪 Fonctionnalités

### 1. 📈 Dashboard
- KPIs : ventes totales, nombre de magasins, nombre de produits
- Ventes par mois, par jour de semaine
- Heatmap des ventes par magasin et produit
- Taux de variation mensuelle

### 2. 🔍 Exploration
- Ventes par magasin, produit, année
- Histogramme et boxplots
- Corrélation entre variables

### 3. 🤖 Prédiction
- Saisie de la date, du magasin, du produit
- Modèle Random Forest entraîné sur les données historiques
- Résultat prédit affiché en temps réel

---

## 🚀 Lancer l’application

### 1. Installer les dépendances
Assurez-vous d’avoir Python 3 installé, puis :

```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit joblib
