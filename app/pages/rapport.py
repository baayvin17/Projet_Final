import streamlit as st

def show():
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
