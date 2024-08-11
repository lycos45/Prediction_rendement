import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split as tts, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import statsmodels.formula.api as smf
import joblib
import plotly.express as px
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
import statsmodels.formula.api as smf
import io
import time

# Fonction de prétraitement des données
def wrangle(filepath):
    df = pd.read_csv(filepath, sep=';', na_values=["", " ", "  ", "   ", "    ", np.nan], low_memory=False)

    # Remplacer le d'un variable
    df['pratique_cultures_fourrageres'] = df['1. pratique_cultures_fourrageres']
    df.drop(columns="1. pratique_cultures_fourrageres", inplace=True)

    # Création du variable rendement et suppression des autres rendements
    def merge_columns(row):
        cols = [row['rend_moyen_arachide'], row['rend_moyen_mil'], row['rend_moyen_niebe'], row['rend_moyen_mais'],
                row['rend_moyen_sorgho'], row['rend_moyen_fonio'], row['rend_moyen_riz_irrigue'],
                row['rend_moyen_riz_pluvial']]
        non_nan_values = [v for v in cols if pd.notna(v)]

        if len(non_nan_values) == 0:
            return np.nan
        elif len(non_nan_values) >= 2:
            return 0
        else:
            return non_nan_values[0]

    df['rendement'] = df.apply(merge_columns, axis=1)
    df.drop(
        columns=['rend_moyen_arachide', 'rend_moyen_mil', 'rend_moyen_niebe', 'rend_moyen_mais', 'rend_moyen_sorgho',
                 'rend_moyen_fonio', 'rend_moyen_riz_irrigue', 'rend_moyen_riz_pluvial'], inplace=True)

    # Définir les colonnes à convertir
    numcol_convert = ['rendement', 'superficie_parcelle', 'pourcentage_superficie_pastèque', 'pourcentage_CP',
                      'quantite_semence_CP', 'quantite_semence_sub_CP', 'quantite_semence_marche_specialisee_CP ',
                      'quantite_semence_reserve_personnelle_CP', 'quantite_semence_don_CP',
                      'quantite_CP', 'pourcentage_CS', 'quantite_semence_CS', 'quantite_semence_sub_CS',
                      'quantite_semence_marche_specialisee_CS', 'quantite_semence_reserve_personnelle_CS',
                      'quantite_semence_don_CS',
                      'quantite_CS', 'quantite_NPK_epandage_avant_recolte', 'quantite_urée_epandage_avant_recolte',
                      'nombre_pieds_arachide_compte', 'nombre_pieds_arachide_recoltes',
                      'poids_total_gousses_arachide_recoltés', 'poids_moyen_arachide_en_gramme',
                      'poids_recolte_arachide', 'nombre_pieds_mil_compte', 'nombre_epis_potentiel_maturite_mil',
                      'nombre_epis_prélevé_mil', 'poids_total_graines_battage_sechage_mil', 'poids_moyen_mil_en_gramme',
                      'poids_recolte_mil',
                      'nombre_pieds_niebe_compte', 'nombre_gousses_niebe_3_pieds', 'nombre_gousses_niebe_par_pieds',
                      'nombre_total_gousses_niebe', 'nombre_gousses_niebe_preleve',
                      'poids_total_gousses_niebe_apres_egrainage', 'poids_moyen_gousses_niebe',
                      'poids_total_niebe_de_la_recolte', 'nombre_pieds_mais_compte',
                      'nombre_epis_potentiel_maturite_mais', 'nombre_epis_mais_preleve',
                      'poids_total_graines_battage_sechage_mais',
                      'poids_moyen_mais_en_gramme', 'poids_recolte_mais', 'nombre_sorgho_compte',
                      'nombre_epis_potentiel_maturite_sorgho', 'nombre_epis_sorgho_preleve',
                      'poids_total_graines_battage_sechage_sorgho', 'poids_moyen_sorgho_en_gramme',
                      'poids_recolte_sorgho', 'poids_recolte_fonio', 'nombre_pieds_riz_irrigue_compte',
                      'nombre_epis_potentiel_maturite_riz_irrigue',
                      'nombre_epis_riz_irrigue_preleve', 'poids_total_graines_battage_sechage_riz_irrigue',
                      'poids_moyen_riz_irrigue_en_gramme', 'poids_recolte_riz_irrigue',
                      'nombre_pieds_riz_pluvial_compte', 'nombre_epis_potentiel_maturite_riz_pluvial',
                      'nombre_epis_riz_pluvial_preleve', 'poids_total_graines_battage_sechage_riz_pluvial',
                      'poids_moyen_riz_pluvial_en_gramme', 'poids_recolte_riz_pluvial',
                      'superficie(ha)_cultures_fourrageres']

    catcol_convert = ['id_reg', 'culture_principale', 'irrigation', 'pastèque_fin_saison', 'rotation_cultures',
                      'culture_principale_précédente_2020_2021', 'méthode_de_culture', 'culture_secondaire',
                      'varietes_arachides_CP', 'origine_semence_CP',
                      'variete_riz_CP', 'type_semence_CP', 'mois_semis_CP', 'semaine_semis_CP', 'etat_produit_CP',
                      'mois_previsionnel_recolte_CP', 'varietes_arachides_CS', 'origine_semence_CS', 'variete_riz_CS',
                      'type_semence_CS', 'mois_semis_CS',
                      'semaine_semis_CS', 'etat_produit_CS', 'mois_previsionnel_recolte_CS',
                      'utilisation_matiere_org_avant_semis', 'matieres_organiques',
                      'utilisation_matiere_miner_avant_semis', 'matieres_minerales',
                      'prevision_epandage_engrais_min_avant_recolte',
                      'type_engrais_mineral_epandage_avant_recolte', 'utilisation_produits_phytosanitaires',
                      'utilisation_residus_alimentation_animaux', 'type_residus_alimentation_animaux', 'type_labour',
                      'type_couverture_sol_interculture',
                      'type_installation', 'pratiques_conservation_sol', 'contraintes_production',
                      'type_materiel_preparation_sol', 'type_materiel_semis', 'type_materiel_entretien_sol',
                      'type_materiel_récolte',
                      'type_culture', 'rendement_arachide', 'rendement_mil', 'rendement_niebe', 'rendement_mais',
                      'rendement_riz_irrigue', 'rendement_sorgho', 'rendement_fonio', 'rendement_riz_pluvial',
                      'pratique_cultures_fourrageres',
                      'cultures_fourrageres_exploitees', 'production_forestiere_exploitation']

    # Conversion des types
    for col in numcol_convert:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace(',', '.')
        df[col] = df[col].astype('float64')

    for col in catcol_convert:
        df[col] = df[col].astype('category')

    # Supprimer les variables avec plus de 50% des données manquantes
    missing_percent = df.drop(columns="rendement").isnull().mean() * 100
    columns_to_drop = missing_percent[missing_percent > 50].index.tolist()
    df.drop(columns=columns_to_drop, inplace=True)

    # Imputation des valeurs manquantes avec HistGradientBoostingRegressor
    def impute_missing_values(data, column, model):
        data_missing = df[df[column].isnull()]
        data_not_missing = df[df[column].notnull()]

        X_missing = data_missing.drop(columns=[column])
        X_not_missing = data_not_missing.drop(columns=[column])
        y_not_missing = data_not_missing[column]

        model.fit(X_not_missing, y_not_missing)
        y_missing_pred = model.predict(X_missing)
        df.loc[df[column].isnull(), column] = y_missing_pred
        return df

    model = HistGradientBoostingRegressor()
    numeric_columns = df.select_dtypes(include=['float64']).columns.tolist()

    for column in numeric_columns:
        df = impute_missing_values(df, column, model)

    # Imputation des valeurs catégorielles avec RandomForestClassifier
    def impute_categorical_values(data, column):
        data_missing = df[df[column].isnull()]
        data_not_missing = df[df[column].notnull()]

        X_missing = data_missing.drop(columns=[column])
        X_not_missing = data_not_missing.drop(columns=[column])
        y_not_missing = data_not_missing[column].cat.codes

        model = RandomForestClassifier()
        model.fit(X_not_missing, y_not_missing)
        y_missing_pred = model.predict(X_missing)
        df.loc[df[column].isnull(), column] = pd.Categorical.from_codes(y_missing_pred, df[column].cat.categories)
        return df

    categorical_columns = df.select_dtypes(include=['category']).columns.tolist()

    for column in categorical_columns:
        df = impute_categorical_values(df, column)

    return df

# Fonction principale
def afficher_infos(df):
    st.write("Aperçu des données prétraitées :", df.head())
    st.write("Informations sur le DataFrame :")
    
    # Utilisation d'un buffer pour capturer le texte de df.info()
    buffer = io.StringIO()
    df.info(buf=buffer)
    infos = buffer.getvalue()

    # Affichage des informations capturées
    st.text(infos)

def main():
    st.title('Application de Prédiction des Rendements Agricoles')

    # Barre latérale
    st.sidebar.title('Navigation')
    option = st.sidebar.radio("Choisissez une section", 
                              ["Aperçu des Données", 
                               "Statistiques Descriptives", 
                               "Matrice de Corrélation", 
                               "Boxplots", 
                               "ANOVA",
                               "Prédiction"])

    # Charger les données depuis le fichier local
    filepath = "base.csv"  # Change ce chemin selon la localisation de ton fichier
    df = wrangle(filepath)

    # Définir la cible et les variables explicatives
    target = 'rendement'
    non_significatif = []  # Assurez-vous que cette liste est remplie correctement
    x = df.drop(columns=[target] + non_significatif)
    y = df[target]

    numeric_vars = ['superficie_parcelle', 'quantite_semence_CP', 'quantite_CP']  # Ajuster en fonction de tes variables numériques
    categorical_vars = ['id_reg',
    'culture_principale',
    'irrigation',
    'rotation_cultures',
    'culture_principale_précédente_2020_2021',
    'méthode_de_culture',
    'origine_semence_CP',
    'type_semence_CP',
    'mois_semis_CP',
    'semaine_semis_CP',
    'etat_produit_CP',
    'mois_previsionnel_recolte_CP',
    'utilisation_matiere_org_avant_semis',
    'utilisation_matiere_miner_avant_semis',
    'utilisation_produits_phytosanitaires',
    'utilisation_residus_alimentation_animaux',
    'type_labour',
    'type_couverture_sol_interculture',
    'type_installation',
    'pratiques_conservation_sol',
    'type_materiel_preparation_sol',
    'type_materiel_semis',
    'type_materiel_entretien_sol',
    'type_materiel_récolte',
    'production_forestiere_exploitation',
    'pratique_cultures_fourrageres']

    if option == "Aperçu des Données":
        st.subheader('Aperçu des Données')
        st.write(df.head())

    elif option == "Statistiques Descriptives":
        st.subheader('Statistiques Descriptives Univariées')
        selected_cols = st.multiselect('Sélectionnez les variables', numeric_vars)
        if selected_cols:
            st.write(df[selected_cols].describe().T)
        else:
            st.write(df[numeric_vars].describe().T)

        st.subheader('Statistiques Descriptives des Variables Catégorielles')
        st.write(df[categorical_vars].describe().T)

    elif option == "Matrice de Corrélation":
        numeric_corr = df[numeric_vars].corr()
        st.subheader('Matrice de Corrélation')
        fig = plt.figure(figsize=(10, 8))
        ax = sns.heatmap(numeric_corr, annot=True, fmt=".2f", cmap='RdBu_r', center=0, linewidths=0.5, linecolor='gray', cbar_kws={'shrink': 0.5})
        ax.set_title('Matrice de Corrélation - Variables Numériques', fontsize=16, pad=20)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        st.pyplot(fig)

    elif option == "Boxplots":
        st.subheader('Boxplots des Variables Numériques')
        for var in numeric_vars:
            fig, ax = plt.subplots()
            sns.boxplot(data=df, x=var, ax=ax)
            ax.set_title(f'Boxplot for {var}')
            st.pyplot(fig)

    elif option == "ANOVA":
        st.subheader('ANOVA pour Variables Catégorielles')
        significant_variables = []
        non_significatif = []
        for categorical_var in categorical_vars:
            unique_values = df[categorical_var].nunique()
            if unique_values > 1:
                formula = f'rendement ~ C({categorical_var})'
                model = smf.ols(formula, data=df).fit()
                anova_table = sm.stats.anova_lm(model, typ=2)
                p_value = anova_table["PR(>F)"][0]

                if p_value < 0.05:
                    significant_variables.append(categorical_var)
                else:
                    non_significatif.append(categorical_var)

        st.write("Variables significatives :", significant_variables)
        st.write("Variables non significatives :", non_significatif)

    elif option == "Prédiction":
        # Séparer les données
       # Séparer les données
        x_train, x_test, y_train, y_test = tts(x, y, test_size=0.2, random_state=3)
        
        # Initialiser et appliquer le scaler
        scaler = StandardScaler()
        x_train[numeric_vars] = scaler.fit_transform(x_train[numeric_vars])
        x_test[numeric_vars] = scaler.transform(x_test[numeric_vars])

        # Instancier le modèle
        model = RandomForestRegressor()

        # Définir la validation croisée
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        # Mesurer le temps d'entraînement
        start_time = time.time()
        model.fit(x_train, y_train)
        end_time = time.time()

        # Évaluer les scores de validation croisée
        cv_scores = cross_val_score(model, x_train, y_train, cv=kf, scoring='r2')

        # Préparer les résultats dans un DataFrame
        results = pd.DataFrame([{
            "modele": "RandomForestRegressor",
            "temps": end_time - start_time,
            "score_train": model.score(x_train, y_train),
            "score_test": model.score(x_test, y_test),
            "cv_score_mean": cv_scores.mean(),
            "cv_score_std": cv_scores.std()
        }])

        st.write("Évaluation du Modèle")
        st.write(results)

        # Entrer les valeurs pour la prédiction
        st.subheader('Prédiction des Rendements')
        input_data = {}

        # Variables numériques
        for var in numeric_vars:
            min_val, max_val = df[var].min(), df[var].max()
            input_data[var] = st.slider(f'Valeur pour {var}', min_value=float(min_val), max_value=float(max_val), value=float((min_val + max_val) / 2))

        # Variables catégorielles
        for var in categorical_vars:
            categories = df[var].unique()
            selected_category = st.selectbox(f'Sélectionnez la catégorie pour {var}', categories)
            input_data[var] = selected_category

        # Créer un DataFrame avec toutes les colonnes (même celles sans valeurs spécifiques)
        input_df = pd.DataFrame([input_data])
        # Ajouter les colonnes manquantes avec des valeurs par défaut (par exemple, NaN)
        for col in x.columns:
            if col not in input_df.columns:
                input_df[col] = np.nan

        # S'assurer que les colonnes sont dans le même ordre que x_train
        input_df = input_df[x.columns]

        # Normaliser les données numériques
        scaled_input_df = input_df.copy()
        scaled_input_df[numeric_vars] = scaler.transform(input_df[numeric_vars].fillna(0))  # Remplacer les NaN par 0

        # Faire la prédiction
        prediction = model.predict(scaled_input_df)
        st.write(f'Prédiction du rendement : {prediction[0]:.2f}')

if __name__ == "__main__":
    main()