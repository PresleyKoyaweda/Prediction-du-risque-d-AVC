import streamlit as st
import pandas as pd
from api.model_loader import load_model_and_predict 
from datetime import datetime
import os 

st.set_page_config(page_title="Détection AVC", page_icon="🧠")

#Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Selectionner une page",["Apercu et Prédiction avec le modèle", "Monitoring des Prédictions"])

if page == "Apercu et Prédiction avec le modèle":

    # Configuration de la page
    st.info("ℹ️ Ce site est à des fins de démonstration. Ne remplace pas un avis médical.")
    st.title("🧠 Prédiction du Risque d'AVC")

    # Introduction
    st.markdown("""
    ##### 💡 Pourquoi ce projet ?

    Les **accidents vasculaires cérébraux (AVC)** sont une cause majeure de **mortalité et de handicap**.

    Ce projet vise à fournir un outil d’aide à la décision simple pour **détecter les patients à risque**  
    et permettre une **prise en charge précoce**.

    ##### 🔍 Facteurs de risque identifiés dans les données

    - **Âge élevé** : les AVC surviennent majoritairement chez les patients de **plus de 60 ans**.
    - **Hypertension** : 13% des patients hypertendus ont eu un AVC, contre 4% des non-hypertendus.
    - **Maladie cardiaque** : le risque est multiplié par 4 chez les patients cardiaques.
    - **Glucose élevé** : des pics autour de 200 mg/dL sont fréquents chez les patients ayant subi un AVC.
    - **Tabagisme** : les anciens fumeurs sont les plus à risque (7,9%).

    ##### 🎯 Objectifs
    - Identifier les profils à risque à partir de données médicales
    - Offrir une interface intuitive pour les professionnels de santé

    Le jeu de données utilisé présente un déséquilibre marqué : **seulement 4,87 % des 5210 observations concernent des patients ayant subi un AVC**. Pour remédier à ce déséquilibre, des techniques de rééchantillonnage ont été utilisées afin d’améliorer la performance du modèle sur la classe minoritaire.
    """)

    # Formulaire de saisie
    st.markdown("Entrez les informations médicales pour estimer le risque d'AVC lié au patient.")

    with st.form("form"):
        gender = st.selectbox("Sexe", ["Male", "Female", "Other"])
        age = st.slider("Âge", 0, 100, 50)
        hypertension = st.selectbox("Hypertension", ["Yes", "No"])
        heart_disease = st.selectbox("Maladie cardiaque", ["Yes", "No"])
        ever_married = st.selectbox("Avez-vous déjà été marié(e) ?", ["Yes", "No"])
        work_type = st.selectbox("Type de travail", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
        residence_type = st.selectbox("Type de résidence", ["Urban", "Rural"])
        avg_glucose_level = st.number_input("Taux moyen de glucose", min_value=50.0, max_value=300.0, value=100.0)
        bmi = st.number_input("IMC", min_value=10.0, max_value=60.0, value=25.0)
        smoking_status = st.selectbox("Statut tabagique", ["never smoked", "formerly smoked", "smokes", "Unknown"])
        
        submitted = st.form_submit_button("Prédire")

    # Appel modèle après soumission
    if submitted:
        input_data = pd.DataFrame([{
            "gender": gender,
            "age": age,
            "hypertension": 1 if hypertension == "Yes" else 0,
            "heart_disease": 1 if heart_disease == "Yes" else 0,
            "ever_married": ever_married,
            "work_type": work_type,
            "Residence_type": residence_type,
            "avg_glucose_level": avg_glucose_level,
            "bmi": bmi,
            "smoking_status": smoking_status
        }])

        try:
            prediction, proba = load_model_and_predict(input_data)
            st.success(f"✅ Prédiction AVC : {'Oui (1)' if prediction else 'Non (0)'}")
            st.info(f"🧪 Probabilité estimée : {round(proba, 3)}")
            
            
            #Log de prediction
            log_data = input_data.copy()
            log_data["prediction"] = prediction
            log_data["probability"] = proba
            log_data["timestamp"] = datetime.now().isoformat()

            #Ajout a mon CVS
            os.makedirs("logs", exist_ok=True)
            
            log_file = "logs/predictions.csv"
            write_header = not os.path.exists(log_file) or os.stat(log_file).st_size == 0
            
            log_data.to_csv(log_file, mode="a", header=write_header, index=False)
            
            
        except Exception as e:
            st.error(f"❌ Erreur de prédiction : {str(e)}")
            

    # Pied de page
    st.markdown("""
    <hr style="border: 0.5px solid #ddd;">
    <div style="text-align:center">
        <small>© 2025 – Outil de prédiction AVC – Fait avec ❤️ par Presley Koyaweda</small>
    </div>
    """, unsafe_allow_html=True)
    
elif  page == "Monitoring des Prédictions":
    
    st.title("📊 Monitoring des prédictions en temps réel")

    log_file = "logs/predictions.csv"

    if not os.path.exists(log_file):
        st.warning("📭 Aucune donnée de prédiction trouvée pour le moment.")
        st.stop()

    # Chargement des prédictions
    df = pd.read_csv(log_file)

    # Vérifie que les colonnes sont correctes
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
    else:
        st.error("⛔ La colonne 'timestamp' est absente du fichier CSV.")
        st.stop()

    # Statistiques de base
    st.subheader("📌 Statistiques générales")
    st.write(df[["predictions", "probability"]].describe())

    # Répartition des prédictions
    st.subheader("📊 Répartition des prédictions")
    df["predictions_label"] = df["predictions"].map({0: "Pas d'AVC", 1: "AVC"})
    st.bar_chart(df["predictions_label"].value_counts(normalize=True))

    # Évolution temporelle
    st.subheader("🕒 Évolution temporelle de la probabilité")
    if len(df) < 10:
        st.warning("⏳ Pas encore assez de prédictions pour afficher une moyenne glissante (minimum 10).")
    else:
        st.line_chart(df.set_index("timestamp")["probability"].rolling(window=10).mean())

    # Données brutes
    with st.expander("📄 Voir les données brutes"):
        st.dataframe(df.tail(20))
        
        
    # Pied de page
    st.markdown("""
    <hr style="border: 0.5px solid #ddd;">
    <div style="text-align:center">
        <small>© 2025 – Outil de prédiction AVC – Fait avec ❤️ par Presley Koyaweda</small>
    </div>
    """, unsafe_allow_html=True)




