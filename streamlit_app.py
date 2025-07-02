import streamlit as st
import pandas as pd
from api.model_loader import load_model_and_predict 

# Configuration de la page
st.set_page_config(page_title="Détection AVC", page_icon="🧠")
st.info("ℹ️ Ce site est à des fins de démonstration. Ne remplace pas un avis médical.")
st.title("🧠 Prédiction du Risque d'AVC")

# Introduction
st.markdown("""
##### 💡 Pourquoi ce projet ?

Les **accidents vasculaires cérébraux (AVC)** sont une cause majeure de **mortalité et de handicap**.

Ce projet vise à fournir un outil d’aide à la décision simple pour **détecter les patients à risque**  
et permettre une **prise en charge précoce**.

st.markdown("### 🔍 Facteurs de risque identifiés dans les données")
st.info("""
- **Âge élevé** : les AVC surviennent majoritairement chez les patients de **plus de 60 ans**.
- **Hypertension** : 13% des patients hypertendus ont eu un AVC, contre 4% des non-hypertendus.
- **Maladie cardiaque** : le risque est multiplié par 4 chez les patients cardiaques.
- **Glucose élevé** : des pics autour de 200 mg/dL sont fréquents chez les patients ayant subi un AVC.
- **Tabagisme** : les anciens fumeurs sont les plus à risque (7,9%).
""")

##### 🎯 Objectifs
- Identifier les profils à risque à partir de données médicales
- Offrir une interface intuitive pour les professionnels de santé
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
    except Exception as e:
        st.error(f"❌ Erreur de prédiction : {str(e)}")

# Pied de page
st.markdown("""
<hr style="border: 0.5px solid #ddd;">
<div style="text-align:center">
    <small>© 2025 – Outil de prédiction AVC – Fait avec ❤️ par Presley Koyaweda</small>
</div>
""", unsafe_allow_html=True)
