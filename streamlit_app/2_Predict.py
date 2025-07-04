import streamlit as st
import requests

st.set_page_config(page_title="Prédiction AVC", page_icon="🧪")

st.title("🧪 Prédiction du Risque d'AVC")
st.markdown("Entrez les informations médicales pour estimer le risque d'AVC.")

# --- Formulaire utilisateur ---
gender = st.selectbox("Sexe", ["Male", "Female"])
age = st.slider("Âge", 0, 100, 50)
hypertension = st.selectbox("Hypertension", ["Yes", "No"])
heart_disease = st.selectbox("Maladie cardiaque", ["Yes", "No"])
ever_married = st.selectbox("Avez-vous déjà été marié(e) ?", ["Yes", "No"])
work_type = st.selectbox("Type de travail", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
residence_type = st.selectbox("Type de résidence", ["Urban", "Rural"])
avg_glucose_level = st.number_input("Niveau moyen de glucose", min_value=0.0, step=0.1)
bmi = st.number_input("Indice de masse corporelle (BMI)", min_value=0.0, step=0.1)
smoking_status = st.selectbox("Statut tabagique", ["never smoked", "formerly smoked", "smokes", "Unknown"])

# --- Prédiction ---
if st.button("Prédire le risque"):
    input_data = {
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
    }

    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=input_data)
        result = response.json()
        st.success(f"✅ Risque prédictif d'AVC : {'Oui' if result['prediction'] == 1 else 'Non'}")
        st.write(f"📊 Probabilité : {result['probability']:.2%}")
    except Exception as e:
        st.error(f"❌ Erreur : {e}")
