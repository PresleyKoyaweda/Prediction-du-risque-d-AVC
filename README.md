# 🧠 Détection d'AVC (AVC Prediction App)

![CI](https://github.com/PresleyKoyaweda/detection-avc/actions/workflows/ci.yaml/badge.svg)

Application de prédiction de risque d'AVC (Accident Vasculaire Cérébral) développée avec **scikit-learn**, **Streamlit** et **GitHub Actions** pour l'intégration continue.

---

## 🚀 Objectif

Prédire si une personne est à risque d'AVC à partir de données médicales et socio-démographiques telles que l'âge, le niveau de glucose, l'hypertension, etc.

---

## 🧰 Stack technique

- Python 3.10
- Scikit-learn
- Pandas / NumPy
- Streamlit
- Joblib
- GitHub Actions (CI)
- (MLflow/DVC – à venir)

---

## 🧪 Exemple d'utilisation

```python
from src.inference import load_model_and_predict
import pandas as pd

sample = pd.DataFrame([{
    "gender": "Male",
    "age": 67,
    "hypertension": 0,
    "heart_disease": 1,
    "ever_married": "Yes",
    "work_type": "Private",
    "Residence_type": "Urban",
    "avg_glucose_level": 228.69,
    "bmi": 36.6,
    "smoking_status": "formerly smoked"
}])

prediction, probability = load_model_and_predict(sample)
print("Prédiction :", prediction)
print("Probabilité :", probability)
