import pandas as pd
from src.inference import load_model_and_predict

def test_model_prediction():
    data = pd.DataFrame([{
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

    prediction, proba = load_model_and_predict(data)
    assert prediction in [0, 1]
    assert 0.0 <= proba <= 1.0
