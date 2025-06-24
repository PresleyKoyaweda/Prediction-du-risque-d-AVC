import pandas as pd
from joblib import load

MODEL_PATH = "models/final_model.joblib"

def load_model_and_predict(df: pd.DataFrame):
    model = load(MODEL_PATH)
    prediction = model.predict(df)[0]
    proba = model.predict_proba(df)[0][1]
    return prediction, proba
