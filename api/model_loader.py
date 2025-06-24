# api/model_loader.py
import mlflow.sklearn
import pandas as pd
import joblib
import os

MODEL_URI = "D:/Perso/Detection_AVC/models/final_model.joblib"  # ou chemin vers le dossier ./models/model_name

def load_model_and_predict(input_df: pd.DataFrame):
    #model = mlflow.sklearn.load_model(MODEL_URI)
    model = joblib.load(MODEL_URI)

    prediction_proba = model.predict_proba(input_df)[:, 1]
    prediction = (prediction_proba >= 0.5).astype(int)[0]

    return prediction, float(prediction_proba[0])
