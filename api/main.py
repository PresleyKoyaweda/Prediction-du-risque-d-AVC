from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from src.inference import load_model_and_predict

app = FastAPI()

class InputData(BaseModel):
    gender: str
    age: float
    hypertension: int
    heart_disease: int
    ever_married: str
    work_type: str
    Residence_type: str
    avg_glucose_level: float
    bmi: float
    smoking_status: str

@app.post("/predict")
def predict_avc(input: InputData):
    data = pd.DataFrame([input.dict()])
    prediction, proba = load_model_and_predict(data)
    return {"prediction": int(prediction), "probability": float(proba)}
