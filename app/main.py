from fastapi import FastAPI
from app.schemes import CreditData
import pandas as pd
import joblib

app = FastAPI(title="Credit Risk Scoring API")

model = joblib.load("models/best_model.pkl")

@app.post("/predict")
def predict(data: CreditData):
    df = pd.DataFrame([data.dict()])

    prediction = model.predict(df)[0]
    proba = model.predict_proba(df)[0][1]

    return {
        "risk_prediction": int(prediction),
        "risk_probability": round(float(proba), 3)
    }