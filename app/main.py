from fastapi import FastAPI
from app.schemes import CreditData
from src.logger_config import setup_logger
import pandas as pd
import joblib
#from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(title="Credit Risk Scoring API")

#instrumentator = Instrumentator()
#instrumentator.instrument(app).expose(app)
logger = setup_logger()

model = joblib.load("models/best_model.pkl")

@app.post("/predict")
def predict(data: CreditData):
    df = pd.DataFrame([data.dict()])

    prediction = model.predict(df)[0]
    proba = model.predict_proba(df)[0][1]

    logger.info(f"Prediction: {prediction}, Probability: {proba}, Data: {data.dict()}")

    return {
        "risk_prediction": int(prediction),
        "risk_probability": round(float(proba), 3)
    }