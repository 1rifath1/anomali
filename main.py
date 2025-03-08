from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# Load the saved Random Forest model and scaler
def load_model():
    model = joblib.load("random_forest_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

rf_model, scaler = load_model()

app = FastAPI()

# Define request body schema
class AnomalyRequest(BaseModel):
    request_interval: float
    token_length: int
    model_number: int

# Prediction function
def check_anomaly(request_interval, token_length, model_number):
    input_data = np.array([[request_interval, token_length, model_number]])
    input_scaled = scaler.transform(input_data)
    prediction = rf_model.predict(input_scaled)
    anomaly_prob = rf_model.predict_proba(input_scaled)[0][1] * 100
    result = "Anomalous" if prediction[0] == 1 else "Not Anomalous"
    return result, anomaly_prob

# /predict endpoint
@app.post("/predict")
async def predict_anomaly(data: AnomalyRequest):
    result, probability = check_anomaly(data.request_interval, data.token_length, data.model_number)
    return {"result": result, "anomaly_percentage": f"{probability:.2f}%"}
