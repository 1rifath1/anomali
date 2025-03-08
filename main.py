from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import os

# Ensure model files exist
MODEL_PATH = "random_forest_model.pkl"
SCALER_PATH = "scaler.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise FileNotFoundError("Model or Scaler file not found! Ensure they exist in the correct directory.")

# Load the saved Random Forest model and scaler
def load_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)  # Ensure the scaler is also saved and loaded
    return model, scaler

rf_model, scaler = load_model()

# Initialize FastAPI app
app = FastAPI()

# Define request body schema
class AnomalyRequest(BaseModel):
    request_interval: float
    token_length: int
    model_number: int

# Function to check anomaly with probability
def check_anomaly(request_interval, token_length, model_number):
    input_data = np.array([[request_interval, token_length, model_number]])
    input_scaled = scaler.transform(input_data)
    prediction = rf_model.predict(input_scaled)
    anomaly_prob = rf_model.predict_proba(input_scaled)[:, 1][0] * 100  # Probability of being an anomaly
    result = "Anomalous" if prediction[0] == 1 else "Not Anomalous"
    return result, anomaly_prob

# Define API endpoint
@app.post("/predict/")
async def predict_anomaly(data: AnomalyRequest):
    try:
        result, probability = check_anomaly(data.request_interval, data.token_length, data.model_number)
        return {"result": result, "anomaly_percentage": f"{probability:.2f}%"}
    except Exception as e:
        return {"error": str(e)}

# Define a root endpoint to check if API is running
@app.get("/")
def home():
    return {"message": "FastAPI is running! Use the /predict endpoint to check anomalies."}