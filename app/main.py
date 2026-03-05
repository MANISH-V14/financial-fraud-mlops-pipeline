from fastapi import FastAPI
import torch
import os
import re
import joblib
from src.model import FraudNet

# -----------------------------
# Initialize FastAPI FIRST
# -----------------------------
app = FastAPI()

MODEL_DIR = "models"
model = None
scaler = None


def get_latest_model():
    files = os.listdir(MODEL_DIR)
    versions = []

    for f in files:
        match = re.search(r"model_v(\d+)\.pt", f)
        if match:
            versions.append(int(match.group(1)))

    if not versions:
        raise Exception("No model versions found.")

    latest_version = max(versions)
    return f"{MODEL_DIR}/model_v{latest_version}.pt"


# -----------------------------
# Load model + scaler at startup
# -----------------------------
@app.on_event("startup")
def load_model():
    global model
    global scaler

    model_path = get_latest_model()

    # Load input dimension
    with open("models/input_dim.txt", "r") as f:
        input_dim = int(f.read().strip())

    # Initialize model
    model_instance = FraudNet(input_dim)
    model_instance.load_state_dict(
        torch.load(model_path, map_location="cpu")
    )
    model_instance.eval()

    model = model_instance

    # Load scaler
    scaler = joblib.load("models/scaler.pkl")


# -----------------------------
# Health Check
# -----------------------------
@app.get("/")
def health():
    return {"status": "Fraud MLOps API running"}


# -----------------------------
# Prediction Endpoint
# -----------------------------
from pydantic import BaseModel
from typing import List

class FeatureInput(BaseModel):
    features: List[float]


@app.post("/predict")
def predict(data: FeatureInput):
    global model, scaler

    features = data.features

    if len(features) != int(open("models/input_dim.txt").read().strip()):
        return {"error": "Feature length mismatch"}

    scaled = scaler.transform([features])
    x = torch.tensor(scaled, dtype=torch.float32)

    with torch.no_grad():
        output = model(x)
        prob = torch.sigmoid(output).item()

    return {
        "fraud_probability": prob,
        "prediction": 1 if prob > 0.6 else 0
    }