from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import torch
import os
import re
import joblib

from src.model import FraudNet

# -----------------------------
# Initialize FastAPI
# -----------------------------
app = FastAPI(title="Financial Fraud Detection API")

MODEL_DIR = "models"

model = None
scaler = None
input_dim = None
threshold = None


# -----------------------------
# Utility: Get Latest Model
# -----------------------------
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
# Startup: Load Everything Once
# -----------------------------
@app.on_event("startup")
def load_artifacts():
    global model, scaler, input_dim, threshold

    model_path = get_latest_model()

    # Load input dimension
    with open(os.path.join(MODEL_DIR, "input_dim.txt"), "r") as f:
        input_dim = int(f.read().strip())

    # Load threshold (fallback if not present)
    threshold_path = os.path.join(MODEL_DIR, "threshold.txt")
    if os.path.exists(threshold_path):
        with open(threshold_path, "r") as f:
            threshold = float(f.read().strip())
    else:
        threshold = 0.6  # fallback

    # Initialize model
    model_instance = FraudNet(input_dim)
    model_instance.load_state_dict(
        torch.load(model_path, map_location="cpu")
    )
    model_instance.eval()

    model = model_instance

    # Load scaler
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))


# -----------------------------
# Health Endpoint
# -----------------------------
@app.get("/")
def health():
    return {
        "status": "Fraud MLOps API running",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }


# -----------------------------
# Metrics Endpoint
# -----------------------------
@app.get("/metrics")
def metrics():
    return {
        "input_dimension": input_dim,
        "decision_threshold": threshold
    }


# -----------------------------
# Request / Response Schemas
# -----------------------------
class FraudRequest(BaseModel):
    features: List[float]


class FraudResponse(BaseModel):
    fraud_probability: float
    prediction: int


# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/predict", response_model=FraudResponse)
def predict(request: FraudRequest):
    global model, scaler, input_dim, threshold

    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if len(request.features) != input_dim:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {input_dim} features, got {len(request.features)}"
        )

    scaled = scaler.transform([request.features])
    x = torch.tensor(scaled, dtype=torch.float32)

    with torch.no_grad():
        output = model(x)
        prob = torch.sigmoid(output).item()

    return FraudResponse(
        fraud_probability=round(prob, 4),
        prediction=int(prob > threshold)
    )