from fastapi import FastAPI, HTTPException
import xgboost as xgb
import joblib
import os
import re
from pydantic import BaseModel
from typing import List
import numpy as np

app = FastAPI()

MODEL_DIR = "models"
model = None
scaler = None
threshold = 0.5
input_dim = None


def get_latest_model():
    if not os.path.exists(MODEL_DIR):
        raise Exception("Models directory not found.")

    files = os.listdir(MODEL_DIR)
    versions = []

    for f in files:
        match = re.search(r"model_v(\d+)\.json", f)
        if match:
            versions.append(int(match.group(1)))

    if not versions:
        raise Exception("No versioned XGBoost models found.")

    latest_version = max(versions)
    return f"{MODEL_DIR}/model_v{latest_version}.json"


@app.on_event("startup")
def load_model():
    global model, scaler, threshold, input_dim

    model_path = get_latest_model()

    model = xgb.XGBClassifier()
    model.load_model(model_path)

    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

    with open(os.path.join(MODEL_DIR, "threshold.txt"), "r") as f:
        threshold = float(f.read().strip())

    input_dim = scaler.n_features_in_

    print("Model loaded:", model_path)
    print("Threshold:", threshold)
    print("Input dimension:", input_dim)


@app.get("/")
def health():
    return {"status": "Fraud XGBoost API running"}


class FeatureInput(BaseModel):
    features: List[float]


@app.post("/predict")
def predict(data: FeatureInput):

    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    features = data.features

    if len(features) != input_dim:
        raise HTTPException(
            status_code=400,
            detail=f"Feature length mismatch. Expected {input_dim}, got {len(features)}"
        )

    features_array = np.array(features).reshape(1, -1)
    scaled = scaler.transform(features_array)

    prob = model.predict_proba(scaled)[0][1]
    prediction = 1 if prob > threshold else 0

    return {
        "fraud_probability": float(prob),
        "prediction": prediction
    }