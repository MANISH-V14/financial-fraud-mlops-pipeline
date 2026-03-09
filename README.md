-Financial Fraud Detection – Production MLOps Pipeline

Live API: https://financial-fraud-mlops-pipeline.onrender.com/

API Docs (Swagger): https://financial-fraud-mlops-pipeline.onrender.com/docs

Overview

*This project implements a production-grade fraud detection pipeline using machine learning and modern MLOps practices.

The dataset contains highly imbalanced credit card transaction data (≈0.4% fraud rate). The objective was to design a robust fraud classification system, compare modeling approaches, optimize decision thresholds, and deploy the best-performing model as a cloud-based API.

*Problem Statement

Fraud detection presents two major challenges:

Extreme class imbalance (≈258:1 non-fraud to fraud ratio)

Precision vs Recall tradeoff in financial systems

Accuracy is not a meaningful metric in this scenario. The system must balance:

High recall (catch fraudulent transactions)

Acceptable precision (minimize false alerts)

8Model Comparison

Two modeling approaches were evaluated:

1. Deep Learning (Neural Network)

BCEWithLogitsLoss with class weighting

Threshold tuning using precision-recall curve

ROC-AUC ≈ 0.82

2. XGBoost (Final Production Model)

scale_pos_weight for imbalance handling

Precision-Recall threshold optimization

MLflow experiment tracking

Versioned model artifacts

Final Model Performance (XGBoost)

Precision: 0.79

Recall: 0.61

F1 Score: 0.69

ROC-AUC: 0.993

XGBoost significantly outperformed the neural network for tabular fraud detection data.

*Key Engineering Decisions

Used scale_pos_weight instead of SMOTE to avoid synthetic noise

Optimized decision threshold using F1 from precision-recall curve

Avoided accuracy as a metric due to imbalance

Compared deep learning vs gradient boosting

Selected tree-based model for production deployment

*Architecture

Training Pipeline:

Data preprocessing

Feature scaling

Imbalance handling

Threshold optimization

MLflow experiment tracking

Model versioning

Deployment Pipeline:

FastAPI REST API

Docker containerization

CI/CD via GitHub

Cloud deployment on Render

*API Usage
Health Check

GET /

Response:

{
  "status": "Fraud XGBoost API running"
}
*Prediction

POST /predict

Request:

{
  "features": [ ... 9 numerical values ... ]
}

Response:

{
  "fraud_probability": 0.87,
  "prediction": 1
}
*Project Structure
app/
  main.py               # FastAPI deployment layer
src/
  train.py              # Neural network training
  train_xgb.py          # XGBoost training (production model)
  preprocess.py         # Data processing
  versioning.py         # Model version management
models/
  model_vX.json         # Versioned XGBoost models
  scaler.pkl
  threshold.txt
Dockerfile
requirements.txt
*Technologies Used

Python

XGBoost

PyTorch (for comparison)

scikit-learn

MLflow

FastAPI

Docker

Render (Cloud Deployment)

*What This Project Demonstrates

Handling extreme class imbalance

Model benchmarking and selection

Threshold optimization for real-world tradeoffs

Experiment tracking with MLflow

Version-controlled model artifacts

Containerized ML API

CI/CD deployment workflow

*Future Improvements

Real-time streaming fraud detection (Kafka)

Drift detection and monitoring

Feature store integration

SHAP-based explainability service

Automated retraining pipeline
