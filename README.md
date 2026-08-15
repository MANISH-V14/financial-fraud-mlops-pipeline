# Financial Fraud Detection - Production MLOps Pipeline

A production-style fraud classification project focused on extreme class imbalance, model comparison, decision-threshold optimization, experiment tracking, and API deployment.

## Live API

**API:** https://financial-fraud-mlops-pipeline.onrender.com/

**Swagger docs:** https://financial-fraud-mlops-pipeline.onrender.com/docs

## Problem Statement

The dataset contains highly imbalanced credit-card transaction data, with an approximate 258:1 ratio of non-fraud to fraud observations. In this setting, overall accuracy can hide poor fraud detection performance.

The main objective is therefore to balance:

- **Recall:** catch a useful share of fraudulent transactions
- **Precision:** limit unnecessary false-positive alerts

## Model Comparison

Two modeling approaches were evaluated.

### 1. Neural Network

- PyTorch implementation
- `BCEWithLogitsLoss` with class weighting
- Threshold tuning using the precision-recall curve
- ROC-AUC of approximately 0.82

### 2. XGBoost - Production Model

- `scale_pos_weight` for imbalance handling
- Precision-recall threshold optimization
- MLflow experiment tracking
- Versioned model artifacts

### Reported XGBoost Performance

| Metric | Score |
| --- | ---: |
| Precision | 0.79 |
| Recall | 0.61 |
| F1 Score | 0.69 |
| ROC-AUC | 0.993 |

For this tabular dataset, XGBoost produced stronger reported results than the neural-network baseline and was selected for deployment.

## Key Engineering Decisions

- Used `scale_pos_weight` rather than synthetic oversampling in the production model
- Tuned the classification threshold using the precision-recall tradeoff
- Prioritized precision, recall, F1, and ROC-AUC over raw accuracy
- Benchmarked deep learning against gradient boosting
- Versioned production model artifacts

## Architecture

### Training Pipeline

`Preprocessing → Feature Scaling → Imbalance Handling → Model Training → Threshold Optimization → MLflow Tracking → Model Versioning`

### Deployment Pipeline

`Versioned Model → FastAPI → Docker → CI/CD → Render`

## API Usage

### Health Check

`GET /`

Example response:

```json
{
  "status": "Fraud XGBoost API running"
}
```

### Prediction

`POST /predict`

Example request:

```json
{
  "features": [0.12, -1.45, 2.33, 0.77, -0.56, 1.12, 0.44, -0.91, 0.35]
}
```

The endpoint returns a fraud probability and binary prediction.

## Project Structure

```text
app/
  main.py               # FastAPI deployment layer
src/
  train.py              # Neural-network training
  train_xgb.py          # XGBoost production training
  preprocess.py         # Data processing
  versioning.py         # Model version management
models/
  model_vX.json         # Versioned XGBoost models
  scaler.pkl
  threshold.txt
Dockerfile
requirements.txt
```

## Technologies

- Python
- XGBoost
- PyTorch
- Scikit-learn
- MLflow
- FastAPI
- Docker
- Render

## What This Project Demonstrates

- Handling severe class imbalance
- Comparing model families instead of assuming one algorithm is best
- Optimizing classification thresholds for business tradeoffs
- Experiment tracking with MLflow
- Version-controlled model artifacts
- Containerized model serving through FastAPI
- Cloud deployment workflow

## Future Improvements

- Kafka-based streaming fraud detection
- Data and prediction drift monitoring
- Feature-store integration
- SHAP-based explainability
- Automated retraining workflow
