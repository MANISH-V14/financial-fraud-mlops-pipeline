import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
import os
import numpy as np
import random
import joblib

from preprocess import load_data, scale_data
from model import FraudNet
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import precision_recall_curve
from versioning import get_next_version


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train():

    set_seed(42)

    # -----------------------------
    # Load + Scale Data
    # -----------------------------
    X_train, X_test, y_train, y_test = load_data()
    X_train, X_test, scaler = scale_data(X_train, X_test)

    if not os.path.exists("models"):
        os.makedirs("models")

    joblib.dump(scaler, "models/scaler.pkl")

    device = torch.device("cpu")

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1).to(device)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1).to(device)

    input_dim = X_train.shape[1]
    model = FraudNet(input_dim).to(device)

    # -----------------------------
    # Handle Class Imbalance
    # -----------------------------
    fraud_count = y_train.sum()
    non_fraud_count = len(y_train) - fraud_count

    pos_weight_value = 100
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)

    print("Fraud count:", fraud_count)
    print("Non-fraud count:", non_fraud_count)
    print("Positive class weight:", pos_weight_value)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 20

    # -----------------------------
    # MLflow Setup
    # -----------------------------
    mlflow.set_experiment("fraud_detection_dl")

    with mlflow.start_run():

        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", 0.001)
        mlflow.log_param("pos_weight", float(pos_weight_value))

        # -----------------------------
        # Training Loop
        # -----------------------------
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

        # -----------------------------
        # Evaluation
        # -----------------------------
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            probs = torch.sigmoid(test_outputs).cpu().numpy()

        precisions, recalls, thresholds = precision_recall_curve(y_test, probs)

        # Avoid division by zero
        f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)

        best_idx = np.argmax(f1_scores)

        # Edge case handling
        if best_idx >= len(thresholds):
            best_threshold = 0.5
        else:
            best_threshold = thresholds[best_idx]

        print("Best Threshold:", best_threshold)

        preds = (probs > best_threshold).astype(int)

        precision = precision_score(y_test, preds)
        recall = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)

        # -----------------------------
        # Log Metrics
        # -----------------------------
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("best_threshold", float(best_threshold))

        # -----------------------------
        # Save Model Version
        # -----------------------------
        version = get_next_version()
        model_path = f"models/model_v{version}.pt"
        torch.save(model.state_dict(), model_path)

        # Save metadata
        with open("models/input_dim.txt", "w") as f:
            f.write(str(input_dim))

        with open("models/threshold.txt", "w") as f:
            f.write(str(best_threshold))

        print("Model saved as:", model_path)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1:", f1)
        print("ROC-AUC:", auc)

        mlflow.pytorch.log_model(model, "model")


if __name__ == "__main__":
    train()