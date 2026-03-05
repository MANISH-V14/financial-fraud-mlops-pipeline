import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
import os
import numpy as np
from preprocess import load_data, scale_data
from model import FraudNet
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from versioning import get_next_version

def train():

    X_train, X_test, y_train, y_test = load_data()
    X_train, X_test, scaler = scale_data(X_train, X_test)

    import joblib
    joblib.dump(scaler, "models/scaler.pkl")

    device = torch.device("cpu")

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1,1)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1,1)

    model = FraudNet(X_train.shape[1]).to(device)
    input_dim = X_train.shape[1]

    # Handle class imbalance
    fraud_count = y_train.sum()
    non_fraud_count = len(y_train) - fraud_count

    pos_weight = torch.tensor(non_fraud_count / fraud_count)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 15

    mlflow.set_experiment("fraud_detection_dl")

    with mlflow.start_run():

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            probs = torch.sigmoid(test_outputs)
            from sklearn.metrics import precision_recall_curve

            # Get precision-recall curve
            precisions, recalls, thresholds = precision_recall_curve(y_test, probs.numpy())

            # Compute F1 for each threshold
            f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)

            best_idx = f1_scores.argmax()
            best_threshold = thresholds[best_idx]

            print("Best Threshold:", best_threshold)
            preds = (probs > best_threshold).float()

        precision = precision_score(y_test, preds.numpy())
        recall = recall_score(y_test, preds.numpy())
        f1 = f1_score(y_test, preds.numpy())
        auc = roc_auc_score(y_test, probs.numpy())

        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", auc)

        if not os.path.exists("models"):
            os.makedirs("models")

        version = get_next_version()
        model_path = f"models/model_v{version}.pt"
        torch.save(model.state_dict(), model_path)
        # Save input dimension metadata
        with open("models/input_dim.txt", "w") as f:
            f.write(str(input_dim))

        print("Model saved as:", model_path)
        mlflow.pytorch.log_model(model, "model")

        print("Precision:", precision)
        print("Recall:", recall)
        print("F1:", f1)
        print("ROC-AUC:", auc)


if __name__ == "__main__":
    train()