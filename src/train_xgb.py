import os
import numpy as np
import joblib
import mlflow
import mlflow.xgboost
import xgboost as xgb

from preprocess import load_data, scale_data
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import precision_recall_curve
from versioning import get_next_version


def train_xgb():

    X_train, X_test, y_train, y_test = load_data()

    # XGBoost does NOT require scaling, but we keep consistent pipeline
    X_train, X_test, scaler = scale_data(X_train, X_test)

    if not os.path.exists("models"):
        os.makedirs("models")

    joblib.dump(scaler, "models/scaler.pkl")

    fraud_count = y_train.sum()
    non_fraud_count = len(y_train) - fraud_count

    scale_pos_weight = non_fraud_count / fraud_count

    print("Fraud count:", fraud_count)
    print("Non-fraud count:", non_fraud_count)
    print("Scale pos weight:", scale_pos_weight)

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        use_label_encoder=False
    )

    mlflow.set_experiment("fraud_detection_xgb")

    with mlflow.start_run():

        mlflow.log_param("n_estimators", 300)
        mlflow.log_param("max_depth", 6)
        mlflow.log_param("learning_rate", 0.05)
        mlflow.log_param("scale_pos_weight", float(scale_pos_weight))

        model.fit(X_train, y_train)

        probs = model.predict_proba(X_test)[:, 1]

        precisions, recalls, thresholds = precision_recall_curve(y_test, probs)
        f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)

        best_idx = np.argmax(f1_scores)

        if best_idx >= len(thresholds):
            best_threshold = 0.5
        else:
            best_threshold = thresholds[best_idx]

        preds = (probs > best_threshold).astype(int)

        precision = precision_score(y_test, preds)
        recall = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)

        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("best_threshold", float(best_threshold))

        version = get_next_version()
        model_path = f"models/model_v{version}.json"
        model.save_model(model_path)

        with open("models/threshold.txt", "w") as f:
            f.write(str(best_threshold))

        print("Model saved as:", model_path)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1:", f1)
        print("ROC-AUC:", auc)

        mlflow.xgboost.log_model(model, "model")


if __name__ == "__main__":
    train_xgb()