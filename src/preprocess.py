import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_data(path="data/creditcard.csv"):
    df = pd.read_csv(path)

    # Keep only numeric features for deep learning
    df = df.select_dtypes(include=[np.number])

    # Drop transaction identifiers
    drop_cols = ["cc_num"]
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    X = df.drop("is_fraud", axis=1)
    y = df["is_fraud"]

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def scale_data(X_train, X_test):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler