#!/usr/bin/env python3
"""
churn_pipeline.py

Run: python churn_pipeline.py
Optional: python churn_pipeline.py --data data.csv --outmodel model.joblib --test-size 0.2

This script:
 - Loads data.csv (Kaggle Telco Customer Churn-style)
 - Cleans basic issues (TotalCharges), encodes categorical features
 - Trains LogisticRegression, RandomForest, GradientBoosting
 - Builds a soft VotingClassifier (GBT + LR + AdaBoost)
 - Prints classification reports & cross-val accuracy
 - Saves model to disk and confusion matrix image
"""

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# -------- helper functions --------
def load_data(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path.resolve()}")
    df = pd.read_csv(path)
    print(f"Loaded data: {df.shape[0]:,} rows, {df.shape[1]:,} columns")
    return df

def basic_cleaning(df: pd.DataFrame):
    # Drop customerID if present
    df = df.copy()
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Convert TotalCharges to numeric (some datasets have spaces)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        # Impute missing by 0 or median; we'll use median
        missing_before = df["TotalCharges"].isna().sum()
        if missing_before > 0:
            median = df["TotalCharges"].median()
            df["TotalCharges"] = df["TotalCharges"].fillna(median)
            print(f"Filled {missing_before} missing TotalCharges with median={median:.2f}")

    # Trim whitespace in object columns
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].str.strip()

    return df

def encode_features(df: pd.DataFrame):
    df = df.copy()
    # Target: 'Churn' -> 0/1
    if "Churn" not in df.columns:
        raise KeyError("Expected 'Churn' column in dataset")
    y = (df["Churn"].map({"Yes": 1, "No": 0}) if df["Churn"].dtype == object else df["Churn"]).astype(int)
    X = df.drop(columns=["Churn"])

    # Numerical columns
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # Categorical columns
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # For binary-like categorical columns with Yes/No -> map to 1/0
    bin_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
    for c in list(cat_cols):  # iterate over copy
        unique_vals = X[c].dropna().unique()
        # if only two unique values and they are Yes/No or similar, map them
        if set(map(str.lower, map(str, unique_vals))).issubset({"yes", "no"}) or set(unique_vals) <= {"Yes", "No"}:
            X[c] = X[c].map(bin_map).astype(float)
            cat_cols.remove(c)
            num_cols = num_cols + [c]
        elif len(unique_vals) == 2 and set(unique_vals) <= {"Male", "Female"}:
            X[c] = X[c].map(bin_map).astype(float)
            cat_cols.remove(c)
            num_cols = num_cols + [c]

    # One-hot encode remaining categorical columns
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # Fill any remaining NaNs with 0
    X = X.fillna(0)

    return X, y

def train_and_evaluate(X, y, test_size=0.2, random_state=42, out_model_path="churn_model.joblib", plot_path="confusion_matrix.png"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

    # Scale numerical features (fit on train only)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Models
    lr = LogisticRegression(max_iter=1000, random_state=random_state)
    rf = RandomForestClassifier(n_estimators=200, random_state=random_state)
    gbc = GradientBoostingClassifier(n_estimators=200, random_state=random_state)
    abc = AdaBoostClassifier(n_estimators=100, random_state=random_state)

    # Fit baseline models and show CV scores
    models = {
        "LogisticRegression": lr,
        "RandomForest": rf,
        "GradientBoosting": gbc,
        "AdaBoost": abc,
    }

    print("\nCross-validation (5-fold) mean accuracy for base models:")
    for name, model in models.items():
        cv = cross_val_score(model, X_train_scaled, y_train, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state), scoring="accuracy")
        print(f"  {name:18s}: {cv.mean():.4f} ± {cv.std():.4f}")

    # Voting classifier - soft voting combining GBC, LR, AdaBoost (like original notebook)
    voting = VotingClassifier(estimators=[("gbc", gbc), ("lr", lr), ("abc", abc)], voting="soft")
    voting.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = voting.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\nVotingClassifier Test performance:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-score : {f1:.4f}\n")

    print("Classification report (test):\n")
    print(classification_report(y_test, y_pred, digits=4))

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Voting Classifier")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved confusion matrix to {plot_path}")

    # Save scaler + model pipeline as joblib: save dict
    joblib.dump({"model": voting, "scaler": scaler, "feature_columns": X.columns.tolist()}, out_model_path)
    print(f"Saved model+scaler to {out_model_path}")

    return voting

# -------- main CLI --------
def main(args):
    data_path = Path(args.data)
    out_model = Path(args.outmodel)
    plot_path = Path(args.plot)

    df = load_data(data_path)
    df = basic_cleaning(df)
    X, y = encode_features(df)
    model = train_and_evaluate(X, y, test_size=args.test_size, random_state=args.random_state, out_model_path=str(out_model), plot_path=str(plot_path))

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train a churn prediction pipeline (single-file).")
    p.add_argument("--data", type=str, default="data.csv", help="path to data CSV (default: data.csv)")
    p.add_argument("--outmodel", type=str, default="churn_model.joblib", help="output model path")
    p.add_argument("--plot", type=str, default="confusion_matrix.png", help="confusion matrix image output")
    p.add_argument("--test-size", type=float, default=0.2, help="test size fraction")
    p.add_argument("--random-state", type=int, default=42, help="random seed")
    args = p.parse_args()
    main(args)
