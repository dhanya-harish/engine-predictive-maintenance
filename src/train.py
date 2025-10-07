import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
import joblib

from utils import find_target_column

def train(csv_path: Path, out_dir: Path) -> None:
    df = pd.read_csv(csv_path)
    target_col = find_target_column(df, "engine_condition")
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])

    # numeric-only features (simple & robust)
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    X = X[num_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pre = ColumnTransformer([("num", StandardScaler(), num_cols)], remainder="drop")
    clf = GradientBoostingClassifier(random_state=42)

    pipe = Pipeline([("preprocessor", pre), ("classifier", clf)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "features": num_cols,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "engine_model.joblib"
    joblib.dump(pipe, model_path)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print("Saved:", model_path)
    print("Metrics:", json.dumps(metrics, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True, help="Path to training CSV (e.g., data/engine_data.csv)")
    ap.add_argument("--out", type=Path, default=Path("models"), help="Output dir for model & metrics")
    args = ap.parse_args()
    train(args.csv, args.out)
