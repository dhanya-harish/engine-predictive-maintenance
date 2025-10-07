import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

def load_model(model_path: Path):
    return joblib.load(model_path)

def predict_csv(model, csv_path: Path, out_path: Path):
    df = pd.read_csv(csv_path)
    # keep only model’s known features if available
    features = getattr(model, "feature_names_in_", None)
    if features is not None:
        df = df[[c for c in df.columns if c in features]]

    # numeric coercion
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="ignore")

    y = model.predict(df)
    proba = None
    if hasattr(model, "predict_proba"):
        try:
            P = model.predict_proba(df)
            proba = P[:, 1] if (P.ndim == 2 and P.shape[1] >= 2) else P.ravel()
        except Exception:
            pass

    out = df.copy()
    out["prediction"] = y
    if proba is not None:
        out["prob_faulty"] = proba
    out.to_csv(out_path, index=False)
    print("Wrote predictions to:", out_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=Path("models/engine_model.joblib"))
    ap.add_argument("--csv", type=Path, required=True, help="CSV to score")
    ap.add_argument("--out", type=Path, default=Path("predictions.csv"))
    args = ap.parse_args()

    model = load_model(args.model)
    predict_csv(model, args.csv, args.out)
