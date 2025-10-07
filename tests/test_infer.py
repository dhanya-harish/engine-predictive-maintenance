from pathlib import Path
import joblib
import numpy as np
from src.infer import load_model

def test_model_load():
    # this only checks for file existence if present
    p = Path("models/engine_model.joblib")
    if p.exists():
        m = load_model(p)
        assert m is not None
