import os, joblib, numpy as np, pandas as pd, streamlit as st
from huggingface_hub import hf_hub_download

LOCAL_MODEL = os.getenv("LOCAL_MODEL", "models/engine_model.joblib")
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "")     # optional fallback
HF_MODEL_FILE = os.getenv("HF_MODEL_FILE", "model/engine_model.joblib")
HF_TOKEN      = os.getenv("HF_TOKEN")              # only if private

@st.cache_resource(show_spinner=True)
def load_model():
    # prefer local artifact, else pull from HF model repo if provided
    if os.path.exists(LOCAL_MODEL):
        return joblib.load(LOCAL_MODEL)
    if HF_MODEL_REPO:
        cache_dir = os.getenv("HF_HUB_CACHE", "/tmp/huggingface/hub")
        os.makedirs(cache_dir, exist_ok=True)
        p = hf_hub_download(
            repo_id=HF_MODEL_REPO, filename=HF_MODEL_FILE,
            repo_type="model", token=HF_TOKEN, cache_dir=cache_dir
        )
        return joblib.load(p)
    raise FileNotFoundError("No local model and no HF model repo configured.")

def get_expected_input_columns(clf):
    pre = getattr(getattr(clf, "named_steps", {}), "get", lambda *_: None)("preprocessor")
    if pre is not None:
        transformers = getattr(pre, "transformers_", getattr(pre, "transformers", []))
        cols = []
        for _, __, selected in transformers:
            if selected in (None, "drop"): continue
            if isinstance(selected, list): cols.extend(selected)
            elif hasattr(selected, "__iter__"): cols.extend(list(selected))
        if cols: return list(dict.fromkeys(cols))
    fni = getattr(clf, "feature_names_in_", None)
    return list(fni) if fni is not None else [
        "engine_rpm","lub_oil_pressure","fuel_pressure",
        "coolant_pressure","lub_oil_temp","coolant_temp"
    ]

def coerce_numeric_df(df):
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="ignore")
    return out

st.set_page_config(page_title="Engine Condition Predictor", layout="centered")
st.title("Predictive Maintenance — Engine Condition")

model = load_model()
EXPECTED = get_expected_input_columns(model)

with st.form("predict_form"):
    col1, col2 = st.columns(2)
    with col1:
        engine_rpm       = st.number_input("Engine RPM", min_value=0, max_value=5000, value=1200, step=10)
        lub_oil_pressure = st.number_input("Lubricating Oil Pressure (bar)", value=3.0, step=0.1)
        fuel_pressure    = st.number_input("Fuel Pressure (bar)", value=5.0, step=0.1)
    with col2:
        coolant_pressure = st.number_input("Coolant Pressure (bar)", value=2.0, step=0.1)
        lub_oil_temp     = st.number_input("Lubricating Oil Temperature (°C)", value=80.0, step=0.1)
        coolant_temp     = st.number_input("Coolant Temperature (°C)", value=75.0, step=0.1)
    sub = st.form_submit_button("Predict")

if sub:
    row = pd.DataFrame({c: [np.nan] for c in EXPECTED})
    for k, v in {
        "engine_rpm": engine_rpm,
        "lub_oil_pressure": lub_oil_pressure,
        "fuel_pressure": fuel_pressure,
        "coolant_pressure": coolant_pressure,
        "lub_oil_temp": lub_oil_temp,
        "coolant_temp": coolant_temp,
    }.items():
        if k in row.columns:
            row.at[0, k] = v

    X = coerce_numeric_df(row)
    y = model.predict(X)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = float(model.predict_proba(X)[0, 1])
        except Exception:
            pass

    if y == 1:
        st.error("⚠️ Faulty engine" + (f" (confidence {proba:.2f})" if proba is not None else ""))
    else:
        st.success("✅ Healthy engine" + (f" (confidence {1-proba:.2f})" if proba is not None else ""))

    with st.expander("Inputs sent to model"):
        st.dataframe(X)
