# Predictive Maintenance — Engine Condition

End-to-end ML project:
- Train a scikit-learn pipeline on `data/engine_data.csv`
- Save model to `models/engine_model.joblib`
- CLI batch prediction (`src/infer.py`)
- Streamlit UI (`app/streamlit_app.py`)

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Put your dataset here:
# data/engine_data.csv

# Train & evaluate
python src/train.py --csv data/engine_data.csv

# Predict on a CSV (same columns as training features)
python src/infer.py --csv data/engine_data.csv --out predictions.csv

# Run UI
streamlit run app/streamlit_app.py
