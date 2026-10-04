import gzip
import tempfile
import streamlit as st
import xgboost as xgb

@st.cache_resource
def load_model():
    # Read the compressed .gz model file
    with gzip.open("xgb_model.json.gz", "rb") as f_in:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_file:
            tmp_file.write(f_in.read())
            tmp_path = tmp_file.name

    model = xgb.XGBClassifier()
    model.load_model(tmp_path)
    return model

model = load_model()
