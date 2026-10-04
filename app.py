import streamlit as st
import xgboost as xgb
import numpy as np

st.title("My XGBoost Model")

# Load your model
@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model("xgb_model.json")
    return model

model = load_model()

# Input field for user
st.write("Enter your inputs below:")
input_data = st.number_input("Feature 1", value=0.0)

if st.button("Predict"):
    prediction = model.predict(np.array([[input_data]]))
    st.write("Prediction:", prediction[0])
