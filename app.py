import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Lung Cancer Risk Predictor",
    page_icon="🫁",
    layout="centered"
)

st.title("🫁 Lung Cancer Risk Prediction Dashboard")
st.write("Enter the patient's survey details below to evaluate lung cancer risk matching your XGBoost classifier model.")

# --- MODEL & SCALER LOADING ---
@st.cache_resource
def load_artifacts():
    # Load model and scaler saved via joblib
    model = joblib.load('XGBoos_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_artifacts()
except Exception as e:
    st.error(f"⚠️ Error loading model artifacts: {e}")
    st.info("Make sure 'XGBoos_model.pkl' and 'scaler.pkl' are placed in the same directory as this file.")
    st.stop()

# --- INPUT FORM ---
st.subheader("📋 Patient Diagnostic Survey Data")

with st.form("prediction_form"):
    # Grouping Demographics
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", options=["Male", "Female"])
    with col2:
        age = st.number_input("Age (Years)", min_value=1, max_value=120, value=50, step=1)
        
    st.write("---")
    st.write("**Symptoms, Medical History, & Lifestyle Risks**")
    
    # 2-Column form layout for categorical survey responses
    col3, col4 = st.columns(2)
    
    with col3:
        smoking = st.selectbox("Smoking", ["No", "Yes"])
        yellow_fingers = st.selectbox("Yellow Fingers", ["No", "Yes"])
        anxiety = st.selectbox("Anxiety", ["No", "Yes"])
        peer_pressure = st.selectbox("Peer Pressure", ["No", "Yes"])
        chronic_disease = st.selectbox("Chronic Disease", ["No", "Yes"])
        fatigue = st.selectbox("Fatigue/Weakness", ["No", "Yes"])
        allergy = st.selectbox("Allergies", ["No", "Yes"])
        
    with col4:
        wheezing = st.selectbox("Wheezing", ["No", "Yes"])
        alcohol = st.selectbox("Alcohol Consuming", ["No", "Yes"])
        coughing = st.selectbox("Frequent Coughing", ["No", "Yes"])
        shortness_of_breath = st.selectbox("Shortness of Breath", ["No", "Yes"])
        swallowing_diff = st.selectbox("Swallowing Difficulty", ["No", "Yes"])
        chest_pain = st.selectbox("Chest Pain", ["No", "Yes"])

    # Submit Button
    submit_btn = st.form_submit_button("Generate Risk Assessment")

# --- INFERENCE PIPELINE ---
# --- INFERENCE PIPELINE ---
if submit_btn:
    # 1. Map values directly matching the notebook training transformations (Binary 0/1 scale)
    input_dict = {
        'GENDER': 1 if gender == "Male" else 0,
        'AGE': age,
        'SMOKING': 1 if smoking == "Yes" else 0,
        'YELLOW_FINGERS': 1 if yellow_fingers == "Yes" else 0,
        'ANXIETY': 1 if anxiety == "Yes" else 0,
        'PEER_PRESSURE': 1 if peer_pressure == "Yes" else 0,
        'CHRONIC DISEASE': 1 if chronic_disease == "Yes" else 0,
        'FATIGUE': 1 if fatigue == "Yes" else 0, # <-- REMOVED THE TRAILING SPACE HERE
        'ALLERGY': 1 if allergy == "Yes" else 0,
        'WHEEZING': 1 if wheezing == "Yes" else 0,
        'ALCOHOL CONSUMING': 1 if alcohol == "Yes" else 0,
        'COUGHING': 1 if coughing == "Yes" else 0,
        'SHORTNESS OF BREATH': 1 if shortness_of_breath == "Yes" else 0,
        'SWALLOWING DIFFICULTY': 1 if swallowing_diff == "Yes" else 0,
        'CHEST PAIN': 1 if chest_pain == "Yes" else 0
    }
    
    # Convert into standard pandas DataFrame to retain clean feature naming/order context
    input_df = pd.DataFrame([input_dict])
    
    # 2. Scale the features using your loaded scaler object
    scaled_features = scaler.transform(input_df)
    
    # 3. Predict probability and final class binary outcome
    prediction = model.predict(scaled_features)[0]
    prediction_proba = model.predict_proba(scaled_features)[0]
    
    # --- DISPLAY SCREENING RESULTS ---
    st.write("---")
    st.subheader("🎯 Model Prediction Results")
    
    risk_percentage = prediction_proba[1] * 100
    
    if prediction == 1:
        st.error(f"⚠️ **High Risk Group** (Probability match: {risk_percentage:.2f}%)")
    else:
        st.success(f"✅ **Low Risk Group** (Probability match: {risk_percentage:.2f}%)")