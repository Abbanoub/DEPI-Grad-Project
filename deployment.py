import streamlit as st
import pandas as pd
import pickle
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Stroke Prediction System",
    page_icon="🏥",
    layout="wide"
)

# --- STYLE CUSTOMIZATION ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODEL ---
# Using relative path to avoid errors on different machines
MODEL_PATH = "stroke_prediction_model.pkl"

@st.cache_resource
def load_model(path):
    # Check if the model file exists before trying to open it
    if not os.path.exists(path):
        st.error(f"Model file not found at {path}. Please ensure the .pkl file is in the same directory.")
        return None
    with open(path, 'rb') as file:
        return pickle.load(file)

model = load_model(MODEL_PATH)

# --- HELPER FUNCTIONS ---
def preprocess_input(value, category):
    """
    Handles categorical to numerical conversion based on the training logic.
    """
    mapping = {
        'gender': {'Male': 1, 'Female': 0},
        'ever_married': {'Yes': 1, 'No': 0},
        'work_type': {'Govt_job': 0, 'Never_worked': 1, 'Private': 2, 'Self-employed': 3, 'children': 4},
        'Residence_type': {'Urban': 0, 'Rural': 1},
        'smoking_status': {'smokes': 1, 'formerly smoked': 1, 'never smoked': 0, 'unknown': 0} 
    }
    return mapping.get(category, {}).get(value, 0)

# --- UI HEADER ---
st.title("🏥 Stroke Prediction Health Assistant")
st.write("Please fill in the patient data below to predict the likelihood of a stroke.")
st.divider()

# --- INPUT FORM ---
if model:
    # Organize inputs into two columns for a professional look
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("👤 Personal Information")
        gender_raw = st.selectbox("Gender", options=["Male", "Female"])
        age = st.slider("Age", 0, 100, 45)
        ever_married_raw = st.selectbox("Ever Married", options=["Yes", "No"])
        work_type_raw = st.selectbox("Work Type", options=["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
        residence_raw = st.selectbox("Residence Type", options=["Urban", "Rural"])

    with col2:
        st.subheader("🩺 Medical Data")
        hypertension = st.radio("Do you have Hypertension?", options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        heart_disease = st.radio("Do you have Heart Disease?", options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, value=100.0)
        bmi = st.number_input("Body Mass Index (BMI)", min_value=0.0, value=25.0)
        smoking_raw = st.selectbox("Smoking Status", options=["never smoked", "formerly smoked", "smokes", "unknown"])

    st.markdown("---")

    # --- PREDICTION LOGIC ---
    if st.button("Analyze & Predict"):
        # Create a spinner for better UX
        with st.spinner('Calculating results...'):
            # Data Preprocessing
            data = {
                'gender': [preprocess_input(gender_raw, 'gender')],
                'age': [age],
                'hypertension': [hypertension],
                'heart_disease': [heart_disease],
                'ever_married': [preprocess_input(ever_married_raw, 'ever_married')],
                'work_type': [preprocess_input(work_type_raw, 'work_type')],
                'Residence_type': [preprocess_input(residence_raw, 'Residence_type')],
                'avg_glucose_level': [avg_glucose_level],
                'bmi': [bmi],
                'smoking_status': [preprocess_input(smoking_raw, 'smoking_status')],
            }
            
            input_df = pd.DataFrame(data)
            
            # Make prediction
            prediction = model.predict(input_df)
            prediction_proba = model.predict_proba(input_df)[0][1] # Get probability of stroke (class 1)

            # --- DISPLAY RESULTS ---
            st.subheader("Result Analysis")
            if prediction[0] == 1:
                st.error(f"⚠️ High Risk: The model predicts a high likelihood of stroke (Confidence: {prediction_proba:.2%})")
                st.warning("Recommendation: Please consult a doctor immediately for a full check-up.")
            else:
                st.success(f"✅ Low Risk: The model predicts a low likelihood of stroke (Confidence: {1 - prediction_proba:.2%})")
                st.info("Recommendation: Maintain a healthy lifestyle and keep monitoring your blood pressure.")
else:
    st.warning("Application is waiting for the model to load...")