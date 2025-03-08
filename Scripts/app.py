import streamlit as st
import pandas as pd
import pickle
import os
from helper_functions import log_info, log_error

# Define paths
ARTIFACTS_PATH = "D:/MLOPS-2025-DSC/mlops2025-DSC/Artifacts"
DATA_OUTPUT_PATH = "D:/MLOPS-2025-DSC/mlops2025-DSC/Data/output"
os.makedirs(DATA_OUTPUT_PATH, exist_ok=True)
MODEL_PATH = os.path.join(ARTIFACTS_PATH, "best_classifier.pkl")
PIPELINE_PATH = os.path.join(ARTIFACTS_PATH, "data_processing_pipeline.pkl")
LABEL_ENCODER_PATH = os.path.join(ARTIFACTS_PATH, "label_encoder.pkl")

def load_artifact(filepath):
    """
    Load a saved pickle artifact.
    """
    try:
        with open(filepath, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        log_error(f"Artifact not found: {filepath}")
        st.error(f"Error: Artifact not found: {filepath}")
        return None

def predict_risk_category(input_data):
    """
    Predicts the loan risk category using the trained model.
    """
    pipeline = load_artifact(PIPELINE_PATH)
    model = load_artifact(MODEL_PATH)
    label_encoder = load_artifact(LABEL_ENCODER_PATH)
    
    if not pipeline or not model or not label_encoder:
        return None
    
    transformed_input = pipeline.transform(pd.DataFrame([input_data], columns=input_data.keys()))
    prediction = model.predict(transformed_input)
    return label_encoder.inverse_transform(prediction)[0]

# Streamlit UI
st.title("🏦 Loan Risk Categorization App")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["Single Prediction", "Batch Prediction"])

if page == "Single Prediction":
    st.header("Enter Customer Details")
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    income = st.number_input("Income", min_value=1000, max_value=100000, value=50000)
    employment_type = st.selectbox("Employment Type", ['Salaried', 'Unemployed', 'Self-employed'])
    residence_type = st.selectbox("Residence Type", ['Parental Home', 'Rented', 'Owned'])
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
    loan_amount = st.number_input("Loan Amount", min_value=1000, max_value=500000, value=10000)
    loan_term = st.number_input("Loan Term (Months)", min_value=6, max_value=360, value=60)
    previous_default = st.selectbox("Previous Default", ['Yes', 'No'])
    
    if st.button("Predict Risk Category"):
        input_data = {
            'Age': age,
            'Income': income,
            'EmploymentType': employment_type,
            'ResidenceType': residence_type,
            'CreditScore': credit_score,
            'LoanAmount': loan_amount,
            'LoanTerm': loan_term,
            'PreviousDefault': previous_default
        }
        prediction = predict_risk_category(input_data)
        if prediction:
            st.success(f"Predicted Risk Category: {prediction}")
            log_info(f"Predicted Risk Category: {prediction}")

elif page == "Batch Prediction":
    st.header("Batch Prediction")
    uploaded_file = st.file_uploader("Upload CSV for Batch Prediction", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        pipeline = load_artifact(PIPELINE_PATH)
        model = load_artifact(MODEL_PATH)
        label_encoder = load_artifact(LABEL_ENCODER_PATH)
        
        if pipeline and model and label_encoder:
            transformed_data = pipeline.transform(df)
            predictions = model.predict(transformed_data)
            df['Predicted Risk Category'] = label_encoder.inverse_transform(predictions)
            
            # Save the batch predictions to the output folder
            output_file = os.path.join(DATA_OUTPUT_PATH, "batch_predictions.csv")
            df.to_csv(output_file, index=False)
            
            st.write(df)
            st.success(f"Batch Prediction Completed! Results saved at {output_file}")
            log_info("Batch Prediction Completed Successfully!")
