import os
import pandas as pd
from data_processing import create_data_pipeline, save_pipeline, encode_response_variable, split_data
from ml_functions import training_pipeline, prediction_pipeline, evaluation_matrices
from helper_functions import log_info, log_error

# Define paths
DATA_PATH = "D:/MLOPS-2025-DSC/mlops2025-DSC/Data/raw/loan_risk_data.csv"
ARTIFACTS_PATH = "D:/MLOPS-2025-DSC/mlops2025-DSC/Artifacts"
os.makedirs(ARTIFACTS_PATH, exist_ok=True)
PIPELINE_PATH = os.path.join(ARTIFACTS_PATH, "data_processing_pipeline.pkl")
LABEL_ENCODER_PATH = os.path.join(ARTIFACTS_PATH, "label_encoder.pkl")
MODEL_PATH = os.path.join(ARTIFACTS_PATH, "best_classifier.pkl")

def main():
    log_info("🚀 Starting Data Processing Step")
    
    # Load dataset
    df = pd.read_csv(DATA_PATH)
    log_info(f"✅ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # Separate features and target variable
    X = df.drop(["RiskCategory"], axis=1)
    y = df["RiskCategory"]

    # Encode target variable and save encoder
    y_encoded = encode_response_variable(y)
    log_info(f"✅ Target variable encoded and saved at {LABEL_ENCODER_PATH}")

    # Create and fit the data processing pipeline
    pipeline = create_data_pipeline(X)
    X_transformed = pipeline.fit_transform(X)
    save_pipeline(pipeline)
    log_info(f"✅ Data pipeline created and saved at {PIPELINE_PATH}")

    # Split the dataset
    X_train, X_val, y_train, y_val = split_data(pd.DataFrame(X_transformed), y_encoded)
    log_info("✅ Data split completed")

    # Train model and save it
    best_model = training_pipeline(X_train, y_train)
    log_info(f"✅ Model trained and saved at {MODEL_PATH}")

    # Make predictions
    predictions = prediction_pipeline(X_val)
    log_info("✅ Predictions completed")

    # Evaluate model
    conf_matrix, acc_score, class_report = evaluation_matrices(X_val, y_val)
    log_info("✅ Model evaluation completed")
    log_info(f"Confusion Matrix:\n{conf_matrix}")
    log_info(f"Accuracy Score: {acc_score}")
    log_info(f"Classification Report:\n{class_report}")

if __name__ == "__main__":
    main()
