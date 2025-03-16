from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import logging
from fastapi.middleware.cors import CORSMiddleware

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the model, scaler, and feature names
try:
    model = joblib.load('credit_card_fraud_model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_names = joblib.load('feature_names.pkl')
    logging.info("Model, scaler, and feature names loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model, scaler, or feature names: {e}")
    exit()

# Define the input data model for the API
class Transaction(BaseModel):
    id: float
    Amount: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

# Initialize FastAPI
app = FastAPI(title="Credit Card Fraud Detection API", description="API for detecting fraudulent transactions")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Replace with your React app's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
def predict(transaction: Transaction):
    try:
        # Log the incoming request
        logging.info(f"Incoming request data: {transaction.model_dump()}")

        # Convert input data to a DataFrame
        input_data = pd.DataFrame([transaction.model_dump()])

        # Ensure the columns are in the correct order
        input_data = input_data[feature_names]

        # Scale the input data
        input_data_scaled = scaler.transform(input_data)

        # Make a prediction
        prediction = model.predict(input_data_scaled)
        prediction_proba = model.predict_proba(input_data_scaled)[:, 1]

        # Return the result
        return {
            "prediction": int(prediction[0]),
            "probability": float(prediction_proba[0]),
            "is_fraud": bool(prediction[0] == 1)
        }
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Run the API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)