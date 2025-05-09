from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import mlflow
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
from pydantic import BaseModel
import time
import logging
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Telco Churn Prediction API",
    description="API for predicting customer churn using multiple ML models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories if they don't exist
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Serve static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Set up MLflow
os.environ['MLFLOW_TRACKING_URI'] = "http://127.0.0.1:5000"
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Add a delay to ensure MLflow is ready
time.sleep(2)  # Wait for MLflow to be ready

# Initialize global variables
models = {}
scaler = StandardScaler()
label_encoders = {}

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

class PredictionResponse(BaseModel):
    model_name: str
    prediction: int
    probability: float

def load_model(model_name: str):
    """Load a model from MLflow"""
    try:
        client = mlflow.tracking.MlflowClient()
        model_uri = f"models:/{model_name}/Production"
        
        if "XGBoost" in model_name:
            return mlflow.xgboost.load_model(model_uri)
        else:
            return mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {str(e)}")
        return None

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    global models
    model_names = ["LogisticRegression_Tuned", "RandomForest_Tuned", "XGBoost_Tuned"]
    
    for model_name in model_names:
        model = load_model(model_name)
        if model is not None:
            models[model_name] = model
            logger.info(f"Successfully loaded model: {model_name}")
        else:
            logger.warning(f"Failed to load model: {model_name}")
    
    if not models:
        logger.error("No models were loaded successfully!")
    else:
        logger.info(f"Loaded {len(models)} models: {list(models.keys())}")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/models")
async def list_models():
    return {"available_models": list(models.keys())}

@app.post("/predict", response_model=List[PredictionResponse])
async def predict(data: CustomerData):
    """Make predictions using loaded models"""
    try:
        if not models:
            raise HTTPException(
                status_code=503,
                detail="No models available for prediction. Please try again later."
            )
        
        # Preprocess input data
        processed_data = preprocess_data(data)
        
        predictions = []
        
        # Get predictions from each model
        for model_name, model in models.items():
            try:
                # Get prediction probability
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(processed_data)[0][1]
                else:
                    proba = model.predict(processed_data)[0]
                
                # Get prediction (0 or 1)
                pred = int(proba >= 0.5)
                
                predictions.append(PredictionResponse(
                    model_name=model_name,
                    prediction=pred,
                    probability=float(proba)
                ))
            except Exception as e:
                logger.error(f"Error getting prediction from {model_name}: {str(e)}")
                continue
        
        if not predictions:
            raise HTTPException(
                status_code=500,
                detail="Failed to get predictions from any model"
            )
        
        return predictions
        
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/{model_name}")
async def get_model_info(model_name: str):
    if model_name not in models:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    try:
        client = mlflow.tracking.MlflowClient()
        model_version = client.get_latest_versions(model_name)[0]
        
        return {
            "model_name": model_name,
            "version": model_version.version,
            "run_id": model_version.run_id,
            "status": model_version.status
        }
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def preprocess_data(data: CustomerData) -> pd.DataFrame:
    """Preprocess input data to match model requirements"""
    try:
        # Convert input to DataFrame
        df = pd.DataFrame([data.dict()])
        
        # Scale numeric features
        numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
        df[numeric_features] = scaler.fit_transform(df[numeric_features])
        
        # One-hot encode categorical features
        categorical_features = [
            'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
            'PaperlessBilling', 'PaymentMethod'
        ]
        
        # Create dummy variables for all categorical features
        df_encoded = pd.get_dummies(df, columns=categorical_features, prefix=categorical_features)
        
        # Ensure all expected columns are present
        expected_columns = [
            'tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen',
            'gender_Female', 'gender_Male',
            'Partner_No', 'Partner_Yes',
            'Dependents_No', 'Dependents_Yes',
            'PhoneService_No', 'PhoneService_Yes',
            'MultipleLines_No', 'MultipleLines_No phone service', 'MultipleLines_Yes',
            'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No',
            'OnlineSecurity_No', 'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
            'OnlineBackup_No', 'OnlineBackup_No internet service', 'OnlineBackup_Yes',
            'DeviceProtection_No', 'DeviceProtection_No internet service', 'DeviceProtection_Yes',
            'TechSupport_No', 'TechSupport_No internet service', 'TechSupport_Yes',
            'StreamingTV_No', 'StreamingTV_No internet service', 'StreamingTV_Yes',
            'StreamingMovies_No', 'StreamingMovies_No internet service', 'StreamingMovies_Yes',
            'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year',
            'PaperlessBilling_No', 'PaperlessBilling_Yes',
            'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)',
            'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
        ]
        
        # Add missing columns with 0 values
        for col in expected_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        
        # Reorder columns to match expected order
        df_encoded = df_encoded[expected_columns]
        
        return df_encoded
        
    except Exception as e:
        logger.error(f"Error in preprocess_data: {str(e)}")
        raise

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000) 