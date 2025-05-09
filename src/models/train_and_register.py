import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import os

# Set up MLflow
os.environ['MLFLOW_TRACKING_URI'] = "http://127.0.0.1:5000"
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Create a new experiment
experiment_name = "Telco Churn Prediction"
mlflow.set_experiment(experiment_name)

def load_and_preprocess_data():
    """Load and preprocess the data"""
    # Load the data
    df = pd.read_csv("data/telco_churn.csv")
    
    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Fill missing values
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].mean())
    
    # Convert target variable
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Select features
    features = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
        'MonthlyCharges', 'TotalCharges'
    ]
    
    X = df[features]
    y = df['Churn']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale numeric features
    scaler = StandardScaler()
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_test[numeric_features] = scaler.transform(X_test[numeric_features])
    
    # One-hot encode categorical features
    X_train = pd.get_dummies(X_train, drop_first=True)
    X_test = pd.get_dummies(X_test, drop_first=True)
    
    return X_train, X_test, y_train, y_test

def train_and_log_model(model_name, model, X_train, y_train, X_test, y_test):
    """Train and log a model to MLflow"""
    with mlflow.start_run(run_name=model_name):
        # Train the model
        model.fit(X_train, y_train)
        
        # Log model parameters
        if hasattr(model, 'get_params'):
            mlflow.log_params(model.get_params())
        
        # Log model
        if isinstance(model, xgb.XGBClassifier):
            mlflow.xgboost.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")
        
        # Log metrics
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        mlflow.log_metric("train_accuracy", train_score)
        mlflow.log_metric("test_accuracy", test_score)
        
        # Register the model
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        mlflow.register_model(model_uri, model_name)
        
        # Transition to Production
        client = mlflow.tracking.MlflowClient()
        model_version = client.get_latest_versions(model_name)[0]
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Production"
        )
        
        print(f"Successfully trained and registered {model_name}")

def main():
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    # Define models
    models = {
        "LogisticRegression_Tuned": LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42
        ),
        "RandomForest_Tuned": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        ),
        "XGBoost_Tuned": xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
    }
    
    # Train and register each model
    for model_name, model in models.items():
        train_and_log_model(model_name, model, X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main() 