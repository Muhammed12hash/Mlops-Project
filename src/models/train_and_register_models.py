import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import os

# Set MLflow tracking URI
mlflow.set_tracking_uri("postgresql://mlflow_user:mlflow_password@localhost:5432/mlflow_db")

def load_and_preprocess_data():
    # Load data
    df = pd.read_csv('data/customer_churn.csv')
    
    # Drop customerID
    df = df.drop('customerID', axis=1)
    
    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()
    
    # Convert categorical variables to numeric
    categorical_columns = df.select_dtypes(include=['object']).columns
    df_encoded = pd.get_dummies(df, columns=[col for col in categorical_columns if col != 'Churn'])
    
    # Convert target variable
    df_encoded['Churn'] = (df_encoded['Churn'] == 'Yes').astype(int)
    
    # Split features and target
    X = df_encoded.drop('Churn', axis=1)
    y = df_encoded['Churn']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X_train.columns

def evaluate_model(y_true, y_pred, y_pred_proba):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_pred_proba)
    }

def train_logistic_regression(X_train, X_test, y_train, y_test, feature_names):
    with mlflow.start_run(run_name="logistic_regression") as run:
        # Set experiment
        mlflow.set_experiment("Logistic Regression")
        
        # Train model
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Log metrics
        metrics = evaluate_model(y_test, y_pred, y_pred_proba)
        mlflow.log_metrics(metrics)
        
        # Log feature importance
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': abs(model.coef_[0])
        }).sort_values('importance', ascending=False)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        print("Logistic Regression metrics:", metrics)
        print("\nTop 5 important features:")
        print(importance.head())

def train_random_forest(X_train, X_test, y_train, y_test, feature_names):
    with mlflow.start_run(run_name="random_forest") as run:
        # Set experiment
        mlflow.set_experiment("Random Forest")
        
        # Train model
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Log metrics
        metrics = evaluate_model(y_test, y_pred, y_pred_proba)
        mlflow.log_metrics(metrics)
        
        # Log feature importance
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        print("\nRandom Forest metrics:", metrics)
        print("\nTop 5 important features:")
        print(importance.head())

def train_xgboost(X_train, X_test, y_train, y_test, feature_names):
    with mlflow.start_run(run_name="xgboost") as run:
        # Set experiment
        mlflow.set_experiment("XGBoost")
        
        # Train model
        model = xgb.XGBClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Log metrics
        metrics = evaluate_model(y_test, y_pred, y_pred_proba)
        mlflow.log_metrics(metrics)
        
        # Log feature importance
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        print("\nXGBoost metrics:", metrics)
        print("\nTop 5 important features:")
        print(importance.head())

if __name__ == "__main__":
    # Load and preprocess data
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data()
    
    # Train and log all models
    train_logistic_regression(X_train, X_test, y_train, y_test, feature_names)
    train_random_forest(X_train, X_test, y_train, y_test, feature_names)
    train_xgboost(X_train, X_test, y_train, y_test, feature_names) 