import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import json
import os
from datetime import datetime
import mlflow
import joblib

def load_model(model_name):
    """Load the model from MLflow."""
    client = mlflow.tracking.MlflowClient()
    
    # Get the latest version of the model
    model_version = client.get_latest_versions(model_name, stages=["Production"])[0]
    model_uri = f"models:/{model_name}/{model_version.version}"
    
    return mlflow.sklearn.load_model(model_uri)

def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate performance metrics."""
    return {
        'roc_auc': roc_auc_score(y_true, y_prob),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }

def log_metrics(metrics):
    """Log metrics to a JSONL file."""
    os.makedirs("model_monitoring", exist_ok=True)
    log_file = os.path.join("model_monitoring", "metrics_log.jsonl")
    
    # Add timestamp to metrics
    metrics['timestamp'] = datetime.now().isoformat()
    
    # Append to JSONL file
    with open(log_file, 'a') as f:
        f.write(json.dumps(metrics) + '\n')

def monitor_model_performance(model_name, X_test, y_test, batch_size=100):
    """Monitor model performance on test data in batches."""
    try:
        # Load the model
        model = load_model(model_name)
        
        # Process data in batches
        n_samples = len(X_test)
        for i in range(0, n_samples, batch_size):
            # Get batch data
            X_batch = X_test[i:min(i + batch_size, n_samples)]
            y_batch = y_test[i:min(i + batch_size, n_samples)]
            
            # Make predictions
            y_pred = model.predict(X_batch)
            y_prob = model.predict_proba(X_batch)[:, 1]
            
            # Calculate metrics
            metrics = calculate_metrics(y_batch, y_pred, y_prob)
            
            # Log metrics
            log_metrics(metrics)
            
            print(f"Processed batch {i//batch_size + 1}, metrics logged.")
            
    except Exception as e:
        print(f"Error during monitoring: {str(e)}")

if __name__ == "__main__":
    # Load test data
    X_test = joblib.load("data/X_test.joblib")
    y_test = joblib.load("data/y_test.joblib")
    
    # Monitor the model (replace with your model name)
    monitor_model_performance("credit_risk_model", X_test, y_test) 