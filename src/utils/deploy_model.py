import mlflow
import pandas as pd
import numpy as np
from datetime import datetime
from utils import load_data, preprocess_data
from mlflow_config.mlflow_config import MLFLOW_TRACKING_URI, EXPERIMENTS
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import time
import json
import os

# Set MLflow tracking URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def load_production_model(experiment_name):
    """Load the production model from MLflow."""
    client = mlflow.tracking.MlflowClient()
    
    # Get the experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        raise ValueError(f"Experiment {experiment_name} not found")
    
    # Get all runs in the experiment
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="",
        order_by=["metrics.mean_cv_auc DESC"]
    )
    
    if not runs:
        raise ValueError("No runs found in the experiment")
    
    # Get the best run
    best_run = runs[0]
    print(f"Loading model from run {best_run.info.run_id} with AUC: {best_run.data.metrics['mean_cv_auc']:.4f}")
    
    # Load the model
    model_uri = f"runs:/{best_run.info.run_id}/model"
    model = mlflow.pyfunc.load_model(model_uri)
    
    return model, best_run.info.run_id

def simulate_production_data(base_data, n_samples=100):
    """Simulate production data by sampling and adding noise."""
    # Randomly sample from base data
    sampled_indices = np.random.choice(len(base_data), size=n_samples, replace=True)
    production_data = base_data.iloc[sampled_indices].copy()
    
    # Add some random noise to numeric columns
    numeric_cols = production_data.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        noise = np.random.normal(0, 0.1 * production_data[col].std(), size=len(production_data))
        production_data[col] = production_data[col] + noise
    
    return production_data

def log_predictions(predictions, actual, timestamp, run_id):
    """Log prediction metrics and monitoring data."""
    metrics = {
        'timestamp': timestamp,
        'roc_auc': roc_auc_score(actual, predictions),
        'precision': precision_score(actual, predictions),
        'recall': recall_score(actual, predictions),
        'f1': f1_score(actual, predictions)
    }
    
    # Log metrics to MLflow
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({
            f"monitoring_{k}": v for k, v in metrics.items() 
            if k != 'timestamp'
        })
    
    # Also save metrics to a local file for monitoring
    monitoring_dir = "model_monitoring"
    os.makedirs(monitoring_dir, exist_ok=True)
    
    monitoring_file = os.path.join(monitoring_dir, "metrics_log.jsonl")
    with open(monitoring_file, 'a') as f:
        f.write(json.dumps(metrics) + '\n')
    
    return metrics

def main():
    # Load the best model
    print("Loading production model...")
    model, run_id = load_production_model(EXPERIMENTS["xgboost"])
    
    # Load and preprocess base data
    print("Loading base data...")
    df = load_data()
    df_processed = preprocess_data(df)
    
    # Simulate production environment
    print("\nStarting production simulation...")
    monitoring_metrics = []
    
    try:
        while True:
            # Simulate batch of production data
            production_data = simulate_production_data(df_processed, n_samples=100)
            X_prod = production_data.drop('Churn', axis=1)
            y_prod = production_data['Churn']
            
            # Make predictions
            predictions = model.predict(X_prod)
            
            # Log metrics
            timestamp = datetime.now().isoformat()
            metrics = log_predictions(predictions, y_prod, timestamp, run_id)
            monitoring_metrics.append(metrics)
            
            # Print current metrics
            print(f"\nMetrics at {timestamp}:")
            for k, v in metrics.items():
                if k != 'timestamp':
                    print(f"{k}: {v:.4f}")
            
            # Sleep for a minute before next batch
            time.sleep(60)
            
    except KeyboardInterrupt:
        print("\nStopping production simulation...")
        
        # Calculate and print average metrics
        if monitoring_metrics:
            print("\nAverage metrics:")
            avg_metrics = pd.DataFrame(monitoring_metrics).mean()
            for k, v in avg_metrics.items():
                if k != 'timestamp':
                    print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main() 