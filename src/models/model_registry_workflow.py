import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from mlflow_config.mlflow_config import MLFLOW_TRACKING_URI, EXPERIMENTS
import time
from sklearn.metrics import roc_auc_score
import joblib
import os

def setup_mlflow():
    """Setup MLflow tracking URI and create a client."""
    os.environ['MLFLOW_TRACKING_URI'] = MLFLOW_TRACKING_URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return MlflowClient()

def register_model_version(client, model_name, run_id):
    """Register a new version of the model."""
    try:
        # Check if model exists
        try:
            client.get_registered_model(model_name)
        except mlflow.exceptions.RestException:
            client.create_registered_model(model_name)
        
        # Register new version
        model_version = client.create_model_version(
            name=model_name,
            source=f"runs:/{run_id}/model",
            run_id=run_id
        )
        return model_version
    except Exception as e:
        print(f"Error registering model version: {str(e)}")
        return None

def validate_model_performance(run_id):
    """Validate model performance using metrics from the run."""
    client = MlflowClient()
    run = client.get_run(run_id)
    
    # Get the AUC score from the run metrics
    auc_score = run.data.metrics.get('roc_auc', 0.0)
    
    # Define validation threshold
    VALIDATION_THRESHOLD = 0.75
    return auc_score > VALIDATION_THRESHOLD, auc_score

def process_model(experiment_name, model_name, client):
    """Process a single model through the registry workflow."""
    print(f"\nProcessing {model_name}...")
    
    # Get the experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment {experiment_name} not found")
    
    # Find the best run
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="",
        order_by=["metrics.roc_auc DESC"]
    )
    
    if not runs:
        raise ValueError(f"No runs found for {model_name}")
    
    best_run = runs[0]
    run_id = best_run.info.run_id
    
    # Register model version
    model_version = register_model_version(client, model_name, run_id)
    if not model_version:
        raise ValueError(f"Failed to register {model_name}")
    
    print(f"Registered {model_name} version {model_version.version}")
    
    # Validate model performance
    print(f"Validating {model_name} performance...")
    is_valid, auc_score = validate_model_performance(run_id)
    
    if is_valid:
        print(f"Model validation successful! AUC Score: {auc_score:.4f}")
        
        # Transition to Production
        print(f"Transitioning {model_name} to PRODUCTION...")
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Production"
        )
        
        # Archive previous production versions
        versions = client.search_model_versions(f"name='{model_name}'")
        for mv in versions:
            if mv.version != model_version.version and mv.current_stage == "Production":
                client.transition_model_version_stage(
                    name=model_name,
                    version=mv.version,
                    stage="Archived"
                )
        
        print(f"Successfully promoted {model_name} to production!")
        
    else:
        print(f"Model validation failed. AUC Score: {auc_score:.4f}")
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Archived"
        )
    
    return model_version, is_valid

def main():
    """Main workflow for model registration and transition."""
    client = setup_mlflow()
    
    models = {
        "logistic_regression": EXPERIMENTS["logistic_regression"],
        "random_forest": EXPERIMENTS["random_forest"],
        "xgboost": EXPERIMENTS["xgboost"]
    }
    
    results = []
    
    for model_name, experiment_name in models.items():
        try:
            model_version, is_valid = process_model(experiment_name, model_name, client)
            results.append({
                "model_name": model_name,
                "version": model_version.version if model_version else None,
                "status": "SUCCESS" if is_valid else "FAILED VALIDATION"
            })
        except Exception as e:
            print(f"Error processing {model_name}: {str(e)}")
            results.append({
                "model_name": model_name,
                "version": None,
                "status": f"ERROR: {str(e)}"
            })
    
    # Print summary
    print("\n=== Model Registration Summary ===")
    for result in results:
        print(f"\nModel: {result['model_name']}")
        print(f"Version: {result['version']}")
        print(f"Status: {result['status']}")

if __name__ == "__main__":
    main() 