import mlflow
import os
from mlflow.tracking import MlflowClient

# Set up MLflow
os.environ['MLFLOW_TRACKING_URI'] = "http://127.0.0.1:5000"
mlflow.set_tracking_uri("http://127.0.0.1:5000")

def register_model(model_name, run_id):
    """Register a model in MLflow Model Registry"""
    try:
        # Create the model if it doesn't exist
        client = MlflowClient()
        try:
            client.create_registered_model(model_name)
        except Exception:
            print(f"Model {model_name} already exists")
        
        # Register the model version
        model_uri = f"runs:/{run_id}/model"
        model_details = client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=run_id
        )
        
        # Transition the model to Production stage
        client.transition_model_version_stage(
            name=model_name,
            version=model_details.version,
            stage="Production"
        )
        
        print(f"Successfully registered {model_name} with version {model_details.version}")
        return True
    except Exception as e:
        print(f"Error registering {model_name}: {str(e)}")
        return False

def main():
    # List of models to register with their run IDs
    models_to_register = [
        ("LogisticRegression_Tuned", "your_run_id_here"),
        ("RandomForest_Tuned", "your_run_id_here"),
        ("XGBoost_Tuned", "your_run_id_here")
    ]
    
    # Get all runs from the default experiment
    client = MlflowClient()
    experiment = client.get_experiment_by_name("Default")
    if experiment is None:
        print("No default experiment found")
        return
    
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    
    # Print available runs
    print("\nAvailable runs:")
    for run in runs:
        print(f"Run ID: {run.info.run_id}")
        print(f"Status: {run.info.status}")
        print(f"Start Time: {run.info.start_time}")
        print("---")
    
    # Ask for run IDs
    print("\nPlease enter the run IDs for each model:")
    for i, (model_name, _) in enumerate(models_to_register):
        run_id = input(f"Enter run ID for {model_name}: ")
        models_to_register[i] = (model_name, run_id)
    
    # Register models
    print("\nRegistering models...")
    for model_name, run_id in models_to_register:
        register_model(model_name, run_id)

if __name__ == "__main__":
    main() 