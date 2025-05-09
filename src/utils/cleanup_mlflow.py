import mlflow
from mlflow.tracking import MlflowClient
from mlflow_config.mlflow_config import MLFLOW_TRACKING_URI, EXPERIMENTS

# Initialize MLflow client
client = MlflowClient()

# Set up MLflow tracking
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# End any active runs
try:
    active_run = mlflow.active_run()
    if active_run is not None:
        print(f"Ending active run: {active_run.info.run_id}")
        mlflow.end_run()
    else:
        print("No active run found")
except Exception as e:
    print(f"Error handling active run: {e}")

# Create experiment if it doesn't exist
experiment_name = "telco_churn_baseline_models"
try:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Creating experiment: {experiment_name}")
        mlflow.create_experiment(experiment_name)
    else:
        print(f"Found existing experiment: {experiment_name}")
        
        # Search for any runs in 'running' state
        running_runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="attributes.status = 'RUNNING'"
        )
        
        for run in running_runs:
            print(f"Cleaning up running run: {run.info.run_id}")
            client.set_terminated(run.info.run_id)
            
        if not running_runs:
            print("No running runs found in experiment")
except Exception as e:
    print(f"Error managing experiment: {e}")

print("Cleanup completed") 