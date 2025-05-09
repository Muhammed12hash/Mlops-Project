import os
import subprocess
from mlflow_config import MLFLOW_TRACKING_URI, ARTIFACT_STORE

def start_server(host="127.0.0.1", port=5000):
    """Start the MLflow tracking server with the configured backend store and artifact store."""
    cmd = [
        "mlflow", "server",
        "--backend-store-uri", MLFLOW_TRACKING_URI,
        "--default-artifact-root", ARTIFACT_STORE,
        "--host", host,
        "--port", str(port)
    ]
    
    print(f"Starting MLflow server with:")
    print(f"Tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"Artifact Store: {ARTIFACT_STORE}")
    print(f"Server URL: http://{host}:{port}")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nShutting down MLflow server...")

if __name__ == "__main__":
    # Ensure the artifact store directory exists
    os.makedirs(ARTIFACT_STORE, exist_ok=True)
    
    # Start the server
    start_server() 