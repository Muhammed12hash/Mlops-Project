import os
from typing import Dict, Any

# Database Configuration
class DatabaseConfig:
    # SQLite Configuration (default)
    SQLITE = {
        'tracking_uri': "sqlite:///mlflow_config/mlflow.db"
    }
    
    # PostgreSQL Configuration
    POSTGRES = {
        'host': 'localhost',
        'port': 5432,
        'database': 'mlflow_db',
        'user': 'mlflow_user',
        'password': 'mlflow_password'
    }
    
    @staticmethod
    def get_postgres_uri() -> str:
        return (f"postgresql://{DatabaseConfig.POSTGRES['user']}:{DatabaseConfig.POSTGRES['password']}"
                f"@{DatabaseConfig.POSTGRES['host']}:{DatabaseConfig.POSTGRES['port']}"
                f"/{DatabaseConfig.POSTGRES['database']}")

# S3 Configuration
class S3Config:
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID', 'your-access-key')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY', 'your-secret-key')
    AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
    S3_BUCKET = os.getenv('MLFLOW_S3_BUCKET', 'your-bucket-name')
    MLFLOW_S3_ENDPOINT_URL = os.getenv('MLFLOW_S3_ENDPOINT_URL', 'https://s3.amazonaws.com')
    
    @staticmethod
    def get_s3_uri() -> str:
        return f"s3://{S3Config.S3_BUCKET}/mlflow-artifacts"

# Storage Configuration
class StorageConfig:
    # Local filesystem (default)
    LOCAL_ARTIFACT_PATH = os.path.abspath("mlflow_config/artifacts")
    
    # Create local artifact store directory
    os.makedirs(LOCAL_ARTIFACT_PATH, exist_ok=True)
    
    @staticmethod
    def get_artifact_uri(storage_type: str = 'local') -> str:
        if storage_type.lower() == 's3':
            return S3Config.get_s3_uri()
        return StorageConfig.LOCAL_ARTIFACT_PATH

# MLflow Configuration
class MLflowConfig:
    # Experiment names
    EXPERIMENTS: Dict[str, str] = {
        "logistic_regression": "logistic_regression_experiment",
        "random_forest": "random_forest_experiment",
        "xgboost": "xgboost_experiment"
    }
    
    # Server configuration
    SERVER_HOST = "127.0.0.1"
    SERVER_PORT = 5000
    
    @staticmethod
    def get_tracking_uri(db_type: str = 'sqlite') -> str:
        if db_type.lower() == 'postgres':
            return DatabaseConfig.get_postgres_uri()
        return DatabaseConfig.SQLITE['tracking_uri']

# Active configuration (modify these to change the active configuration)
ACTIVE_DB = 'postgres'  # Options: 'sqlite', 'postgres'
ACTIVE_STORAGE = 'local'  # Options: 'local', 's3'

# Export configured values
MLFLOW_TRACKING_URI = MLflowConfig.get_tracking_uri(ACTIVE_DB)
ARTIFACT_STORE = StorageConfig.get_artifact_uri(ACTIVE_STORAGE)
EXPERIMENTS = MLflowConfig.EXPERIMENTS 