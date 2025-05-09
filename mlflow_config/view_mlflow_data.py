import psycopg2
import pandas as pd
from tabulate import tabulate
from mlflow_config import DatabaseConfig

def connect_to_db():
    """Connect to the PostgreSQL database."""
    return psycopg2.connect(
        host=DatabaseConfig.POSTGRES['host'],
        port=DatabaseConfig.POSTGRES['port'],
        database=DatabaseConfig.POSTGRES['database'],
        user=DatabaseConfig.POSTGRES['user'],
        password=DatabaseConfig.POSTGRES['password']
    )

def execute_query(query, params=None):
    """Execute a query and return results as a pandas DataFrame."""
    with connect_to_db() as conn:
        return pd.read_sql_query(query, conn, params=params)

def view_experiments():
    """View all experiments."""
    query = """
    SELECT experiment_id, name, artifact_location, lifecycle_stage
    FROM experiments;
    """
    df = execute_query(query)
    print("\n=== MLflow Experiments ===")
    print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))

def view_model_performance():
    """View performance metrics for all runs."""
    query = """
    SELECT e.name as experiment_name,
           r.run_uuid,
           MAX(CASE WHEN m.key = 'accuracy' THEN m.value END) as accuracy,
           MAX(CASE WHEN m.key = 'precision' THEN m.value END) as precision,
           MAX(CASE WHEN m.key = 'recall' THEN m.value END) as recall,
           MAX(CASE WHEN m.key = 'f1' THEN m.value END) as f1,
           MAX(CASE WHEN m.key = 'roc_auc' THEN m.value END) as roc_auc
    FROM experiments e
    JOIN runs r ON e.experiment_id = r.experiment_id
    JOIN metrics m ON r.run_uuid = m.run_uuid
    GROUP BY e.name, r.run_uuid
    ORDER BY e.name, r.run_uuid;
    """
    df = execute_query(query)
    print("\n=== Model Performance Metrics ===")
    print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))

def view_run_parameters(experiment_name=None):
    """View parameters for all runs in an experiment."""
    query = """
    SELECT e.name as experiment_name,
           r.run_uuid,
           p.key as parameter,
           p.value as value
    FROM experiments e
    JOIN runs r ON e.experiment_id = r.experiment_id
    JOIN params p ON r.run_uuid = p.run_uuid
    {}
    ORDER BY e.name, r.run_uuid, p.key;
    """.format("WHERE e.name = %(experiment_name)s" if experiment_name else "")
    
    params = {'experiment_name': experiment_name} if experiment_name else None
    df = execute_query(query, params)
    print(f"\n=== Run Parameters{f' for {experiment_name}' if experiment_name else ''} ===")
    print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))

def view_artifact_locations():
    """View artifact locations for all runs."""
    query = """
    SELECT e.name as experiment_name,
           r.run_uuid,
           r.artifact_uri
    FROM experiments e
    JOIN runs r ON e.experiment_id = r.experiment_id;
    """
    df = execute_query(query)
    print("\n=== Artifact Locations ===")
    print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))

if __name__ == "__main__":
    print("MLflow PostgreSQL Database Contents:")
    view_experiments()
    view_model_performance()
    view_run_parameters()
    view_artifact_locations() 