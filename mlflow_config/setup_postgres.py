import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from mlflow_config import DatabaseConfig

def setup_postgres_db():
    """Set up PostgreSQL database for MLflow."""
    
    # Connect to PostgreSQL server using the default postgres database
    print("Connecting to PostgreSQL server...")
    conn = psycopg2.connect(
        host=DatabaseConfig.POSTGRES['host'],
        port=DatabaseConfig.POSTGRES['port'],
        user=DatabaseConfig.POSTGRES['user'],
        password=DatabaseConfig.POSTGRES['password'],
        database='postgres'  # Connect to default postgres database
    )
    
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()
    
    # Create database if it doesn't exist
    print(f"Creating database '{DatabaseConfig.POSTGRES['database']}'...")
    try:
        cursor.execute(f"CREATE DATABASE {DatabaseConfig.POSTGRES['database']}")
        print("Database created successfully!")
    except psycopg2.Error as e:
        if "already exists" in str(e):
            print("Database already exists.")
        else:
            print(f"Error creating database: {e}")
    
    # Close connection to PostgreSQL server
    cursor.close()
    conn.close()
    
    # Connect to MLflow database
    print("\nConnecting to MLflow database...")
    conn = psycopg2.connect(
        host=DatabaseConfig.POSTGRES['host'],
        port=DatabaseConfig.POSTGRES['port'],
        user=DatabaseConfig.POSTGRES['user'],
        password=DatabaseConfig.POSTGRES['password'],
        database=DatabaseConfig.POSTGRES['database']
    )
    
    cursor = conn.cursor()
    
    # Create necessary extensions
    print("Creating required extensions...")
    try:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\"")
        print("UUID extension created successfully!")
    except psycopg2.Error as e:
        print(f"Error creating extension: {e}")
    
    # Commit changes and close connection
    conn.commit()
    cursor.close()
    conn.close()
    
    print("\nPostgreSQL setup completed!")
    print(f"MLflow tracking URI: {DatabaseConfig.get_postgres_uri()}")

if __name__ == "__main__":
    setup_postgres_db() 