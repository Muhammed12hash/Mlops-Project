-- Create MLflow user if it doesn't exist
DO
$do$
BEGIN
   IF NOT EXISTS (
      SELECT FROM pg_catalog.pg_roles
      WHERE  rolname = 'mlflow_user') THEN
      CREATE USER mlflow_user WITH PASSWORD 'mlflow_password';
   END IF;
END
$do$;

-- Grant necessary permissions
ALTER USER mlflow_user CREATEDB;

-- Create database if it doesn't exist
CREATE DATABASE mlflow_db WITH OWNER mlflow_user;

-- Connect to the mlflow database
\c mlflow_db;

-- Grant all privileges on all tables in the database to mlflow_user
GRANT ALL PRIVILEGES ON DATABASE mlflow_db TO mlflow_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO mlflow_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO mlflow_user;

-- Create extension if it doesn't exist
CREATE EXTENSION IF NOT EXISTS "uuid-ossp"; 