-- Ensure the user exists and has proper permissions
DO
$do$
BEGIN
   IF NOT EXISTS (
      SELECT FROM pg_catalog.pg_roles
      WHERE rolname = 'mlflow_user'
   ) THEN
      CREATE USER mlflow_user WITH PASSWORD 'mlflow_password';
   END IF;
END
$do$;

-- Grant login permission
ALTER ROLE mlflow_user WITH LOGIN;

-- Grant database creation permission
ALTER ROLE mlflow_user WITH CREATEDB;

-- Create the database if it doesn't exist
DO
$do$
BEGIN
   IF NOT EXISTS (
      SELECT FROM pg_database WHERE datname = 'mlflow_db'
   ) THEN
      CREATE DATABASE mlflow_db WITH OWNER = mlflow_user;
   END IF;
END
$do$;

-- Connect to mlflow_db
\c mlflow_db;

-- Grant all privileges on all tables in the database to mlflow_user
GRANT ALL PRIVILEGES ON DATABASE mlflow_db TO mlflow_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO mlflow_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO mlflow_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO mlflow_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO mlflow_user;

-- Make sure the user owns the schema
ALTER SCHEMA public OWNER TO mlflow_user;

-- Create extension if it doesn't exist
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Verify permissions
SELECT 
    table_catalog,
    table_schema,
    table_name,
    privilege_type
FROM information_schema.table_privileges 
WHERE grantee = 'mlflow_user'
ORDER BY table_name; 