import boto3
from botocore.exceptions import ClientError
from mlflow_config import S3Config

def setup_s3_storage():
    """Set up S3 bucket for MLflow artifact storage."""
    
    print("Configuring S3 storage for MLflow artifacts...")
    
    # Create S3 client
    s3_client = boto3.client(
        's3',
        aws_access_key_id=S3Config.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=S3Config.AWS_SECRET_ACCESS_KEY,
        region_name=S3Config.AWS_REGION,
        endpoint_url=S3Config.MLFLOW_S3_ENDPOINT_URL
    )
    
    # Create bucket if it doesn't exist
    try:
        s3_client.head_bucket(Bucket=S3Config.S3_BUCKET)
        print(f"Bucket '{S3Config.S3_BUCKET}' already exists.")
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            print(f"Creating bucket '{S3Config.S3_BUCKET}'...")
            try:
                if S3Config.AWS_REGION == 'us-east-1':
                    s3_client.create_bucket(Bucket=S3Config.S3_BUCKET)
                else:
                    s3_client.create_bucket(
                        Bucket=S3Config.S3_BUCKET,
                        CreateBucketConfiguration={'LocationConstraint': S3Config.AWS_REGION}
                    )
                print("Bucket created successfully!")
            except ClientError as e:
                print(f"Error creating bucket: {e}")
                return
        else:
            print(f"Error accessing bucket: {e}")
            return
    
    # Configure bucket for MLflow
    try:
        # Enable versioning
        s3_client.put_bucket_versioning(
            Bucket=S3Config.S3_BUCKET,
            VersioningConfiguration={'Status': 'Enabled'}
        )
        print("Bucket versioning enabled.")
        
        # Set lifecycle rules (optional)
        lifecycle_rules = {
            'Rules': [
                {
                    'ID': 'MLflow artifacts cleanup',
                    'Status': 'Enabled',
                    'Prefix': 'mlflow-artifacts/',
                    'NoncurrentVersionExpiration': {'NoncurrentDays': 90},
                    'AbortIncompleteMultipartUpload': {'DaysAfterInitiation': 7}
                }
            ]
        }
        s3_client.put_bucket_lifecycle_configuration(
            Bucket=S3Config.S3_BUCKET,
            LifecycleConfiguration=lifecycle_rules
        )
        print("Lifecycle rules configured.")
        
    except ClientError as e:
        print(f"Error configuring bucket: {e}")
        return
    
    print("\nS3 setup completed!")
    print(f"MLflow artifact store URI: {S3Config.get_s3_uri()}")
    print("\nMake sure to set the following environment variables:")
    print(f"export AWS_ACCESS_KEY_ID={S3Config.AWS_ACCESS_KEY_ID}")
    print(f"export AWS_SECRET_ACCESS_KEY={S3Config.AWS_SECRET_ACCESS_KEY}")
    print(f"export AWS_DEFAULT_REGION={S3Config.AWS_REGION}")
    print(f"export MLFLOW_S3_ENDPOINT_URL={S3Config.MLFLOW_S3_ENDPOINT_URL}")

if __name__ == "__main__":
    setup_s3_storage() 