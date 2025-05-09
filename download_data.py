import os
import pandas as pd
import requests
from io import StringIO

def download_data():
    """Download the Telco Churn dataset"""
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # URL for the dataset
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    
    try:
        # Download the data
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Save the data
        with open("data/telco_churn.csv", "w") as f:
            f.write(response.text)
        
        print("Dataset downloaded successfully!")
        
        # Verify the data
        df = pd.read_csv("data/telco_churn.csv")
        print(f"\nDataset shape: {df.shape}")
        print("\nFirst few rows:")
        print(df.head())
        
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")

if __name__ == "__main__":
    download_data() 