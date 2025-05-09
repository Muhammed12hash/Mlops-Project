import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import os

def load_and_preprocess_data():
    """Load and preprocess the data."""
    # Load the data
    df = pd.read_csv("data/customer_churn.csv")
    
    # Drop unnecessary columns
    df = df.drop(['customerID'], axis=1)
    
    # Split features and target before encoding
    X = df.drop('Churn', axis=1)
    y = (df['Churn'] == 'Yes').astype(int)
    
    # Convert categorical variables to numeric
    categorical_columns = X.select_dtypes(include=['object']).columns
    X = pd.get_dummies(X, columns=categorical_columns)
    
    return X, y

def save_test_data(test_size=0.2, random_state=42):
    """Save test data for monitoring."""
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Load and preprocess data
    X, y = load_and_preprocess_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Save test data
    joblib.dump(X_test, "data/X_test.joblib")
    joblib.dump(y_test, "data/y_test.joblib")
    
    print("Test data saved successfully!")
    print(f"Number of test samples: {len(X_test)}")

if __name__ == "__main__":
    save_test_data() 