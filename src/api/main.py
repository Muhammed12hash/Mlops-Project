import warnings
import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

# Suppress warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load the Telco Customer Churn dataset."""
    # Download data from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    return df

def preprocess_data(df):
    """Preprocess the customer churn data."""
    # Create copy of dataframe
    df_processed = df.copy()

    # Remove customer ID column
    if 'customerID' in df_processed.columns:
        df_processed.drop('customerID', axis=1, inplace=True)

    # Convert TotalCharges to numeric
    df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')
    
    # Fill missing values
    df_processed['TotalCharges'].fillna(df_processed['MonthlyCharges'] * df_processed['tenure'], inplace=True)
    
    # Convert categorical variables to numeric
    le = LabelEncoder()
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df_processed[col] = le.fit_transform(df_processed[col])

    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df_processed[numerical_cols] = scaler.fit_transform(df_processed[numerical_cols])
    
    return df_processed

def plot_correlation_matrix(df):
    """Plot correlation matrix of features."""
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()

def plot_feature_importance(model, feature_names, model_name):
    """Plot feature importance."""
    if hasattr(model, 'feature_importances_'):
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
    elif hasattr(model, 'coef_'):
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': np.abs(model.coef_[0])
        }).sort_values('importance', ascending=False)
    else:
        return None
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=importance)
    plt.title(f'Feature Importance - {model_name}')
    plt.tight_layout()
    plt.savefig(f'feature_importance_{model_name}.png')
    plt.close()
    
    return importance

def plot_roc_curve(model, X, y, model_name):
    """Plot ROC curve."""
    y_pred_proba = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.tight_layout()
    plt.savefig(f'roc_curve_{model_name}.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_name}.png')
    plt.close()

def evaluate_model(model, X, y, model_name):
    """Evaluate the model and return various metrics."""
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1': f1_score(y, y_pred),
        'roc_auc': roc_auc_score(y, y_pred_proba)
    }
    
    # Generate plots
    plot_confusion_matrix(y, y_pred, model_name)
    plot_roc_curve(model, X, y, model_name)
    
    return metrics

def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression model."""
    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train, params):
    """Train Random Forest model."""
    model = RandomForestClassifier(
        n_estimators=int(params['n_estimators']),
        max_depth=int(params['max_depth']),
        min_samples_split=int(params['min_samples_split']),
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train, params):
    """Train XGBoost model."""
    model = xgb.XGBClassifier(
        n_estimators=int(params['n_estimators']),
        max_depth=int(params['max_depth']),
        learning_rate=params['learning_rate'],
        scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),  # Handle class imbalance
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def main():
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("telco_churn_baseline_models")

    # Ensure no active runs
    active_run = mlflow.active_run()
    if active_run:
        print("Ending existing active run...")
        mlflow.end_run()

    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_data()
    df_processed = preprocess_data(df)
    
    # Plot correlation matrix
    plot_correlation_matrix(df_processed)
    mlflow.log_artifact("correlation_matrix.png")

    # Split features and target
    X = df_processed.drop('Churn', axis=1)
    y = df_processed['Churn']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train and evaluate Logistic Regression
    print("\nTraining Logistic Regression...")
    with mlflow.start_run(run_name="logistic_regression"):
        # Train model
        lr_model = train_logistic_regression(X_train, y_train)
        
        # Log parameters
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("class_weight", "balanced")
        
        # Evaluate and log metrics
        lr_metrics = evaluate_model(lr_model, X_test, y_test, "LogisticRegression")
        mlflow.log_metrics(lr_metrics)

        # Plot and log feature importance
        lr_importance = plot_feature_importance(lr_model, X.columns, "LogisticRegression")
        if lr_importance is not None:
            print("\nTop 5 features (Logistic Regression):")
            print(lr_importance.head())
        
        # Log artifacts
        mlflow.log_artifact(f"confusion_matrix_LogisticRegression.png")
        mlflow.log_artifact(f"roc_curve_LogisticRegression.png")
        mlflow.log_artifact(f"feature_importance_LogisticRegression.png")
        
        # Log model
        mlflow.sklearn.log_model(lr_model, "model")
    
    # Train and evaluate Random Forest
    print("\nTraining Random Forest...")
    rf_params = {
        'n_estimators': 200,
        'max_depth': 10,
        'min_samples_split': 5
    }
    
    with mlflow.start_run(run_name="random_forest"):
        # Train model
        rf_model = train_random_forest(X_train, y_train, rf_params)
        
        # Log parameters
        mlflow.log_params(rf_params)
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("class_weight", "balanced")
        
        # Evaluate and log metrics
        rf_metrics = evaluate_model(rf_model, X_test, y_test, "RandomForest")
        mlflow.log_metrics(rf_metrics)
        
        # Plot and log feature importance
        rf_importance = plot_feature_importance(rf_model, X.columns, "RandomForest")
        print("\nTop 5 features (Random Forest):")
        print(rf_importance.head())
        
        # Log artifacts
        mlflow.log_artifact(f"confusion_matrix_RandomForest.png")
        mlflow.log_artifact(f"roc_curve_RandomForest.png")
        mlflow.log_artifact(f"feature_importance_RandomForest.png")
        
        # Log model
        mlflow.sklearn.log_model(rf_model, "model")
    
    # Train and evaluate XGBoost
    print("\nTraining XGBoost...")
    xgb_params = {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.1
    }
    
    with mlflow.start_run(run_name="xgboost"):
        # Train model
        xgb_model = train_xgboost(X_train, y_train, xgb_params)
        
        # Log parameters
        mlflow.log_params(xgb_params)
        mlflow.log_param("model_type", "XGBoost")
        
        # Evaluate and log metrics
        xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
        mlflow.log_metrics(xgb_metrics)
        
        # Plot and log feature importance
        xgb_importance = plot_feature_importance(xgb_model, X.columns, "XGBoost")
        print("\nTop 5 features (XGBoost):")
        print(xgb_importance.head())
        
        # Log artifacts
        mlflow.log_artifact(f"confusion_matrix_XGBoost.png")
        mlflow.log_artifact(f"roc_curve_XGBoost.png")
        mlflow.log_artifact(f"feature_importance_XGBoost.png")
        
        # Log model
        mlflow.sklearn.log_model(xgb_model, "model")
    
    # Print final comparison
    print("\nModel Comparison:")
    print("Logistic Regression:")
    for metric, value in lr_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nRandom Forest:")
    for metric, value in rf_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nXGBoost:")
    for metric, value in xgb_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nAll models and artifacts have been logged to MLflow")
    print("You can view the results at http://127.0.0.1:5000")

if __name__ == "__main__":
    main()
