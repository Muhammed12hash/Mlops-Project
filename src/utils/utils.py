import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

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
    from sklearn.metrics import roc_curve
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
    from sklearn.metrics import confusion_matrix
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
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
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