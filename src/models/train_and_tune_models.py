import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from utils import load_data, preprocess_data, evaluate_model, plot_feature_importance
import os
import logging
from typing import Dict, Any, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Define search spaces for each model
logistic_space = {
    'C': hp.loguniform('C', np.log(0.001), np.log(10.0)),
    'max_iter': hp.choice('max_iter', [500, 1000, 1500, 2000]),
    'solver': hp.choice('solver', ['lbfgs', 'liblinear', 'saga'])
}

random_forest_space = {
    'n_estimators': hp.choice('n_estimators', [100, 200, 300, 400, 500]),
    'max_depth': hp.choice('max_depth', [5, 7, 10, 15, 20]),
    'min_samples_split': hp.quniform('min_samples_split', 2, 20, 1),
    'min_samples_leaf': hp.choice('min_samples_leaf', [1, 2, 4, 8])
}

xgboost_space = {
    'n_estimators': hp.choice('n_estimators', [100, 200, 300, 400, 500]),
    'max_depth': hp.choice('max_depth', [3, 4, 5, 6, 7, 8]),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
    'subsample': hp.uniform('subsample', 0.6, 1.0),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
    'min_child_weight': hp.quniform('min_child_weight', 1, 7, 1)
}

def train_logistic_regression(X_train, y_train, params=None):
    """Train Logistic Regression model."""
    if params is None:
        params = {
            'max_iter': 1000,
            'class_weight': 'balanced',
            'random_state': 42
        }
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train, params=None):
    """Train Random Forest model."""
    if params is None:
        params = {
            'n_estimators': 200,
            'max_depth': 10,
            'min_samples_split': 5,
            'class_weight': 'balanced',
            'random_state': 42
        }
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train, params=None):
    """Train XGBoost model."""
    if params is None:
        params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'use_label_encoder': False,
            'eval_metric': 'auc'
        }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model

def objective_logistic(params):
    """Objective function for Logistic Regression."""
    with mlflow.start_run(nested=True):
        # Map hyperopt parameters to actual values
        actual_params = {
            'C': float(params['C']),
            'max_iter': params['max_iter'],  # Already the actual value from hp.choice
            'solver': params['solver'],      # Already the actual value from hp.choice
            'class_weight': 'balanced',
            'random_state': 42
        }
        
        model = LogisticRegression(**actual_params)
        
        # Get cross-validation score
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        mean_auc = scores.mean()
        
        # Log parameters and metrics
        mlflow.log_params(actual_params)
        mlflow.log_metric('mean_cv_auc', mean_auc)
        
        return {'loss': -mean_auc, 'status': STATUS_OK}

def objective_random_forest(params):
    """Objective function for Random Forest."""
    with mlflow.start_run(nested=True):
        # Map hyperopt parameters to actual values
        actual_params = {
            'n_estimators': params['n_estimators'],  # Already the actual value from hp.choice
            'max_depth': params['max_depth'],        # Already the actual value from hp.choice
            'min_samples_split': int(params['min_samples_split']),
            'min_samples_leaf': params['min_samples_leaf'],  # Already the actual value from hp.choice
            'class_weight': 'balanced',
            'random_state': 42
        }
        
        model = RandomForestClassifier(**actual_params)
        
        # Get cross-validation score
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        mean_auc = scores.mean()
        
        # Log parameters and metrics
        mlflow.log_params(actual_params)
        mlflow.log_metric('mean_cv_auc', mean_auc)
        
        return {'loss': -mean_auc, 'status': STATUS_OK}

def objective_xgboost(params):
    """Objective function for XGBoost."""
    with mlflow.start_run(nested=True):
        # Map hyperopt parameters to actual values
        actual_params = {
            'n_estimators': params['n_estimators'],  # Already the actual value from hp.choice
            'max_depth': params['max_depth'],        # Already the actual value from hp.choice
            'learning_rate': float(params['learning_rate']),
            'subsample': float(params['subsample']),
            'colsample_bytree': float(params['colsample_bytree']),
            'min_child_weight': int(params['min_child_weight']),
            'random_state': 42,
            'use_label_encoder': False,
            'eval_metric': 'auc'
        }
        
        model = xgb.XGBClassifier(**actual_params)
        
        # Get cross-validation score
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        mean_auc = scores.mean()
        
        # Log parameters and metrics
        mlflow.log_params(actual_params)
        mlflow.log_metric('mean_cv_auc', mean_auc)
        
        return {'loss': -mean_auc, 'status': STATUS_OK}

def load_and_preprocess_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load and preprocess the data"""
    # Load the data
    df = pd.read_csv("data/telco_churn.csv")
    
    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Fill missing values
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].mean())
    
    # Convert target variable
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Select features
    features = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
        'MonthlyCharges', 'TotalCharges'
    ]
    
    X = df[features]
    y = df['Churn']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale numeric features
    scaler = StandardScaler()
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_test[numeric_features] = scaler.transform(X_test[numeric_features])
    
    # One-hot encode categorical features
    X_train = pd.get_dummies(X_train, drop_first=True)
    X_test = pd.get_dummies(X_test, drop_first=True)
    
    return X_train, X_test, y_train, y_test

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Evaluate model performance"""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_prob)
    }

def train_and_tune_model(
    model_name: str,
    model: Any,
    param_grid: Dict[str, list],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[Any, Dict[str, float]]:
    """Train and tune a model using GridSearchCV"""
    with mlflow.start_run(run_name=f"{model_name}_Tuning"):
        # Log parameters
        mlflow.log_params(param_grid)
        
        # Perform grid search
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Make predictions
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]
        
        # Evaluate model
        metrics = evaluate_model(y_test, y_pred, y_prob)
        
        # Log metrics
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)
        
        # Log best parameters
        mlflow.log_params(grid_search.best_params_)
        
        # Log model
        if isinstance(best_model, xgb.XGBClassifier):
            mlflow.xgboost.log_model(best_model, "model")
        else:
            mlflow.sklearn.log_model(best_model, "model")
        
        # Register the model
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        mlflow.register_model(model_uri, f"{model_name}_Tuned")
        
        # Transition to Production
        client = mlflow.tracking.MlflowClient()
        model_version = client.get_latest_versions(f"{model_name}_Tuned")[0]
        client.transition_model_version_stage(
            name=f"{model_name}_Tuned",
            version=model_version.version,
            stage="Production"
        )
        
        logger.info(f"Successfully trained and registered {model_name}_Tuned")
        return best_model, metrics

def main():
    # Set experiment
    experiment_name = "Telco Churn Prediction"
    mlflow.set_experiment(experiment_name)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    # First, train baseline models
    print("\nTraining baseline models...")
    
    # Train and evaluate Logistic Regression
    print("\nTraining baseline Logistic Regression...")
    with mlflow.start_run(run_name="LogisticRegression_Baseline") as run:
        # Train model
        lr_model = train_logistic_regression(X_train, y_train)
        
        # Log parameters
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("class_weight", "balanced")
        
        # Set tags
        mlflow.set_tags({
            "model_type": "LogisticRegression",
            "model_stage": "baseline",
            "description": "Baseline Logistic Regression model with default parameters"
        })
        
        # Evaluate and log metrics
        lr_metrics = evaluate_model(y_test, lr_model.predict(X_test), lr_model.predict_proba(X_test)[:, 1])
        mlflow.log_metrics(lr_metrics)
        
        # Plot and log feature importance
        lr_importance = plot_feature_importance(lr_model, X_train.columns, "LogisticRegression")
        if lr_importance is not None:
            print("\nTop 5 features (Logistic Regression):")
            print(lr_importance.head())
        
        # Log artifacts
        mlflow.log_artifact("confusion_matrix_LogisticRegression.png")
        mlflow.log_artifact("roc_curve_LogisticRegression.png")
        mlflow.log_artifact("feature_importance_LogisticRegression.png")
        
        # Log model
        mlflow.sklearn.log_model(
            lr_model,
            "model",
            registered_model_name="LogisticRegression_Baseline"
        )
    
    # Train and evaluate Random Forest
    print("\nTraining baseline Random Forest...")
    with mlflow.start_run(run_name="RandomForest_Baseline") as run:
        # Train model
        rf_model = train_random_forest(X_train, y_train)
        
        # Log parameters
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 10)
        mlflow.log_param("min_samples_split", 5)
        mlflow.log_param("class_weight", "balanced")
        
        # Set tags
        mlflow.set_tags({
            "model_type": "RandomForest",
            "model_stage": "baseline",
            "description": "Baseline Random Forest model with default parameters"
        })
        
        # Evaluate and log metrics
        rf_metrics = evaluate_model(y_test, rf_model.predict(X_test), rf_model.predict_proba(X_test)[:, 1])
        mlflow.log_metrics(rf_metrics)
        
        # Plot and log feature importance
        rf_importance = plot_feature_importance(rf_model, X_train.columns, "RandomForest")
        print("\nTop 5 features (Random Forest):")
        print(rf_importance.head())
        
        # Log artifacts
        mlflow.log_artifact("confusion_matrix_RandomForest.png")
        mlflow.log_artifact("roc_curve_RandomForest.png")
        mlflow.log_artifact("feature_importance_RandomForest.png")
        
        # Log model
        mlflow.sklearn.log_model(
            rf_model,
            "model",
            registered_model_name="RandomForest_Baseline"
        )
    
    # Train and evaluate XGBoost
    print("\nTraining baseline XGBoost...")
    with mlflow.start_run(run_name="XGBoost_Baseline") as run:
        # Train model
        xgb_model = train_xgboost(X_train, y_train)
        
        # Log parameters
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 6)
        mlflow.log_param("learning_rate", 0.1)
        
        # Set tags
        mlflow.set_tags({
            "model_type": "XGBoost",
            "model_stage": "baseline",
            "description": "Baseline XGBoost model with default parameters"
        })
        
        # Evaluate and log metrics
        xgb_metrics = evaluate_model(y_test, xgb_model.predict(X_test), xgb_model.predict_proba(X_test)[:, 1])
        mlflow.log_metrics(xgb_metrics)
        
        # Plot and log feature importance
        xgb_importance = plot_feature_importance(xgb_model, X_train.columns, "XGBoost")
        print("\nTop 5 features (XGBoost):")
        print(xgb_importance.head())
        
        # Log artifacts
        mlflow.log_artifact("confusion_matrix_XGBoost.png")
        mlflow.log_artifact("roc_curve_XGBoost.png")
        mlflow.log_artifact("feature_importance_XGBoost.png")
        
        # Log model
        mlflow.sklearn.log_model(
            xgb_model,
            "model",
            registered_model_name="XGBoost_Baseline"
        )
    
    # Now perform hyperparameter tuning
    print("\nStarting hyperparameter tuning...")
    
    # Define models and their parameter grids
    models = {
        "LogisticRegression": {
            "model": LogisticRegression(random_state=42),
            "param_grid": {
                'C': [0.1, 1.0, 10.0],
                'max_iter': [1000],
                'solver': ['liblinear', 'saga']
            }
        },
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42),
            "param_grid": {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        },
        "XGBoost": {
            "model": xgb.XGBClassifier(random_state=42),
            "param_grid": {
                'n_estimators': [100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
        }
    }
    
    # Train and tune each model
    best_models = {}
    for model_name, model_config in models.items():
        logger.info(f"Training and tuning {model_name}...")
        best_model, metrics = train_and_tune_model(
            model_name,
            model_config["model"],
            model_config["param_grid"],
            X_train,
            y_train,
            X_test,
            y_test
        )
        best_models[model_name] = best_model
        logger.info(f"{model_name} metrics: {metrics}")
    
    print("\nAll models have been trained and tuned!")
    print("You can view the results at http://localhost:5000")

if __name__ == "__main__":
    main() 