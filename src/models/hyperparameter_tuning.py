import mlflow
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from utils import load_data, preprocess_data
from mlflow_config.mlflow_config import MLFLOW_TRACKING_URI, EXPERIMENTS
from sklearn.model_selection import train_test_split

# Set MLflow tracking URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def get_data():
    """Load and preprocess data."""
    df = load_data()
    df_processed = preprocess_data(df)
    X = df_processed.drop('Churn', axis=1)
    y = df_processed['Churn']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Define search spaces for each model
logistic_space = {
    'C': hp.loguniform('C', np.log(0.001), np.log(10.0)),
    'max_iter': hp.choice('max_iter', [500, 1000, 1500, 2000]),
    'solver': hp.choice('solver', ['lbfgs', 'liblinear', 'saga'])
}

random_forest_space = {
    'n_estimators': hp.choice('n_estimators', [100, 200, 300, 400, 500]),
    'max_depth': hp.choice('max_depth', [5, 7, 10, 15, 20]),
    'min_samples_split': hp.quniform('min_samples_split', 2, 20, 1),  # Integer values from 2 to 20
    'min_samples_leaf': hp.choice('min_samples_leaf', [1, 2, 4, 8])
}

xgboost_space = {
    'n_estimators': hp.choice('n_estimators', [100, 200, 300, 400, 500]),
    'max_depth': hp.choice('max_depth', [3, 4, 5, 6, 7, 8]),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
    'subsample': hp.uniform('subsample', 0.6, 1.0),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
    'min_child_weight': hp.quniform('min_child_weight', 1, 7, 1)  # Integer values from 1 to 7
}

def objective_logistic(params):
    """Objective function for Logistic Regression."""
    with mlflow.start_run(nested=True):
        model_params = {
            'C': float(params['C']),
            'max_iter': int(params['max_iter']),
            'solver': params['solver'],
            'class_weight': 'balanced',
            'random_state': 42
        }
        
        model = LogisticRegression(**model_params)
        
        # Get cross-validation score
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        mean_auc = scores.mean()
        
        # Log parameters and metrics
        mlflow.log_params(model_params)
        mlflow.log_metric('mean_cv_auc', mean_auc)
        
        return {'loss': -mean_auc, 'status': STATUS_OK}

def objective_random_forest(params):
    """Objective function for Random Forest."""
    with mlflow.start_run(nested=True):
        model_params = {
            'n_estimators': int(params['n_estimators']),
            'max_depth': int(params['max_depth']),
            'min_samples_split': int(params['min_samples_split']),
            'min_samples_leaf': int(params['min_samples_leaf']),
            'class_weight': 'balanced',
            'random_state': 42
        }
        
        model = RandomForestClassifier(**model_params)
        
        # Get cross-validation score
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        mean_auc = scores.mean()
        
        # Log parameters and metrics
        mlflow.log_params(model_params)
        mlflow.log_metric('mean_cv_auc', mean_auc)
        
        return {'loss': -mean_auc, 'status': STATUS_OK}

def objective_xgboost(params):
    """Objective function for XGBoost."""
    with mlflow.start_run(nested=True):
        model_params = {
            'n_estimators': int(params['n_estimators']),
            'max_depth': int(params['max_depth']),
            'learning_rate': float(params['learning_rate']),
            'subsample': float(params['subsample']),
            'colsample_bytree': float(params['colsample_bytree']),
            'min_child_weight': int(params['min_child_weight']),
            'random_state': 42
        }
        
        model = xgb.XGBClassifier(**model_params)
        
        # Get cross-validation score
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        mean_auc = scores.mean()
        
        # Log parameters and metrics
        mlflow.log_params(model_params)
        mlflow.log_metric('mean_cv_auc', mean_auc)
        
        return {'loss': -mean_auc, 'status': STATUS_OK}

def optimize_model(objective, space, experiment_name, max_evals=50):
    """Optimize model hyperparameters."""
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name="hyperparameter_optimization"):
        trials = Trials()
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials
        )
        
        # Get the best parameters with their actual values
        best_params = space_eval(space, best)
        return best_params, trials

if __name__ == "__main__":
    # Load data
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = get_data()
    
    # Optimize Logistic Regression
    print("\nOptimizing Logistic Regression...")
    best_lr, trials_lr = optimize_model(
        objective_logistic,
        logistic_space,
        EXPERIMENTS["logistic_regression"],
        max_evals=50
    )
    print("Best Logistic Regression parameters:", best_lr)
    
    # Optimize Random Forest
    print("\nOptimizing Random Forest...")
    best_rf, trials_rf = optimize_model(
        objective_random_forest,
        random_forest_space,
        EXPERIMENTS["random_forest"],
        max_evals=50
    )
    print("Best Random Forest parameters:", best_rf)
    
    # Optimize XGBoost
    print("\nOptimizing XGBoost...")
    best_xgb, trials_xgb = optimize_model(
        objective_xgboost,
        xgboost_space,
        EXPERIMENTS["xgboost"],
        max_evals=50
    )
    print("Best XGBoost parameters:", best_xgb) 