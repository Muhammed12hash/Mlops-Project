from flask import Flask, render_template, request, jsonify
import mlflow
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

app = Flask(__name__)

# Set up MLflow - use PostgreSQL backend directly
os.environ['MLFLOW_TRACKING_URI'] = "postgresql://mlflow_user:mlflow_password@localhost:5432/mlflow_db"
mlflow.set_tracking_uri("postgresql://mlflow_user:mlflow_password@localhost:5432/mlflow_db")

def load_production_models():
    models = {}
    scaler = None
    
    try:
        # Load the scaler from the training data
        df = pd.read_csv('data/customer_churn.csv')
        df = df.drop('customerID', axis=1)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df = df.dropna()
        
        # Get numeric columns for scaling
        numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
        scaler = StandardScaler()
        scaler.fit(df[numeric_features])
        
        # Load models from MLflow
        client = mlflow.tracking.MlflowClient()
        
        # Get experiments
        experiments = client.search_experiments()
        print("Available experiments:", [exp.name for exp in experiments])
        
        model_configs = {
            'logistic_regression': 'Logistic Regression',
            'random_forest': 'Random Forest',
            'xgboost': 'XGBoost'
        }
        
        for model_name, exp_name in model_configs.items():
            try:
                # Get the experiment
                experiment = client.get_experiment_by_name(exp_name)
                if experiment is None:
                    print(f"Experiment {exp_name} not found")
                    continue
                
                print(f"Found experiment {exp_name} with id {experiment.experiment_id}")
                
                # Get all runs for the experiment
                runs = client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    filter_string="",
                    order_by=["metrics.roc_auc DESC"]
                )
                
                if not runs:
                    print(f"No runs found for {model_name}")
                    continue
                
                best_run = runs[0]
                print(f"Loading {model_name} from run {best_run.info.run_id}")
                
                # Load the model from the run
                model_uri = f"runs:/{best_run.info.run_id}/model"
                print(f"Loading model from: {model_uri}")
                
                # Try different model flavors
                try:
                    model = mlflow.sklearn.load_model(model_uri)
                except Exception as e1:
                    print(f"Failed to load as sklearn model: {str(e1)}")
                    try:
                        model = mlflow.pyfunc.load_model(model_uri)
                    except Exception as e2:
                        print(f"Failed to load as pyfunc model: {str(e2)}")
                        raise Exception(f"Could not load model: {str(e1)}, {str(e2)}")
                
                models[model_name] = model
                print(f"Successfully loaded {model_name}")
                
            except Exception as e:
                print(f"Error loading {model_name}: {str(e)}")
    
    except Exception as e:
        print(f"Error in load_production_models: {str(e)}")
    
    return models, scaler

# Initialize models and scaler
print("Loading models...")
MODELS, SCALER = load_production_models()
print("Models loaded successfully!")
print("Available models:", list(MODELS.keys()))

def create_feature_columns():
    """Create a list of all possible feature columns after one-hot encoding."""
    # Base features that don't need encoding
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    # Categorical features and their possible values
    categorical_mappings = {
        'gender': ['Female', 'Male'],
        'SeniorCitizen': [0, 1],
        'Partner': ['No', 'Yes'],
        'Dependents': ['No', 'Yes'],
        'PhoneService': ['No', 'Yes'],
        'MultipleLines': ['No', 'No phone service', 'Yes'],
        'InternetService': ['DSL', 'Fiber optic', 'No'],
        'OnlineSecurity': ['No', 'No internet service', 'Yes'],
        'OnlineBackup': ['No', 'No internet service', 'Yes'],
        'DeviceProtection': ['No', 'No internet service', 'Yes'],
        'TechSupport': ['No', 'No internet service', 'Yes'],
        'StreamingTV': ['No', 'No internet service', 'Yes'],
        'StreamingMovies': ['No', 'No internet service', 'Yes'],
        'Contract': ['Month-to-month', 'One year', 'Two year'],
        'PaperlessBilling': ['No', 'Yes'],
        'PaymentMethod': ['Bank transfer (automatic)', 'Credit card (automatic)', 
                         'Electronic check', 'Mailed check']
    }
    
    # Create column names for one-hot encoded features
    feature_columns = numeric_features.copy()
    
    # Add SeniorCitizen as is (it's already binary)
    feature_columns.append('SeniorCitizen')
    
    # Add one-hot encoded features
    for feature, values in categorical_mappings.items():
        if feature != 'SeniorCitizen':  # Skip SeniorCitizen as we added it above
            feature_columns.extend([f"{feature}_{value}" for value in values])
    
    return feature_columns

def preprocess_input(data):
    """Preprocess input data to match model requirements."""
    try:
        # Convert single row to DataFrame
        df = pd.DataFrame([data])
        print("Input data:", df)
        
        # Convert SeniorCitizen to int
        df['SeniorCitizen'] = df['SeniorCitizen'].astype(int)
        
        # Convert numeric columns
        numeric_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col])
        
        # Scale numeric features
        if SCALER is not None:
            df[numeric_columns] = SCALER.transform(df[numeric_columns])
        
        # Get all possible feature columns
        all_features = create_feature_columns()
        print("Expected features:", len(all_features), all_features)
        
        # One-hot encode categorical columns
        categorical_columns = [col for col in df.columns if col not in numeric_columns and col != 'SeniorCitizen']
        df_encoded = pd.get_dummies(df, columns=categorical_columns)
        
        # Ensure all expected columns are present
        for col in all_features:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        
        # Reorder columns to match training data
        df_encoded = df_encoded[all_features]
        print("Processed features:", len(df_encoded.columns), df_encoded.columns.tolist())
        print("Feature values:", df_encoded.values)
        
        return df_encoded
    
    except Exception as e:
        print(f"Error in preprocess_input: {str(e)}")
        raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data
        data = request.json
        print("Received data:", data)
        
        if not MODELS:
            return jsonify({
                'success': False,
                'error': 'No models available for prediction'
            })
        
        # Preprocess input
        processed_data = preprocess_input(data)
        print("Processed features:", processed_data.columns.tolist())
        
        # Get predictions from all models
        predictions = {}
        probabilities = {}
        
        for name, model in MODELS.items():
            try:
                if hasattr(model, 'predict_proba'):
                    pred_proba = model.predict_proba(processed_data)[0][1]  # Get probability of class 1
                else:
                    # For models that don't have predict_proba
                    pred_proba = float(model.predict(processed_data)[0])
                
                pred = 1 if pred_proba >= 0.5 else 0
                predictions[name] = "Likely to Churn" if pred == 1 else "Likely to Stay"
                probabilities[name] = float(pred_proba * 100)  # Convert to percentage
                print(f"{name} prediction:", predictions[name], "probability:", probabilities[name])
            except Exception as e:
                print(f"Error making prediction with {name}:", str(e))
                predictions[name] = "Error"
                probabilities[name] = 0.0
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'probabilities': probabilities
        })
        
    except Exception as e:
        print("Error in predict route:", str(e))
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True, port=5003) 