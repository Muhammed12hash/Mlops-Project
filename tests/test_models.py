import pytest
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from src.models.train_and_tune_models import train_model

def test_model_training():
    # Create sample data
    X = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100)
    })
    y = np.random.randint(0, 2, 100)
    
    # Train model
    model = train_model(X, y, model_type='logistic_regression')
    
    # Make predictions
    predictions = model.predict(X)
    
    # Check predictions shape
    assert len(predictions) == len(y)
    
    # Check accuracy is reasonable
    accuracy = accuracy_score(y, predictions)
    assert 0 <= accuracy <= 1

def test_model_input_validation():
    # Test with invalid input
    with pytest.raises(ValueError):
        train_model(None, None, model_type='invalid_model') 