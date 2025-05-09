# Telco Customer Churn Prediction - MLOps Project

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![MLflow](https://img.shields.io/badge/MLflow-2.8.0-orange.svg)](https://mlflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100.0-blue.svg)](https://fastapi.tiangolo.com/)

<div align="center">
  <img src="docs/images/mlops-pipeline.png" alt="MLOps Pipeline" width="600"/>
</div>

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Technical Implementation](#technical-implementation)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## 🎯 Overview
This project implements an end-to-end machine learning pipeline for predicting customer churn in a telecommunications company. The solution includes model training, versioning, deployment, and serving through a REST API.

## ✨ Features
- 🤖 Multiple ML models (Logistic Regression, Random Forest, XGBoost)
- 🔄 Automated model training and hyperparameter tuning
- 📊 MLflow integration for experiment tracking and model versioning
- 🌐 FastAPI web service for model serving
- 📈 Comprehensive model evaluation metrics
- 🔌 RESTful API endpoints for predictions

## 📁 Project Structure
```
mlops-project/
├── data/                      # Data directory
│   └── telco_churn.csv       # Dataset
├── docs/                      # Documentation
│   └── images/               # Project images
├── mlruns/                    # MLflow tracking data
├── src/                       # Source code
│   ├── models/               # Model implementations
│   ├── utils/                # Utility functions
│   └── api/                  # API implementation
├── tests/                     # Test files
├── app_fastapi_new.py        # FastAPI application
├── train_and_tune_models.py  # Model training script
├── utils.py                  # Utility functions
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

## 📊 Model Performance
| Model               | Accuracy | Precision | Recall | F1 Score | ROC AUC |
| ------------------- | -------- | --------- | ------ | -------- | ------- |
| Logistic Regression | 80.5%    | 79.8%     | 81.2%  | 80.5%    | 0.82    |
| Random Forest       | 82.3%    | 83.1%     | 81.5%  | 82.3%    | 0.85    |
| XGBoost             | 83.7%    | 84.2%     | 83.1%  | 83.6%    | 0.87    |

## 🚀 Getting Started

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mlops-project.git
cd mlops-project
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

1. Train and tune models:
```bash
python train_and_tune_models.py
```

2. Start the API server:
```bash
python app_fastapi_new.py
```

## �� API Documentation

### Available Endpoints
- `POST /predict`: Get churn predictions from all models
- `GET /models`: List available models
- `GET /model/{model_name}`: Get specific model information

### Example Request
```python
import requests

# Make prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "customer_data": {
            # Add customer data here
        }
    }
)
print(response.json())
```

## 🔧 Technical Implementation

### Model Training Pipeline
- Automated training process
- Hyperparameter tuning using GridSearchCV
- Cross-validation for robust evaluation
- Model performance tracking

### Model Registry
- Version control for models
- Production/Staging environments
- Model metadata tracking
- Easy model rollback capability

### API Service
- Real-time predictions
- Input validation
- Error handling
- Model loading and caching
- Concurrent request handling

## 🔮 Future Improvements
1. Model Monitoring:
   - Add model performance monitoring
   - Implement data drift detection
   - Add automated retraining triggers

2. API Enhancements:
   - Add authentication
   - Implement rate limiting
   - Add more detailed model metrics
   - Add batch prediction endpoint

3. Infrastructure:
   - Containerize the application
   - Add CI/CD pipeline
   - Implement automated testing
   - Add monitoring and logging

4. Model Improvements:
   - Add more model types
   - Implement ensemble methods
   - Add feature importance analysis
   - Implement A/B testing capability

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact
Mohamed Rejjo - mohamed.rejjo@bahcesehir.edu.tr

Project Link: [https://github.com/Muhammed12hash/Mlops-Project](https://github.com/Muhammed12hash/Mlops-Project) 