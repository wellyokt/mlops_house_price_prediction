# House Price Prediction

This project implements an end-to-end machine learning pipeline for predicting House Price. It includes data processing, model training, MLflow tracking, and model serving via FastAPI.

## Project Structure
```
assigment_deployment/
|── artifacts/
|   |── train.csv
├── src/
|   |──run_pipeline.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py      # Data loading utilities
│   │   └── data_processor.py   # Data preprocessing pipeline
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model.py           # Model architecture definitions
│   │   └── trainer.py         # Training logic
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger.py          # Logging configuration
│   │   └── config.py          # Configuration management
│   └── api/
│       ├── __init__.py
│       ├── main.py            # FastAPI application
│       └── schemas.py         # Pydantic models
├── notebooks/
│   └── research.ipynb
├── config/
│   └── config.yaml
├── requirements.txt
└── README.md
|__ 
```

## Features

- Data preprocessing pipeline with scikit-learn
- Multiple tree-based models (Decision Tree Regressor, Random Forest Regressor, XGB Regressor)
- MLflow experiment tracking and model registry
- Model serving via FastAPI
- Comprehensive logging system
- Configuration management
- Production-ready code structure

## Installation

1. Clone the repository:
```bash
git https://github.com/wellyokt/mlops_house_price_prediction.git
cd mlops_house_price_prediction
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

Update `config/config.yaml` with your settings:
```yaml
# Data Configuration
data_path: "artifacts/train.csv"
preprocessing_path: "models/preprocessing"

# MLflow Configuration
mlflow:
  tracking_uri: "sqlite:///mlflow.db"
  experiment_name: "House_price_Prediction"

# Model Parameters
model_params:
  random_forest:
    n_estimators: 100
    max_depth: 10
    random_state: 42
```

## Usage

### Running the Pipeline

1. Place your data file in the data directory:
```bash
cp path/to/your/train.csv data/
```

2. Run the training pipeline:
```bash
python src/run_pipeline.py
```

3. View experiments in MLflow:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### Starting the API

1. Start the FastAPI server:
```bash
uvicorn src.api.main:app --reload
```

2. Access the API documentation:
```
http://localhost:8000/docs
```

## API Endpoints

- `POST /predict`: Make churn predictions
- `GET /health`: Health check endpoint
- `GET /model-info`: Get current model information


## Docker 
```Bash
docker pull python:3.10-slim
```

```Bash
docker build -t mlops -f Dockerfile .
```
```Bash
docker run mlops
```


## Model Training

The pipeline trains three types of models:
- Decision Tree Regressor
- Random Forest Regressor
- XGB Regressor

The best model is selected based on a R2 Score.

## Monitoring

- Logs are stored in the `logs/` directory
- MLflow tracking information is stored in `mlflow.db`
- Model artifacts are stored in `models/`

## Development

### Running Tests
```bash
pytest tests/
```

### Adding New Models

1. Update `src/models/model.py` with your new model configuration
2. Add model-specific parameters in `config.yaml`
3. Update the training pipeline if necessary

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
