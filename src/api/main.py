from fastapi import FastAPI, HTTPException
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import os
from datetime import datetime
from src.utils.logger import default_logger as logger
from src.data.data_processor import DataProcessor
from src.api.schemas import (
    HouseSaleRequest,
    HousePredictionResponse,
    ModelInfo,
    ModelMetrics
)
import pickle

app = FastAPI(
    title="House Price Prediction",
    description="API for predicting House Price",
    version="1.0.0"
)

def get_latest_model_path() -> tuple[str, ModelInfo]:
    """
    Get the path to the latest trained model
    
    Returns:
        Tuple containing model path and model info
    """
    client = MlflowClient()
    
    # Get experiment
    experiment = client.get_experiment_by_name('houseprice_prediction')
    if not experiment:
        raise ValueError("No experiment found with name 'houseprice_prediction'")
    print(experiment)

    logger.info(f"Found experiment with ID: {experiment.experiment_id}")
    
    # Get all runs
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.recall DESC", "metrics.f1 DESC"]
    )
    
    logger.info(f"Found {len(runs)} runs")
    
    if not runs:
        raise ValueError("No runs found in the experiment")
    
    # Find best run based on average of recall and f1
    best_run = None
    best_score = -1
    
    for run in runs:
        metrics = run.data.metrics
        if  'R2 Score'in metrics:
            combined_score = metrics['R2 Score']
            logger.info(f"Run {run.info.run_id} score: {combined_score}")
            if combined_score > best_score:
                best_score = combined_score
                best_run = run
                
    if not best_run:
        raise ValueError("No valid runs found with required metrics")
    
    # Get model path
    run_id = best_run.info.run_id
    logger.info(f"Best run ID: {run_id}")
    
    # Try to load model by run ID
    try:
        # Load directly from run ID and model name
        model_path = f"runs:/{run_id}/gradient_boosting"
        logger.info(f"Trying to load model from: {model_path}")
        model = mlflow.pyfunc.load_model(model_path)
        logger.info(f"Successfully loaded model from: {model_path}")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        # Try alternative path with 'model' artifact name
        try:
            model_path = f"runs:/{run_id}/model"
            logger.info(f"Trying alternative path: {model_path}")
            model = mlflow.pyfunc.load_model(model_path)
            logger.info(f"Successfully loaded model from alternative path")
        except Exception as e2:
            logger.error(f"Error loading from alternative path: {str(e2)}")
            # Try local filesystem path
            try:
                local_path = os.path.join("mlruns", experiment.experiment_id, 
                                        run_id, "artifacts/model")
                logger.info(f"Trying local filesystem path: {local_path}")
                if not os.path.exists(local_path):
                    raise ValueError(f"Local path does not exist: {local_path}")
                model = mlflow.pyfunc.load_model(local_path)
                model_path = local_path
                logger.info(f"Successfully loaded model from local path")
            except Exception as e3:
                logger.error(f"Error loading from local path: {str(e3)}")
                raise ValueError(f"Could not load model from any path. Errors:\n"
                               f"Primary: {str(e)}\n"
                               f"Alternative: {str(e2)}\n"
                               f"Local: {str(e3)}")
    
    # Create model info
    metrics = ModelMetrics(
        r2_score=best_run.data.metrics.get('R2 Score', 0.0),
        mse=best_run.data.metrics.get('MSE', 0.0),
        rmse=best_run.data.metrics.get('RMSE', 0.0),
        mae=best_run.data.metrics.get('MAE', 0.0),
        mape=best_run.data.metrics.get('MAPE', 0.0)
    )
    
    model_info = ModelInfo(run_id=run_id, metrics=metrics)
    return model_path, model_info

@app.on_event("startup")
async def startup_event():
    """Load model and preprocessor on startup"""
    global model, preprocessor, model_info
    try:
        logger.info("Loading best model from MLflow")
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri('sqlite:///mlflow.db')
        
        # Find and load best model
        # model_path, model_info = get_latest_model_path()
        # logger.info(f"Loading model from path: {model_path}")
        # model = mlflow.pyfunc.load_model(model_path)

        # Load the model using the correct method

        model_path  =r"mlruns/1/32a49ce7a4174c87b637115323754eb6/artifacts/xgboost/model.pkl"
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Initialize and load preprocessor
        preprocessor = DataProcessor()
        preprocessor.load_preprocessors()
        
        logger.info(f"Model and preprocessor loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "House Price Prediction API",
        # "model_info": model_info,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=HousePredictionResponse)
async def predict(request: HouseSaleRequest):
    """
    House Price Prediction

    Args:
        request: Prediction request containing customer data
        
    Returns:
        Prediction response with House Price and model info
    """
    try:
        logger.info(f"Received prediction request for customer: {request.Id}")
        
        # Convert request to DataFrame
        data = pd.DataFrame([request.dict()])
        print(data.columns)


        
        # # Preprocess data
        processed_data = preprocessor.transform(data)

        # print(processed_data)
        
        # Make prediction
        prediction = model.predict(processed_data )
    

        
        response = HousePredictionResponse(
            Id=request.Id,
            housepriceprediction=prediction
        )
        
        logger.info(f"Prediction completed for customer: {request.Id}")
        return response
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_info": model_info
    }