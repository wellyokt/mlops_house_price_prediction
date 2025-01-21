import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
from src.utils.logger import default_logger as logger
from src.models.model import ModelFactory
from src.utils.config import config

class ModelTrainer:
    """Class for training and evaluating models"""
    
    def __init__(self, experiment_name: str = "house_price_prediction"):
        """
        Initialize ModelTrainer
        
        Args:
            experiment_name: Name of MLflow experiment
        """
        self.experiment_name = experiment_name
        self.models_info = {}
        self.best_model = None
        self.setup_mlflow()
        logger.info(f"Initialized ModelTrainer with experiment: {experiment_name}")
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate model evaluation metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        try:
            metrics = {
            'R2 Score': r2_score(y_true, y_pred),
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'MAPE': np.mean(np.abs((y_true - y_pred) / y_true))
            }

            return metrics
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            raise
    
    def setup_mlflow(self) -> None:
        """Setup MLflow tracking"""
        try:
            # Set MLflow tracking URI
            tracking_uri = config.get('mlflow.tracking_uri', 'sqlite:///mlflow.db')
            mlflow.set_tracking_uri(tracking_uri)
            
            # Create or get experiment
            try:
                self.experiment_id = mlflow.create_experiment(self.experiment_name)
            except:
                self.experiment_id = mlflow.get_experiment_by_name(self.experiment_name).experiment_id
            
            mlflow.set_experiment(self.experiment_name)
            logger.info("MLflow setup completed successfully")
            
        except Exception as e:
            logger.error(f"Error setting up MLflow: {str(e)}")
            raise
    
    def train_model(self, model_type: str, X_train: pd.DataFrame, y_train: pd.Series,
                    X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Train a single model
        
        Args:
            model_type: Type of model to train
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary containing model info and metrics
        """
        try:
            logger.info(f"Training {model_type} model")
            
            # Create and train model
            model = ModelFactory.create_model(model_type)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred)
            
            # Log with MLflow using nested runs
            with mlflow.start_run(run_name=model_type, nested=True) as run:
                # Log parameters and metrics
                mlflow.log_params(model.get_params())
                mlflow.log_metrics(metrics)
                
                # Log feature importance if available
                if hasattr(model, 'feature_importances_'):
                    feature_importance = dict(zip(X_train.columns, model.feature_importances_))
                    mlflow.log_params({"feature_importance": str(feature_importance)})
                
                # Log model
                mlflow.sklearn.log_model(
                    model,
                    model_type,
                    registered_model_name=f"house_price_{model_type}"
                )
            
            # Store model info
            model_info = {
                'model': model,
                'metrics': metrics,
                'run_id': run.info.run_id
            }
            self.models_info[model_type] = model_info
            
            logger.info(f"Completed training {model_type} model")
            logger.info(f"Metrics: {metrics}")
            
            return model_info
                
        except Exception as e:
            logger.error(f"Error training {model_type} model: {str(e)}")
            raise
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                        X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict[str, Any]]:
        """
        Train all configured models
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary containing info for all trained models
        """
        try:
            logger.info("Starting training of all models")
            
            for model_type in ModelFactory.get_model_config().keys():
                self.train_model(model_type, X_train, y_train, X_test, y_test)
            
            # Select best model
            self._select_best_model()
            
            logger.info("Completed training all models")
            return self.models_info
            
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            raise
            
    def _select_best_model(self) -> None:
        """Select best model based on recall and f1 score"""
        try:
            logger.info("Selecting best model")
            
            best_score = -1
            best_model_type = None
            
            for model_type, model_info in self.models_info.items():
                # Calculate combined score (average of recall and f1)
                metrics = model_info['metrics']
                combined_score = metrics['R2 Score']
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_model_type = model_type
            
            if best_model_type:
                self.best_model = self.models_info[best_model_type]
                
                # Transition best model to production in MLflow
                client = mlflow.tracking.MlflowClient()
                model_name = f"house_price_{best_model_type}"
                
                latest_versions = client.get_latest_versions(model_name)
                if latest_versions:
                    latest_version = latest_versions[0]
                    client.transition_model_version_stage(
                        name=model_name,
                        version=latest_version.version,
                        stage="Production"
                    )
                
                logger.info(f"Selected {best_model_type} as best model")
                logger.info(f"Best model metrics: {self.best_model['metrics']}")
            
        except Exception as e:
            logger.error(f"Error selecting best model: {str(e)}")
            raise
    
    def get_best_model(self) -> Dict[str, Any]:
        """Get best model info"""
        if self.best_model is None:
            raise ValueError("No best model selected. Train models first.")
        return self.best_model
    
    def get_all_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get metrics for all trained models"""
        return {model_type: info['metrics'] 
                for model_type, info in self.models_info.items()}