import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
from statistics import mode

from typing import Tuple, Dict, Optional
from pathlib import Path
from src.utils.logger import default_logger as logger
from src.utils.config import config

class DataProcessor:
    """Data preprocessing pipeline"""
    
    def __init__(self, preprocessing_path: Optional[str] = None):
        """
        Initialize data processor
        
        Args:
            preprocessing_path: Path to save/load preprocessing objects
        """
        self.preprocessing_path = preprocessing_path or config.get('preprocessing_path', 'models/preprocessing')
        self.encoders: Dict[str, LabelEncoder] = {}
        self.scaler = MinMaxScaler()
        self.trained = False
        logger.info("Initialized DataProcessor")
    
    def _prepare_preprocessing_path(self) -> None:
        """Create preprocessing directory if it doesn't exist"""
        Path(self.preprocessing_path).mkdir(parents=True, exist_ok=True)
    
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values by filling them"""
        # Drop columns with more than 20% missing values
        missing_percentage = (df.isnull().sum() / len(df)) * 100
        columns_to_drop = missing_percentage[missing_percentage > 20].index
        df = df.drop(columns=columns_to_drop)

        # Handling missing values in numeric columns (fill with median)
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            median = df[col].dropna().median()
            df[col] = df[col].fillna(median)
        
        # Handling missing values in categorical columns (fill with mode)
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            mode_value = mode(df[col].dropna())  # Drop NaN before calculating mode
            df[col] = df[col].fillna(mode_value)
        
        return df
    

    def fit_transform(self, df: pd.DataFrame, target_col: str = 'SalePrice') -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Fit preprocessors and transform data
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            
        Returns:
            Tuple of transformed features and target
        """
        try:
            logger.info("Starting fit_transform process")
            
            
            # Split features and target
            X = df.drop(columns=['Id', target_col], errors='ignore')
            y = df[target_col] if target_col in df.columns else None
            
            numerical_cols =  X.select_dtypes(include=['int64','float64']).columns.to_list()
            categorical_cols = X.select_dtypes(include=['object']).columns

            X = self.handle_missing_values(X)

            numerical_cols =  X.select_dtypes(include=['int64','float64']).columns.to_list()
            categorical_cols = X.select_dtypes(include=['object']).columns
            # Process categorical columns
            for col in categorical_cols:
                logger.info(f"Fitting LabelEncoder for column: {col}")
                self.encoders[col] = LabelEncoder()
                X[col] = self.encoders[col].fit_transform(X[col])
            
            # Process numerical columns
            logger.info("Fitting StandardScaler for numerical columns")
            X[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])
            
            self.trained = True
            logger.info("Fit_transform completed successfully")
            
            if y is not None:
                y = LabelEncoder().fit_transform(y)
            return X, y
            
        except Exception as e:
            logger.error(f"Error in fit_transform: {str(e)}")
            raise
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted preprocessors
        
        Args:
            df: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        if not self.trained:
            raise ValueError("DataProcessor not fitted. Call fit_transform first.")
            
        try:
            logger.info("Starting transform process")
            
            print('==============')
            print(self.scaler.feature_names_in_)
            # Create a copy to avoid modifying original data
            X = df.copy()
            
            # Remove Id if present
            if 'Id' in X.columns:
                X = X.drop('Id', axis=1)
                        
            numerical_cols =  ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond',
                            'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
                            'BsmtUnfSF', 'TotalBsmtSF', 'onestFlrSF', 'twondFlrSF', 'LowQualFinSF',
                            'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
                            'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt',
                            'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
                            'threeSsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']

            # Transform categorical columns
            for col, encoder in self.encoders.items():
                X[col] = encoder.transform(X[col])
            
            # Transform numerical columns
            X[numerical_cols] = self.scaler.transform(X[numerical_cols])
            # X[selfscaler.feature_names_in] = self.scaler.transform(X[scaler.feature_names_in])

            
            logger.info("Transform completed successfully")
            return X
            
        except Exception as e:
            logger.error(f"Error in transform: {str(e)}")
            raise
    
    def save_preprocessors(self) -> None:
        """Save preprocessor objects"""
        try:
            logger.info(f"Saving preprocessors to {self.preprocessing_path}")
            self._prepare_preprocessing_path()
            
            # Save encoders
            joblib.dump(
                self.encoders,
                Path(self.preprocessing_path) / 'encoders.joblib'
            )
            
            # Save scaler
            joblib.dump(
                self.scaler,
                Path(self.preprocessing_path) / 'scaler.joblib'
            )
            
            logger.info("Preprocessors saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving preprocessors: {str(e)}")
            raise
    
    def load_preprocessors(self) -> None:
        """Load preprocessor objects"""
        try:
            logger.info(f"Loading preprocessors from {self.preprocessing_path}")
            
            # Load encoders
            encoders_path = Path(self.preprocessing_path) / 'encoders.joblib'
            self.encoders = joblib.load(encoders_path)
            
            # Load scaler
            scaler_path = Path(self.preprocessing_path) / 'scaler.joblib'
            self.scaler = joblib.load(scaler_path)
            
            self.trained = True
            logger.info("Preprocessors loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading preprocessors: {str(e)}")
            raise