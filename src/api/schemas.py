from pydantic import BaseModel, Field
from typing import Dict, Optional, List


class ModelMetrics(BaseModel):
    """Model metrics schema"""
    r2_score: float = Field(..., description="Model R2 Score")
    mse: float = Field(..., description="Model MSE")
    rmse: float = Field(..., description="Model RMSE")
    mae: float = Field(..., description="Model MAE")
    mape: float = Field(..., description="Model MAPE")

class ModelInfo(BaseModel):
    """Model information schema"""
    run_id: str = Field(..., description="MLflow run ID")
    metrics: ModelMetrics = Field(..., description="Model metrics")


class HouseSaleRequest(BaseModel):
    Id: int = Field(..., description="Unique identifier for the house")
    MSSubClass: int = Field(..., description="The building class")
    MSZoning: str = Field(..., description="The general zoning classification")
    LotFrontage: float = Field(..., description="Linear feet of street connected to property")
    LotArea: int = Field(..., description="Lot size in square feet")
    Street: str = Field(..., description="Type of road access")
    LotShape: str = Field(..., description="General shape of property")
    LandContour: str = Field(..., description="Flatness of the property")
    Utilities: str = Field(..., description="Type of utilities available")
    LotConfig: str = Field(..., description="Lot configuration")
    LandSlope: str = Field(..., description="Slope of property")
    Neighborhood: str = Field(..., description="Physical locations within Ames city limits")
    Condition1: str = Field(..., description="Proximity to main road or railroad")
    Condition2: str = Field(..., description="Proximity to main road or railroad (if a second is present)")
    BldgType: str = Field(..., description="Type of dwelling")
    HouseStyle: str = Field(..., description="Style of dwelling")
    OverallQual: int = Field(..., description="Overall material and finish quality")
    OverallCond: int = Field(..., description="Overall condition rating")
    YearBuilt: int = Field(..., description="Original construction date")
    YearRemodAdd: int = Field(..., description="Remodel date")
    RoofStyle: str = Field(..., description="Type of roof")
    RoofMatl: str = Field(..., description="Roof material")
    Exterior1st: str = Field(..., description="Exterior covering on house")
    Exterior2nd: str = Field(..., description="Exterior covering on house (if more than one material)")
    MasVnrArea :float = Field(...,description="Masonry veneer area in square feet")
    ExterQual: str = Field(..., description="Exterior material quality")
    ExterCond: str = Field(..., description="Present condition of the material on the exterior")
    Foundation: str = Field(..., description="Type of foundation")
    BsmtQual: str= Field(..., description="Height of the basement")
    BsmtCond: str = Field(..., description="General condition of the basement")
    BsmtExposure: str = Field(..., description="Walkout or garden level basement walls")
    BsmtFinType1: str = Field(..., description="Quality of basement finished area")
    BsmtFinSF1: int = Field(..., description="Type 1 finished square feet")
    BsmtFinType2: str = Field(..., description="Quality of second finished area (if present)")
    BsmtFinSF2: int = Field(..., description="Type 2 finished square feet")
    BsmtUnfSF: int = Field(..., description="Unfinished square feet of basement area")
    TotalBsmtSF: int = Field(..., description="Total square feet of basement area")
    Heating: str = Field(..., description="Type of heating")
    HeatingQC: str = Field(..., description="Heating quality and condition")
    CentralAir: str = Field(..., description="Central air conditioning")
    Electrical: str= Field(..., description="Electrical system")
    onestFlrSF: int = Field(..., description="First Floor square feet")
    twondFlrSF: int = Field(..., description="Second floor square feet")
    LowQualFinSF: int = Field(..., description="Low quality finished square feet (all floors)")
    GrLivArea: int = Field(..., description="Above grade (ground) living area square feet")
    BsmtFullBath: int = Field(..., description="Basement full bathrooms")
    BsmtHalfBath: int = Field(..., description="Basement half bathrooms")
    FullBath: int = Field(..., description="Full bathrooms above grade")
    HalfBath: int = Field(..., description="Half baths above grade")
    BedroomAbvGr: int = Field(..., description="Number of bedrooms above basement level")
    KitchenAbvGr: int = Field(..., description="Number of kitchens")
    KitchenQual: str = Field(..., description="Kitchen quality")
    TotRmsAbvGrd: int = Field(..., description="Total rooms above grade (does not include bathrooms)")
    Functional: str = Field(..., description="Home functionality rating")
    Fireplaces: int = Field(..., description="Number of fireplaces")
    GarageType: str= Field(..., description="Garage location")
    GarageYrBlt: float = Field(..., description="Year garage was built")
    GarageFinish: str= Field(..., description="Interior finish of the garage")
    GarageCars: int = Field(..., description="Size of garage in car capacity")
    GarageArea: int = Field(..., description="Size of garage in square feet")
    GarageQual: str = Field(..., description="Garage quality")
    GarageCond: str = Field(..., description="Garage condition")
    PavedDrive: str = Field(..., description="Paved driveway")
    WoodDeckSF: int = Field(..., description="Wood deck area in square feet")
    OpenPorchSF: int = Field(..., description="Open porch area in square feet")
    EnclosedPorch: int = Field(..., description="Enclosed porch area in square feet")
    threeSsnPorch: int = Field(..., description="Three season porch area in square feet")
    ScreenPorch: int = Field(..., description="Screen porch area in square feet")
    PoolArea: int = Field(..., description="Pool area in square feet")
    MiscVal: int = Field(..., description="$Value of miscellaneous feature")
    MoSold: int = Field(..., description="Month Sold")
    YrSold: int = Field(..., description="Year Sold")
    SaleType: str = Field(..., description="Type of sale")
    SaleCondition: str = Field(..., description="Condition of sale")

    class Config:
        schema_extra = {
            "example": {
        "Id": 1,
        "MSSubClass": 60,
        "MSZoning": "RL",
        "LotFrontage": 65.0,
        "LotArea": 8450,
        "Street": "Pave",
        "LotShape": "Reg",
        "LandContour": "Lvl",
        "Utilities": "AllPub",
        "LotConfig": "Inside",
        "LandSlope": "Gtl",
        "Neighborhood": "CollgCr",
        "Condition1": "Norm",
        "Condition2": "Norm",
        "BldgType": "1Fam",
        "HouseStyle": "2Story",
        "OverallQual": 7,
        "OverallCond": 5,
        "YearBuilt": 2003,
        "YearRemodAdd": 2003,
        "RoofStyle": "Gable",
        "RoofMatl": "CompShg",
        "Exterior1st": "VinylSd",
        "Exterior2nd": "VinylSd",
        "MasVnrArea":190.0,
        "ExterQual": "Gd",
        "ExterCond": "TA",
        "Foundation": "PConc",
        "BsmtQual": "Gd",
        "BsmtCond": "TA",
        "BsmtExposure": "No",
        "BsmtFinType1": "GLQ",
        "BsmtFinSF1": 706,
        "BsmtFinType2": "Unf",
        "BsmtFinSF2": 0,
        "BsmtUnfSF": 150,
        "TotalBsmtSF": 856,
        "Heating": "GasA",
        "HeatingQC": "Ex",
        "CentralAir": "Y",
        "Electrical": "SBrkr",
        "onestFlrSF": 856,
        "twondFlrSF": 854,
        "LowQualFinSF": 0,
        "GrLivArea": 1710,
        "BsmtFullBath": 1,
        "BsmtHalfBath": 0,
        "FullBath": 2,
        "HalfBath": 1,
        "BedroomAbvGr": 3,
        "KitchenAbvGr": 1,
        "KitchenQual": "Gd",
        "TotRmsAbvGrd": 8,
        "Functional": "Typ",
        "Fireplaces": 0,
        "GarageType": "Attchd",
        "GarageYrBlt": 2003.0,
        "GarageFinish": "RFn",
        "GarageCars": 2,
        "GarageArea": 548,
        "GarageQual": "TA",
        "GarageCond": "TA",
        "PavedDrive": "Y",
        "WoodDeckSF": 0,
        "OpenPorchSF": 61,
        "EnclosedPorch": 0,
        "threeSsnPorch": 0,
        "ScreenPorch": 0,
        "PoolArea": 0,
        "MiscVal": 0,
        "MoSold": 2,
        "YrSold": 2008,
        "SaleType": "WD",
        "SaleCondition": "Normal"
        }
        }




class HousePredictionResponse(BaseModel):
    """House Price prediction response schema"""
    Id: int = Field(..., description="Id")
    SalePrice : int = Field(...,description ='Sale Price Prediction')