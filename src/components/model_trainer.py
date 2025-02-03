import os
import sys

from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
  AdaBoostRegressor,
  GradientBoostingRegressor,
  RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score

from src.logger import logging
from src.exception import CustomException

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
  trained_model_file_path = os.path.join("artifacts", "model.pkl")
  
class ModelTrainer:
  def __init__(self):
    self.model_trainer_config = ModelTrainerConfig()
    
  def initiate_model_trainer(self, train_array, test_array):
    try:
      logging.info("Split training and test input data")
      X_train, y_train, X_test, y_test = (
        train_array[:, :-1],
        train_array[:, -1],
        test_array[:, :-1],
        test_array[:, -1]
      )
      
      models = {
        "Linear Regression": LinearRegression(),
        "Gradiend Boosting Regressor": GradientBoostingRegressor(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest Regressor": RandomForestRegressor(),
        "K-Neighbors Regressor": KNeighborsRegressor(),
        "XGBRegressor": XGBRegressor(),
        "CatBoosting Regressor": CatBoostRegressor(verbose=False),
        "Adaboost Regressor": AdaBoostRegressor()
      }
      
      model_report: dict = evaluate_models(X_train, y_train, X_test, y_test, models)
      
      best_model_score = max(sorted(model_report.values()))
      
      best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
      
      best_model = models[best_model_name]
      
      if best_model_score < 0.6:
        raise Exception("No best model found")
      
      logging.info(f"Best Model Found , Model Name: {best_model_name} , R2 Score: {best_model_score}")
      
      save_object(
        file_path = self.model_trainer_config.trained_model_file_path,
        obj = best_model
      )
      
      predicted = best_model.predict(X_test)
      
      score = r2_score(y_test, predicted)
      
      return score
      
    except Exception as e:
      raise CustomException(e, sys)