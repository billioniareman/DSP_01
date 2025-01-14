import pandas as pd
import numpy as np
from dataclasses import dataclass
import os
import sys

from catboost import CatBoostRegressor

from sklearn.ensemble import (
    AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from  xgboost import XGBRegressor


from src.exception import CustomException
from src.logger import logging
from src.utlis import save_object,evaluate_model



@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifact',"model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Spliting Train and test data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
                'Random Forest Regressor':RandomForestRegressor(),
                'Decision Tree Regressor':DecisionTreeRegressor(),
                'Gradient Boosting Regressor':GradientBoostingRegressor(),
                'Linear Regression':LinearRegression(),
                'K Neighbors Regressor':KNeighborsRegressor(),
                'Cat Boost Regressor':CatBoostRegressor(),
                'AdaBoost Regressor':AdaBoostRegressor(),
                'XGB Regressor':XGBRegressor(),
            }
            model_Report:dict = evaluate_model(X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test,models=models)

            best_model_score=max(sorted(model_Report.values()))

            best_model_name=list(model_Report.keys())[list(model_Report.values()).index(best_model_score)]
            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException('NO best model found')
            logging.info("best model found on both train and test datasets")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted=best_model.predict(X_test)
            r2_s=r2_score(y_test,predicted)
            return r2_s

        except Exception as e:
            CustomException(e,sys)