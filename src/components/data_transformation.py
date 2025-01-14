import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utlis import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifact', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            nf = ['writing_score', 'reading_score']
            cf = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='median')),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='most_frequent')),
                    ("one_hot_encode", OneHotEncoder(sparse_output=False)),
                    ("standardscaler",StandardScaler())
                ]
            )

            logging.info('Numerical column standard scaling completed.')
            logging.info('Categorical column encoding completed.')

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, nf),
                    ("cat_pipeline", cat_pipeline, cf)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(f"Error in get_data_transformer_object: {str(e)}", sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            if not os.path.exists(train_path) or not os.path.exists(test_path):
                raise FileNotFoundError("Train or test file path is invalid.")
            
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test datasets loaded successfully.")

            target_column_name = "math_score"
            input_features = ['writing_score', 'reading_score']
            preprocessing_obj = self.get_data_transformer_object()

            # Validate presence of target column
            if target_column_name not in train_df.columns or target_column_name not in test_df.columns:
                raise ValueError(f"Target column '{target_column_name}' not found in dataset.")

            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]

            logging.info("Preprocessing training and testing data.")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, target_feature_train_df.to_numpy()]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_df.to_numpy()]

            logging.info("Saving preprocessing object.")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(f"Error in initiate_data_transformation: {str(e)}", sys)
