import sys
import os
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.exception import CustomException
from src.log import logging 
from src.utils import save_object
@dataclass
class DataTransformationConfig:
    preprocesser_obj_file_path = os.path.join('artifacts', 'preprocesser.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()  

    def get_data_transformer_object(self):
        '''
        function is responsible for returning the data transformer object
        '''
        try:
            numerical_columns = [
                'writing_score',
                'reading_score'
            ]
            categorical_columns = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]

            numerical_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            categorical_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            logging.info("Numerical columns: {numerical_columns}")
            logging.info("Categorical columns: {categorical_columns}")

            preprocessor = ColumnTransformer(
                transformers = [
                    ('num_pipeline', numerical_pipeline, numerical_columns),
                    ('cat_pipeline', categorical_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read the train and test data successfully")
            logging.info("Obtaining the data transformer object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = 'math_score'

            input_feature_train_df = train_df.drop(columns = [target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]  

            input_feature_test_df = test_df.drop(columns = [target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe"
            )

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.fit_transform(input_feature_test_df) 
        
            train_arr = np.c_[input_feature_train_arr,
                              np.array(target_feature_train_df)]
            
            test_arr = np.c_[input_feature_test_arr,
                             np.array(target_feature_test_df)]
            
            logging.info(f"saved preprocessing object.")

            save_object(
                file_path = self.data_transformation_config.preprocesser_obj_file_path,
                obj = preprocessing_obj
            )
            
            return (
               train_arr,
               test_arr,
               self.data_transformation_config.preprocesser_obj_file_path 
            )
        except Exception as e:
            raise CustomException(e, sys)

            