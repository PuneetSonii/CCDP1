import sys
import os
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.preprocessing import OneHotEncoder


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

from imblearn.under_sampling import RandomUnderSampler
# modelling
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

from collections import Counter

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        numerical_columns = ["LIMIT_BAL", "AGE", "PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
                                "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
                                "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]

        categorical_columns = ["SEX", "EDUCATION", "MARRIAGE"]

        logging.info('Pipeline Initiated')

        # Define the custom ranking for each ordinal variable
        education_categories =['Graduation','University','High_School','Others']
        marriage_categories = ['Married', 'Single', 'Others']

        logging.info('Pipeline Initiated')
        print(f"Education Categories: {education_categories}")
        print(f"Marriage Categories: {marriage_categories}")
        num_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=False))
            ]
        )

        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehotencoder", OneHotEncoder(drop='first')),
                ("scaler",StandardScaler(with_mean=False))
            ]
        )

        logging.info(f"Categorical columns: {categorical_columns}")
        logging.info(f"Ordinal Categorical columns: {education_categories,marriage_categories}")

        logging.info(f"Numerical columns: {numerical_columns}")

        preprocessor = ColumnTransformer(
            [
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipelines", cat_pipeline, categorical_columns),
                
            ]
        )

        return preprocessor

    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info("obtaining preprocessing object")

            # Replace SMOTE with RandomUnderSampler
            random_under_sampler = RandomUnderSampler(random_state=42)

            target_column_name = "default"

            numerical_columns = ["LIMIT_BAL", "AGE", "PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
                                  "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
                                  "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")

            preprocessing_obj = self.get_data_transformer_object()

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)

            logging.info(f"Shape of original training data: {input_feature_train_arr.shape}")
            # Log the class distribution before random under-sampling for training data
            logging.info("original training data of Class distribution OF DEFAULT column training data: %s", Counter(target_feature_train_df))

            # Apply random under-sampling to handle imbalanced data for training data
            X_train_res, y_train_res = random_under_sampler.fit_resample(input_feature_train_arr,
                                                                            target_feature_train_df)

            logging.info("After random under-sampling for training data:")
            logging.info(f"Shape of balanced X_train_res after Random under Sampling: {X_train_res.shape}")
            logging.info(f"Shape of balanced y_train_res after Random under Sampling: {y_train_res.shape}")
            logging.info("Class distribution of DEFAULT column after random under-sampling for training data: %s", Counter(y_train_res))

            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Before random under-sampling for test data:")
            logging.info("Class distribution before random under-sampling: %s", Counter(target_feature_test_df))

            # Apply random under-sampling to handle imbalanced data for test data
            X_test_res, y_test_res = random_under_sampler.fit_resample(input_feature_test_arr,
                                                                        target_feature_test_df)

            logging.info("After random under-sampling for test data:")
            logging.info(f"Shape of balanced X_test_res after Random under Sampling: {X_test_res.shape}")
            logging.info(f"Shape of balanced y_test_res after Random under Sampling: {y_test_res.shape}")
            logging.info("Class distribution after random under-sampling for test data: %s", Counter(y_test_res))


            # Plot the distribution of the target column after random under-sampling
            plt.figure(figsize=(8, 6))
            sns.countplot(x=target_feature_test_df)
            plt.title(f'Distribution of {target_column_name} after Random Under-Sampling')
            plt.show()

            train_arr = np.c_[
                X_train_res, np.array(y_train_res)
            ]

            test_arr = np.c_[X_test_res, np.array(y_test_res)]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)