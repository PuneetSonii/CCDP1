# Import the necessary libraries
import os
import sys
from dataclasses import dataclass
import numpy as np
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models
from sklearn.model_selection import cross_val_score, StratifiedKFold

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info(f"Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Random Forest": RandomForestClassifier(
                    n_estimators=100, max_depth=10, random_state=42, class_weight="balanced"
                ),
                "Logistic Regression": LogisticRegression(
                    solver="liblinear", max_iter=1000, class_weight="balanced"
                ),
                "SVM": SVC(max_iter=500, random_state=42),
                "XGBoost": XGBClassifier(max_depth=5, random_state=42, scale_pos_weight=1),
                "Decision Tree": DecisionTreeClassifier(
                    random_state=42, class_weight="balanced"
                )
            }

            params = {
                "Random Forest": {
                    "n_estimators": [50, 100, 150],
                    "max_depth": [5, 10, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["sqrt", "log2"],
                },
                "XGBoost": {
                    "n_estimators": [50, 100, 150],
                    "max_depth": [3, 4, 5],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "subsample": [0.8, 0.9, 1.0],
                    "colsample_bytree": [0.8, 0.9, 1.0],
                },
                "Logistic Regression": {
                    "C": [0.1, 1, 10], 
                    "penalty": ["l2"]
                },
                "SVM": {
                    "C": [0.1, 1, 10], 
                    "kernel": ["linear", "poly", "rbf", "sigmoid"]
                },
                "Decision Tree": {
                    "max_depth": [None, 5, 10, 15],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                }
            }

            logging.info(f"Hyperparameter tuning started")
            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params,
            )
            logging.info(f"Hyperparameter tuning done")

            # Sort the models based on their performance (descending order)
            sorted_models = sorted(model_report.items(), key=lambda x: x[1]["test_f1"], reverse=True)

            # Log the models with automatically assigned numbers based on their score
            for i, (model_name, model_scores) in enumerate(sorted_models, start=1):
                logging.info(f"{i}. {model_name} - Train F1 Score: {model_scores['train_f1']:.4f}, Test F1 Score: {model_scores['test_f1']:.4f}")

            # Get the best model based on the test F1 score
            best_model_name, best_model_scores = sorted_models[0]
            best_model = models[best_model_name]  # Access the actual model from the dictionary

            if best_model_scores["test_f1"] < 0.6:
                raise CustomException(f"No best model found")

            logging.info(f"Best model on both training and testing dataset was: {best_model_name}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            f1 = f1_score(y_test, predicted)
            return f1

        except Exception as e:
            raise CustomException(e, sys)
