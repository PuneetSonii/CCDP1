import os
import sys
import dill
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import recall_score
from src.logger import logging

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, f1_score
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for model_name, model in models.items():
            para = param[model_name]
            logging.info(f"Model fitting started for {model_name}")

            # Changed GridSearchCV to RandomizedSearchCV
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            logging.info(f"Model fitting done for {model_name}")

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate recall, precision, and f1-score
            train_f1 = f1_score(y_train, y_train_pred, average='binary')
            test_f1 = f1_score(y_test, y_test_pred, average='binary')

            # Log the metrics
            logging.info(f"Model: {model_name}")
            logging.info(f"Train F1 Score: {train_f1:.4f}")
            logging.info(f"Test F1 Score: {test_f1:.4f}")

            # Save the metrics to the report dictionary
            report[model_name] = {
                'train_f1': train_f1,
                'test_f1': test_f1
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
