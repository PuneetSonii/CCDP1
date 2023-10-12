import os
import sys
import dill
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import recall_score
from src.logger import logging

from sklearn.model_selection import RandomizedSearchCV  # Changed import
from sklearn.metrics import r2_score

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

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]
            logging.info(f"model fitting started")
            
            # Changed GridSearchCV to RandomizedSearchCV
            rs = RandomizedSearchCV(model, para, cv=3, n_iter=3, random_state=42)
            rs.fit(X_train, y_train)

            model.set_params(**rs.best_params_)
            model.fit(X_train, y_train)
            logging.info(f"model fitting done")

            y_train_pred = model.predict(X_train)
            logging.info(f"x_test train model")
            y_test_pred = model.predict(X_test)
            logging.info(f"X_test test model")
            # Calculate recall
            train_recall = recall_score(y_train, y_train_pred, average='binary')  # Change 'binary' if needed
            test_recall = recall_score(y_test, y_test_pred, average='binary')  # Change 'binary' if needed

            report[list(models.keys())[i]] = test_recall

            # Log the recall score
            logging.info(f"Model: {list(models.keys())[i]}")
            logging.info(f"Train Recall Score: {train_recall:.4f}")
            logging.info(f"Test Recall Score: {test_recall:.4f}")
        return report

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
