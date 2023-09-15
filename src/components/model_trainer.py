import os
import sys
from dataclasses import dataclass

from catboost import CatBoostClassifier
from sklearn.ensemble import(
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            logging.info(train_array,test_array)
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            logging.info(X_test,y_test)
            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting" : GradientBoostingClassifier(),
                "K-Neighbors Classifier" : KNeighborsClassifier(),
                "XGBClassifier": XGBClassifier(),
                "CatBoosting Classifier": CatBoostClassifier(verbose=False),
                "AdaBoost Classifier": AdaBoostClassifier(),
                "SVM": SVC(),
                "Neural Network (MLP)": MLPClassifier()
            }
            logging.info("hyperparameter is started")
            params={
                "Decision Tree": {
                'criterion': ['gini', 'entropy'],
                'max_depth': [None, 5, 10, 20],
                'min_samples_split': [2, 5, 10]
                },
                "Random Forest": {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 5, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['auto', 'sqrt', 'log2']
                },
                "Gradient Boosting": {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.1, 0.05, 0.01],
                'max_depth': [3, 4, 5],
                'min_samples_split': [2, 5, 10]
                },
                "XGBoost": {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 4, 5],
                'learning_rate': [0.1, 0.05, 0.01],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
                },
                "CatBoosting Classifier": {
                'depth': [6, 8, 10],
                'learning_rate': [0.01, 0.05, 0.1],
                'iterations': [30, 50, 100]
                },
                "AdaBoost Classifier": {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.1, 0.05, 0.01]
                },
                "SVM": {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf']
                },
                "Neural Network (MLP)": {
                'hidden_layer_sizes': [(100,), (50, 50), (30, 30, 30)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'invscaling', 'adaptive']
                },
                "K-Neighbors Classifier": {
                'n_neighbors': [3, 5, 7, 9]
    }
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                            models=models,param=params)
            logging.info(model_report)
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## to get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            print(best_model_name)
            logging.info(best_model_name)

            if best_model_score<0.6:
                raise Exception("No best model found")
            logging.info(f"best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square



        except Exception as e:
            raise CustomException(e,sys)