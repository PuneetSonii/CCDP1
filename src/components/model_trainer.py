import os
import sys
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from sklearn.metrics import recall_score
from catboost import CatBoostClassifier
from sklearn.ensemble import(
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models
from sklearn.model_selection import cross_val_score, StratifiedKFold

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info(f"Split training and test input data")
            logging.info(f"train_array,test_array")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            logging.info(f"X_test,y_test")
            models = {
                "Random Forest": RandomForestClassifier(n_estimators=100,max_depth=10,random_state=42),
                "Logistic Regression": LogisticRegression(solver="liblinear",max_iter=1000),
                "SVM": SVC(max_iter=500,random_state=42),
                "XGBoost": XGBClassifier(max_depth=5,random_state=42),
                "CatBoost": CatBoostClassifier(random_state=42),  
                "Decision Tree": DecisionTreeClassifier(random_state=42),  
                "K-Nearest Neighbors": KNeighborsClassifier(),  
                "Neural Network": MLPClassifier(max_iter=500, random_state=42)
                
            }
            logging.info(f"hyperparameter is started")
            
            params={
                "Random Forest" : {
                'n_estimators': [50, 100, 150],
                'max_depth': [5, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            },

                "XGBoost" : {
                'n_estimators': [50, 100, 150],
                'max_depth': [3, 4, 5],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },

            "Logistic Regression": {
                'C': [0.1, 1, 10],
                'penalty': ['l2']
            },

            "SVM": {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
            },
            "CatBoost": {
                'iterations': [50, 100, 150],
                'learning_rate': [0.01, 0.1, 0.2],
                'depth': [4, 6, 8],  
                'l2_leaf_reg': [1, 3, 5],  
            },
            "Decision Tree": {
                'max_depth': [None, 5, 10, 15],  
                'min_samples_split': [2, 5, 10],  
                'min_samples_leaf': [1, 2, 4],   
            },
            "K-Nearest Neighbors": {
                'n_neighbors': [3, 5, 7],  
                'weights': ['uniform', 'distance'],  
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],  
            },
            "Neural Network": {
                'hidden_layer_sizes': [(50,), (100,), (50, 50)],  
                'activation': ['logistic', 'tanh', 'relu'],  
                'solver': ['adam', 'lbfgs'],  
                'alpha': [0.0001, 0.001, 0.01], 
            }
            }
            
            logging.info(f"hyperparameter done")

            

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                            models=models,param=params)
            
            logging.info(f"Model Report: {model_report}")
            print('\n====================================================================================\n')
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # Sort the models based on their performance (descending order)
            sorted_models = sorted(model_report.items(), key=lambda x: x[1], reverse=True)

            # Log the models with automatically assigned numbers based on their score
            for i, (model_name, model_score) in enumerate(sorted_models, start=1):
                logging.info(f"{i}. {model_name} - Score: {model_score:.4f}")

            ## to get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            

            if best_model_score<0.6:
                raise CustomException(f"No best model found")
            logging.info(f"best found model on both training and testing dataset was: {best_model_name}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            recall = recall_score(y_test, predicted)
            return recall



        except Exception as e:
            raise CustomException(e,sys)
        

    