import os
import sys
sys.path.append("src")
from dataclasses import dataclass
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import r2_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from exception import CustomException
from logger import logging
from utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.sav")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Logistic Regression": LogisticRegression()
                #"Random Forest": RandomForestClassifier(),
                #"Decision Tree": DecisionTreeClassifier(),
                #"K Neighbors Classifier": KNeighborsClassifier(),
                #"Gradient Boosting": GradientBoostingClassifier(),
                #"AdaBoost Classifier": AdaBoostClassifier()
            }
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            accuracy = accuracy_score(y_test, predicted)
            return (r2_square, accuracy, best_model, model_report.values())
            
        except Exception as e:
            raise CustomException(e,sys)
        