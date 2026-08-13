import sys
import os
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


@dataclass
class ModelConfig:
    model_obj_file_path: str = os.path.join(
        "artifacts",
        "model.pkl"
    )


class ModelTrainer:

    def __init__(self):
        self.model_config = ModelConfig()

    def evaluate_metrics(self,y_test, y_pred, y_prob):

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc = roc_auc_score(y_test, y_prob)

        hash_map={
            "accuracy_score":accuracy,
            "precision_score":precision,
           "recall_score":recall,
            "f1_score":f1,
            "roc_auc_score":roc
        }

        return hash_map
 
    

    def initiate_model_trainer(self,X_train,X_test,y_train,y_test):
        models = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000,
                random_state=42
            ),

            "Random Forest": RandomForestClassifier(
                random_state=42
            ),

            "XGBoost": XGBClassifier(
                random_state=42,
                eval_metric="logloss"
            ),

            "LightGBM": LGBMClassifier(
                random_state=42,
                verbose=-1
            )
        }

        param_grids = {

            "Logistic Regression": {
                "C": [0.01, 0.1, 1, 10, 100],
                "solver": ["liblinear", "lbfgs"]
            },


            "XGBoost": {
                "n_estimators": [100, 200],
                "max_depth": [3, 5],
                "learning_rate": [0.05, 0.1],
                "subsample": [0.8],
                "colsample_bytree": [0.8]
            },

            "LightGBM": {
                "n_estimators": [100, 200],
                "learning_rate": [0.1, 0.05],
                "num_leaves": [31, 50],
                "max_depth": [-1, 10],
            },
            
            "Random Forest": {
                "n_estimators": [100, 200],
                "max_depth": [None,20],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2]
            }
        }
        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        )
        best_score = 0
        best_model = None
        best_model_name = None

        for name, model in models.items():
            logging.info(f"Training {name}")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            r = self.evaluate_metrics(
                y_test,
                y_pred,
                y_prob
            )
            logging.info(f"{name} metrics: {r}")

            print(f"{name} ROC-AUC: {r['roc_auc_score']:.4f}")

            if r["roc_auc_score"] > best_score:

                best_score = r["roc_auc_score"]
                best_model = model
                best_model_name = name
        print("Best Model:", best_model_name)
        print("Best ROC-AUC:", best_score)

        grid_search=GridSearchCV(
                estimator=best_model,
                param_grid=param_grids[best_model_name],
                scoring='roc_auc',
                cv=cv,
                n_jobs=-1
            )


        grid_search.fit(X_train,y_train)
        best_model = grid_search.best_estimator_


        logging.info(f"{best_model_name} training completed")
        logging.info(f"best CV ROC-AUC: {grid_search.best_score_}")
        logging.info(f"best parameters: {grid_search.best_params_}")


        save_obj(file_path=self.model_config.model_obj_file_path,
                 obj=best_model)

        y_pred=best_model.predict(X_test)
        y_prob=best_model.predict_proba(X_test)[:, 1]

        results=self.evaluate_metrics(y_test,y_pred,y_prob)
        logging.info(f"Test metrics: {results}")

        logging.info("saved model.")
        return results




