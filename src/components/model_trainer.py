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

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
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

    def evaluate_metrics(self,X_train,X_test,y_train,y_test,models):
        reports = {}
        for name, model in models.items():
            logging.info(f"Training {name}")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_prob)
            reports[name] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "roc_auc": roc_auc,
                "classification_report": classification_report(
                    y_test,
                    y_pred
                )
            }
        
            logging.info(
                f"{name} | "
                f"Accuracy: {accuracy:.4f}, "
                f"Precision: {precision:.4f}, "
                f"Recall: {recall:.4f}, "
                f"F1: {f1:.4f}, "
                f"ROC-AUC: {roc_auc:.4f}"
            )

        best_model_name = None
        best_roc_auc= 0

        for name, metrics in reports.items():

            if metrics["roc_auc"] > best_roc_auc:
                best_roc_auc = metrics["roc_auc"]
                best_model_name = name
        best_model=models[best_model_name]
        print("Best Model:", best_model_name)
        print("Best ROC-AUC Score:", best_roc_auc)
        if best_roc_auc < 0.80:
            raise Exception("No good model")

        return reports,best_model

    def initiate_model_trainer(self,X_train,X_test,y_train,y_test):
        models = {
            "Logistic Regression": LogisticRegression(),
            "Random Forest": RandomForestClassifier(),
            "XG boost": XGBClassifier(),
            "Lightgbm": LGBMClassifier()
        }

        reports,best_model = self.evaluate_metrics(X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test,models=models)
        save_obj(file_path=self.model_config.model_obj_file_path,
                 obj=best_model)


