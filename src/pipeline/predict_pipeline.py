import sys
import os
import pandas as pd

from src.exception import CustomException
from src.utils import load_object

from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelConfig

CHURN_FEATURES = [
                    "complaint_count",
                    "satisfaction_score",
                    "monthly_spend",
                    "total_spend",
                    "avg_order_value",
                    "tenure_months",
                    "contract_type",
                    "payment_method",
                    "autopay",
                    "login_frequency_30d",
                    "recency_days",
                    "frequency",
                    "cart_abandonment_rate",
                    "engagement_tier",
                    "late_payments_12m",
                    "payment_failures_90d"
                ]

class PredictPipeline:
    def __init__(self):
        self.model_file_path= ModelConfig()
        self.processor_file_path=DataTransformationConfig()
    def predict(self,data):
        model_path=self.model_file_path.model_obj_file_path
        processor_path=self.processor_file_path.preprocessor_obj_file_path
        model=load_object(model_path)
        processor=load_object(processor_path)
        data=processor.transform(data)
        prediction=model.predict(data)
        return prediction


class CustomData:
    def __init__(self,data):
        self.data=data
    def tranform_to_df(self):
        missing_columns=[column for column in CHURN_FEATURES if column not in self.data ]     
        if missing_columns:
            raise ValueError(f"Missing features: {missing_columns}")
        for col  in CHURN_FEATURES:
            if col not in ("contract_type", "payment_method", "engagement_tier"):
                    self.data[col]=pd.to_numeric(self.data[col],errors="raise")
            

        return pd.DataFrame([self.data],columns=CHURN_FEATURES)