import os
import sys 
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    customer_churn_data_path: str = os.path.join("artifacts", "customer_churn.csv")
    customers_data_path: str = os.path.join("artifacts", "customers.csv")
    customer_engagement_metrics_data_path: str = os.path.join("artifacts", "customer_engagement_metrics.csv")
    orders_data_path: str = os.path.join("artifacts", "orders.csv")
    subscription_billing_data_path: str = os.path.join("artifacts", "subscription_billing.csv")
    support_tickets_data_path: str = os.path.join("artifacts", "support_tickets.csv")
    campaign_responses_data_path: str = os.path.join("artifacts", "campaign_responses.csv")

    merged_data_path: str = os.path.join("artifacts", "merged_data.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestionconfig=DataIngestionConfig()
    def initiate_ingestion(self):
        logging.info('initiated ingestion')
        try:
            pass
        except:
            pass
        
