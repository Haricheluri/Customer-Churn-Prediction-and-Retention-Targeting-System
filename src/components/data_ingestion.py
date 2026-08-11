import os
import sys 
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
@dataclass
class DataIngestionConfig:
    merged_data_path: str=os.path.join('artifacts',"data.csv")
class DataIngestion:
    def __init__(self):
        self.ingestionconfig=DataIngestionConfig()
    def read_datasets(self):
        try:
            logging.info("Reading Data sets...")
            customer_churn_df=pd.read_csv('notebook/data/customer_churn.csv')
            customers_df=pd.read_csv('notebook/data/customers.csv')
            engagement_metrics_df=pd.read_csv('notebook/data/customer_engagement_metrics.csv')
            subscription_df=pd.read_csv('notebook/data/subscription_billing.csv')


            logging.info("Datasets reading completed.")
            return({
                     "customer_churn":customer_churn_df,
                     "customers":customers_df,
                     "customer_engagement_metrics":engagement_metrics_df,
                     "subscription_billing":subscription_df
                   })
            
        except Exception as e:
            raise CustomException(e,sys)
    def merge_datasets(self,datasets:dict):
        try:
            logging.info("Data Merging Started")
            #extract datasets from dictionary
            customer_churn_df = datasets["customer_churn"]
            customers_df = datasets["customers"]
            engagement_metrics_df = datasets["customer_engagement_metrics"]
            subscription_df = datasets["subscription_billing"]
    

            missing_ds = []
            duplicate_ds = []
            for name,df in datasets.items():
                if df.isna().sum().sum()>0:
                    missing_ds.append(name)
                if df.duplicated().sum()>0:
                    duplicate_ds.append(name)

            # if missing_ds:
            #     print("These datasets contain Null values",missing_ds)
            # else:
            #     print("No Null values")

            # if duplicate_ds:
            #     print("The datasets conatain Duplicated rows",duplicate_ds)
            # else:
            #     print("No Duplicated Rows")

            logging.info("Null and Duplicate Analysis completed.")
            logging.info("Handling missing values..")
            for name in missing_ds:
                
                df=datasets[name]
                if name=='customers':
                    state_country_map=(
                        df.dropna(subset=['country'])
                        .drop_duplicates("state")
                        .set_index("state")['country']
                    )
                    df["country"] = df["country"].fillna(df["state"].map(state_country_map))
                    # print("Filled missing values in 'country' using 'state' mapping.")
                #updates the dataset dictionary
                datasets[name] = df
            logging.info("Missing value handling completed.")

            
            logging.info("Handling Relationships started...")
            #keeping customers with latest records in customer churn
            customer_churn_df['updated_at']=pd.to_datetime(customer_churn_df['updated_at'])
            customer_churn_df=customer_churn_df.sort_values(by='updated_at')
            customer_churn_df=customer_churn_df.drop_duplicates(subset='customer_id',keep='last')
            #updates the dataset dictionary
            datasets["customer_churn"] = customer_churn_df

            # handling duplicate customers in enagement metrics 
            engagement_metrics_df['measured_date']=pd.to_datetime(engagement_metrics_df['measured_date'])
            engagement_metrics_df=engagement_metrics_df.sort_values(by='measured_date')
            engagement_metrics_df=engagement_metrics_df.drop_duplicates(subset='customer_id',keep='last')
            #updates the dataset dictionary
            datasets['customer_engagement_metrics']=engagement_metrics_df

            logging.info("Handling Relationships completed.")
            integrated_df=customer_churn_df.copy()

            logging.info("Merging Started...")
           
            integrated_df=integrated_df.merge(
                customers_df,
                on="customer_id",
                how="left"
            )
            integrated_df=integrated_df.merge(
                engagement_metrics_df,
                on="customer_id",
                how="left"
            )
            integrated_df=integrated_df.merge(
                subscription_df,
                on="customer_id",
                how="left"
            )
            logging.info("Datasets merged successfully.")
            return integrated_df
        except Exception as e:
            raise CustomException(e,sys)

    def save_clean_data(self,df):
        try:
            logging.info("dropping columns...")
            df.rename(
                 columns={
                    "tenure_months_x": "tenure_months",
                    "contract_type_x": "contract_type",
                    "payment_method_x": "payment_method",
                    "nps_score_x": "nps_score"
                 },inplace=True)
            
            columns_to_drop = [
                # Duplicate Columns (_y)
                "tenure_months_y",
                "contract_type_y",
                "payment_method_y",
                "nps_score_y",
                # Administrative Columns
                "account_manager_x",
                "account_manager_y",
                "updated_at_x",
                "updated_at_y",
                "last_login_date_x",
                "last_login_date_y",
                "created_at_x",
                "created_at_y",
                "last_contact_date",
                "measured_date",
                "contract_end_date",
                "signup_date",
                # Personal 
                "email",
                "name",
                "first_name",
                "last_name",
                "phone",
                "street_address",
                "postal_code",
                # Internal Identifiers
                "metric_id",
                "billing_id",
                # Target Leakage
                "churn_reason",
                "predicted_churn_prob",
                # High Cardinality Columns
                "plan_name",
                "billing_address_city",
                "city",
                "occupation",
                "country",
                "state"
            ]
            df.drop(columns=columns_to_drop, inplace=True)
            os.makedirs('artifacts',exist_ok=True)
            df.to_csv(self.ingestionconfig.merged_data_path,index=False)
            
            logging.info("Data set saved")

            return self.ingestionconfig.merged_data_path

        except Exception as e:
            raise CustomException(e,sys)
    def initiate_data_ingestion(self):
        try:
            logging.info("Data Ingestion Started")
            datasets=self.read_datasets()
            integrated_df=self.merge_datasets(datasets)
            merge_path=self.save_clean_data(integrated_df)
            logging.info("Data Ingestion Completed Successfully")
            return merge_path
            
        except Exception as e:
            raise CustomException(e,sys)

            
if __name__=='__main__':
    obj=DataIngestion()
    merge_path=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    X_train,X_test,y_train,y_test,_=data_transformation.initiate_transformation(merge_path)
    print(pd.DataFrame(X_train).isna().sum().sum())
    print(pd.DataFrame(X_test).isna().sum().sum())

    