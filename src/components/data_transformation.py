import os
import sys
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
import numpy as np
from src.utils import save_obj
@dataclass
class DataTransformationConfig:
    train_data_path=os.path.join("artifacts",'train.csv')
    preprocessor_obj_file_path=os.path.join("artifacts",'preprocessor.pkl')
    test_data_path=os.path.join("artifacts",'test.csv')
    featured_ds_path=os.path.join("artifacts",'feature_engineered.csv')

class DataTransformation:
    def __init__(self):
        self.transformationConfig=DataTransformationConfig()
    def feature_Rfm(self,merge_path):
        try:
            orders=pd.read_csv('notebook/data/orders.csv')
            merge_df=pd.read_csv(merge_path)
            logging.info("Started null treatment in orders")

            #handing missing in orders
            country_map=(orders.dropna(subset='shipping_country')
                         .drop_duplicates('shipping_state')
                         .set_index("shipping_state")['shipping_country'])
            orders['shipping_country']=orders['shipping_country'].fillna(orders['shipping_state'].map(country_map))
            orders['promo_code']=orders['promo_code'].fillna('No promo')

            logging.info("completed null treatment in orders")

            logging.info("Feature engineering RFM from orders")

            orders['order_date']=pd.to_datetime(orders['order_date'])
            reference_date=orders['order_date'].max()+pd.Timedelta(days=1)
            customer_rfm=(
                orders.groupby('customer_id')
                .agg(
                    recency_days=('order_date',lambda x:(reference_date-x.max()).days),
                    frequency=('order_id','count'),
                    monetary=('total_amount','sum')
                )
            )

            logging.info('creating features F score,R score,M score')
            customer_rfm['R_score'] = pd.qcut(
                customer_rfm['recency_days'].rank(method='first'),
                q=5,
                labels=[5, 4, 3, 2, 1]
            ).astype(int)

            customer_rfm['F_score'] = pd.qcut(
                customer_rfm['frequency'].rank(method='first'),
                q=5,
                labels=[1, 2, 3, 4, 5]
            ).astype(int)

            customer_rfm['M_score'] = pd.qcut(
                customer_rfm['monetary'].rank(method='first'),
                q=5,
                labels=[1, 2, 3, 4, 5]
            ).astype(int)
            logging.info('created features F score,R score,M score')


            merge_df=merge_df.merge(
                customer_rfm,
                on='customer_id',
                how='left'
                )
            logging.info("Added features to Merge data set")

            merge_df.to_csv(self.transformationConfig.featured_ds_path,index=False)
            logging.info("Saved featured csv in artifacts")

            return self.transformationConfig.featured_ds_path

        except Exception as e:
            raise CustomException(e,sys)
            
    def train_test_split_data(self):
        try:
            logging.info("spliting data into train test")
            df=pd.read_csv(self.transformationConfig.featured_ds_path)
            #drops customer id from feature data set but not inplace
            df=df.drop('customer_id',axis=1)
            train_set,test_set=train_test_split(df,test_size=0.2,stratify=df['churned'])
            train_set.to_csv(self.transformationConfig.train_data_path,index=False)
            test_set.to_csv(self.transformationConfig.test_data_path,index=False)
            logging.info('Splitted the dataset')

            return (self.transformationConfig.train_data_path,self.transformationConfig.test_data_path)

        except Exception as e:
            raise CustomException(e,sys)

    def get_data_transformer_obj(self):

        try:
                logging.info('data transfomer obj started.')
                df=pd.read_csv(self.transformationConfig.featured_ds_path)
                median_columns = [
                    "age",
                    "household_size",
                    "lifetime_value",
                    "loyalty_points",
                    "monthly_recurring_revenue",
                    "annual_contract_value"
                ]

                cat_cols = [
                    "contract_type",
                    "payment_method",
                    "gender",
                    "marital_status",
                    "segment",
                    "preferred_channel",
                    "preferred_language",
                    "account_status",
                    "referral_source",
                    "engagement_tier",
                    "billing_cycle",
                    "invoice_delivery"
                ]
                zero_columns = [
                    "email_opt_in",
                    "sms_opt_in",
                    "autopay_enabled",
                    "paperless_billing",
                    "late_payments_12m",
                    "price_increase_last_year",
                    "discount_pct",
                    "payment_failures_90d",
                    "tax_exempt",
                    "frequency",
                    "monetary",
                    "R_score",
                    "F_score",
                    "M_score"
                ]
                num_median_pipeline=Pipeline(
                     steps=[
                          ("imputer",SimpleImputer(strategy='median')),
                          ("scaler",StandardScaler())
                     ]
                )

                num_zero_pipeline=Pipeline(
                     steps=[
                          ('imputer',SimpleImputer(strategy='constant',fill_value=0)),
                          ('scaler',StandardScaler())
                     ]
                )

                cat_pipeline=Pipeline(
                     steps=[
                          ('imputer',SimpleImputer(strategy='constant',fill_value='unknown')),
                          ('encoder',OneHotEncoder(handle_unknown='ignore'))
                     ]
                )
                logging.info( f"Median numerical columns: {median_columns}")
                logging.info( f"Zero numerical columns: {zero_columns}")
                logging.info( f"categorical columns: {cat_cols}")



                preprocessor=ColumnTransformer(
                     transformers=[
                          ('num_median',num_median_pipeline,median_columns),
                          ('num_zero',num_zero_pipeline,zero_columns),
                          ('cat',cat_pipeline,cat_cols)
                     ]
                )
                logging.info("preprocessing completed")
                return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
    def initiate_transformation(self,merge_path):
        try:
            feature_ds_path=self.feature_Rfm(merge_path)
            train_path,test_path=self.train_test_split_data()

            #Reading train and test
            train_data=pd.read_csv(train_path)
            test_data=pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_obj()

            target_column='churned'

            input_train_features=train_data.drop(columns=[target_column])
            target_train_feature=train_data[target_column]

            input_test_features=test_data.drop(columns=[target_column])
            target_test_feature=test_data[target_column]

            input_train_features=preprocessing_obj.fit_transform(input_train_features)
            input_test_features=preprocessing_obj.transform(input_test_features)


            train_arr=np.c_[input_train_features,np.array(target_train_feature)]
            test_arr=np.c_[input_test_features,np.array(target_test_feature)]
            logging.info(f"Saved preprocessing object.")

            save_obj(
                file_path=self.transformationConfig.preprocessor_obj_file_path,
                obj=preprocessing_obj
                )
            return(
                    train_arr,
                    test_arr,
                    self.transformationConfig.preprocessor_obj_file_path,
                )

        except Exception as e:
            raise CustomException(e,sys)



  

