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
            
    def train_test_split_data(self,path):
        try:
            logging.info("spliting data into train test")
            df=pd.read_csv(path)
            #drops customer id from feature data set but not inplace
            X=df.drop(columns=['customer_id','churned'])
            y=df['churned']
            X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=y,test_size=0.2,random_state=42)
            train_set=pd.concat([X_train,y_train],axis=1)
            test_set=pd.concat([X_test,y_test],axis=1)
            train_set.to_csv(self.transformationConfig.train_data_path,index=False)
            test_set.to_csv(self.transformationConfig.test_data_path,index=False)
            logging.info('Splitted the dataset')

            return (X_train,X_test,y_train ,y_test)

        except Exception as e:
            raise CustomException(e,sys)

    def get_data_transformer_obj(self):

        try:
                df=pd.read_csv(self.transformationConfig.featured_ds_path)
                logging.info('data transfomer obj started.')
                num_cols = df.select_dtypes(include=["int64", "float64"]).columns.to_list()
                num_cols.remove('churned')
                cat_cols = df.select_dtypes(include=["object", "category"]).columns.to_list()
                cat_cols.remove('customer_id')
                num_pipeline=Pipeline(
                     steps=[
                          ("imputer",SimpleImputer(strategy='median')),
                          ("scaler",StandardScaler())
                     ]
                )

                cat_pipeline=Pipeline(
                     steps=[
                          ('imputer',SimpleImputer(strategy='constant',fill_value='unknown')),
                          ('encoder',OneHotEncoder(handle_unknown='ignore'))
                     ]
                )

                preprocessor=ColumnTransformer(
                     transformers=[
                          ('num_median',num_pipeline,num_cols),
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
            X_train,X_test,y_train,y_test=self.train_test_split_data(feature_ds_path)
            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_obj()

            X_train=preprocessing_obj.fit_transform(X_train)
            X_test=preprocessing_obj.transform(X_test)

            logging.info(f"Saved preprocessing object.")

            save_obj(
                file_path=self.transformationConfig.preprocessor_obj_file_path,
                obj=preprocessing_obj
                )
            return(
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    self.transformationConfig.preprocessor_obj_file_path,
                )

        except Exception as e:
            raise CustomException(e,sys)



  

