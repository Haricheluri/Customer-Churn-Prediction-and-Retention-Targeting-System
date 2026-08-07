import os
import sys
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
import pandas as pd
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join("artifacts",'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.transformationConfig=DataTransformationConfig()
    def handling_missing_values(train_path,test_path):
        try:
            pass
        except Exception as e:
            raise CustomException(e,sys)
        