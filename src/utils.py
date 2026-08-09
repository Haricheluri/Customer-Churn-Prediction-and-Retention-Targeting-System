import os
import sys

import pickle
from src.exception import CustomException

def save_obj(file_path,obj):
    try:
        dir_name= os.path.dirname(file_path)
        os.makedirs(dir_name,exist_ok=True)
        with open(file_path,'wb') as file:
            pickle.dump(obj,file)
    except Exception as e:
        raise CustomException(e,sys)


