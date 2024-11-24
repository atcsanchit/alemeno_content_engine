import sys
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.components.prepare_base_model import PrepareBaseModel
from src.components.prepare_vector_db import PrepareVectorDB

@dataclass
class PrepareArtifactsPipeline:
    def __init__(self):
        pass

    def initiate_pipeline(self):
        try:
            logging.info("Initiating prepare artifacts pipeline")
            print("Initiating prepare artifacts pipeline")

            prepare_base_model_obj = PrepareBaseModel()
            prepare_base_model_obj.load_and_save()

            prepare_vector_db_obj = PrepareVectorDB()
            prepare_vector_db_obj.save_database()

            print("prepare artifacts pipeline has been successfully executed")
            print("*"*20)        

        except Exception as e:
            logging.info("Error in prepare artifacts pipeline method")
            raise CustomException(e,sys)

if __name__ == "__main__":
    pipeline_obj = PrepareArtifactsPipeline()
    pipeline_obj.initiate_pipeline()