#script to execute all the pipelines at once from scratch
import sys
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException

from src.pipeline.prepare_artifacts_pipeline import PrepareArtifactsPipeline

@dataclass
class Pipeline:
    def __init__(self):
        pass

    def execute_pipeline(self):
        try:
            prepare_artifacts_obj = PrepareArtifactsPipeline()
            prepare_artifacts_obj.initiate_pipeline()

            print("all pipelines are successfully executed")
            logging.info("all pipelines are successfully executed")

        except Exception as e:
            logging.info("Error in execute_pipeline method in main strategy")
            raise CustomException(e,sys)
        

if __name__ == "__main__":
    pipeline_obj = Pipeline()
    pipeline_obj.execute_pipeline()