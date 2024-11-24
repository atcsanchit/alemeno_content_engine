import sys
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.components.prediction_mistral_ai import PredictionMistral

@dataclass
class PredictionMistralPipeline:
    def __init__(self):
        pass

    def initiate_pipeline(self,text):
        try:
            prediction_obj = PredictionMistral()
            return prediction_obj.generate_text(text=text)

        except Exception as e:
            logging.info("Error in prepare base pipeline method")
            raise CustomException(e,sys)

if __name__ == "__main__":
    pipeline_obj = PredictionMistralPipeline()
    pipeline_obj.initiate_pipeline()