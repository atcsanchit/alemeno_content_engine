import sys
import os
import tensorflow as tf
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException

@dataclass
class PrepareBaseModelConfig:
    model_name = "Qwen/Qwen2.5-0.5B"
    output_path = os.path.join("artifacts","prepare_base_model","Qwen")

class PrepareBaseModel:
    def __init__(self):
        self.prepare_base_model = PrepareBaseModelConfig()
    
    def load_model_tokenizer(self):
        try:
            self.model = AutoModelForCausalLM.from_pretrained(self.prepare_base_model.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.prepare_base_model.model_name)

        except Exception as e:
            logging.info(f"Error in load_model_tokenizer -- {e}")
            raise CustomException(e,sys)
    
    def save_model_tokenizer(self):
        try:
            self.model.save_pretrained(self.prepare_base_model.output_path)
            self.tokenizer.save_pretrained(self.prepare_base_model.output_path)

        except Exception as e:
            logging.info(f"Error in save_model_tokenizer -- {e}")
            raise CustomException(e,sys)
    
    def load_and_save(self):
        try:
            self.load_model_tokenizer()
            self.save_model_tokenizer()

        except Exception as e:
            logging.info(f"Error in load_and_save -- {e}")
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    prepare_base_model = PrepareBaseModel()
    prepare_base_model.load_and_save()