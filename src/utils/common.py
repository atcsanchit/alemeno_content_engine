import os
import sys
from langchain_chroma import Chroma
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException


def import_chroma_db(filepath, embedding_model):
    try:
        vectorstore_load_list = []
        no_of_dict = os.listdir(filepath)
        directories = [entry for entry in no_of_dict if os.path.isdir(os.path.join(filepath, entry))]
        for index in len(directories):
            persist_directory = os.path.join("content","chroma_db","db_"+str(index))
            vectorstore_load = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
            vectorstore_load_list.append(vectorstore_load)
        
        logging.info("import_chroma_db ran successfully")
        return vectorstore_load_list

    except Exception as e:
        raise CustomException(e,sys)