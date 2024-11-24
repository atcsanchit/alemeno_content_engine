import sys
import os
import tensorflow as tf
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException


@dataclass
class PrepareVectorDBConfig:
    file_path_list = [os.path.join("artifacts","data","goog-10-k-2023 (1).pdf"),os.path.join("artifacts","data","tsla-20231231-gen.pdf"),os.path.join("artifacts","data","uber-10-k-2023.pdf")]
    chunk_size = 1000
    chunk_overlap = 200
    model_name = "BAAI/bge-base-en"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    output_path = os.path.join("artifacts","prepare_vector_db")

class PrepareVectorDB:
    def __init__(self):
        self.prepare_vector_db = PrepareVectorDBConfig()

    def load_pdf(self):
        try:
            self.list_of_pdf = []
            for file_path in self.prepare_vector_db.file_path_list:
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                self.list_of_docs.append(docs)

        except Exception as e:
            logging.info(f"Error in load_pdf -- {e}")
            raise CustomException(e,sys)
        
    
    def text_splitting(self):
        try:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.prepare_vector_db.chunk_size, chunk_overlap=self.prepare_vector_db.chunk_overlap)
            self.list_of_splits = []
            for doc in self.list_of_docs:
                splits = text_splitter.split_documents(doc)
                self.list_of_splits.append(splits)

        except Exception as e:
            logging.info(f"Error in text_splitting -- {e}")
            raise CustomException(e,sys)
        
    def save_database(self):
        try:
            self.embedding_model = HuggingFaceBgeEmbeddings(
                model_name=self.prepare_vector_db.model_name,
                model_kwargs=self.prepare_vector_db.model_kwargs,
                encode_kwargs=self.prepare_vector_db.encode_kwargs
                )
            
            for index, splits in enumerate(self.list_of_splits):
                persist_directory = os.path.join(self.prepare_vector_db.output_path, "db_"+str(index))
                vectorstore = Chroma.from_documents(
                    documents=splits, 
                    embedding=self.embedding_model,
                    persist_directory=persist_directory
                    )


        except Exception as e:
            logging.info(f"Error in save_database -- {e}")
            raise CustomException(e,sys)
        
    def initiate_preparing_vector_db(self):
        try:
            self.load_pdf()
            self.text_splitting()
            self.save_database()

        except Exception as e:
            logging.info(f"Error in initiate_preparing_vector_db -- {e}")
            raise CustomException(e,sys)
        
# if __name__ == "__main__":
#     prepare_vector_db = PrepareVectorDB()
#     prepare_vector_db.save_database()