import sys
import os
from langchain.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_chroma import Chroma
from dataclasses import dataclass
from dotenv import load_dotenv

from src.logger import logging
from src.exception import CustomException

@dataclass
class PredictionMistralConfig:
    model="mistral-large-latest"
    vector_db_path = os.path.join("artifacts","prepare_vector_db")
    embedding_model_name = "BAAI/bge-base-en"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}

class PredictionMistral:
    def __init__(self):
        self.prediction_config = PredictionMistralConfig()

    def load_model(self):
        try:
            load_dotenv()
            api_key = os.getenv("MISTRAL_API_KEY")
            os.environ["MISTRAL_API_KEY"] = api_key
            self.llm = ChatMistralAI(model=self.prediction_config.model)
            print("*"*100)
            print("model load successfully")
        except Exception as e:
            logging.info(f"Error in load_model -- {e}")
            raise CustomException(e,sys)
        
    def import_chroma_db(self, filepath, embedding_model):
        try:
            vectorstore_load_list = []
            no_of_dict = os.listdir(filepath)
            # directories = [entry for entry in no_of_dict if os.path.isdir(os.path.join(filepath, entry))]
            for index in range(0,len(no_of_dict)):
                persist_directory = os.path.join(filepath,"db_"+str(index))
                vectorstore_load = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
                vectorstore_load_list.append(vectorstore_load)
            
            logging.info("import_chroma_db ran successfully")
            return vectorstore_load_list

        except Exception as e:
            raise CustomException(e,sys)


    def get_context(self, question):
        try:
            embedding_model = HuggingFaceBgeEmbeddings(
                model_name=self.prediction_config.embedding_model_name, 
                model_kwargs=self.prediction_config.model_kwargs, 
                encode_kwargs=self.prediction_config.encode_kwargs
            )
            vectorstore_list = self.import_chroma_db(filepath=self.prediction_config.vector_db_path, embedding_model=embedding_model)
            context_list = []
            for vectorstore in vectorstore_list:
                docs = vectorstore.similarity_search(question)[0].page_content
                context_list.append(docs)

            context = ""
            for i in range(0,len(context_list)):
                
                context += f"doc {i}: " + context_list[i] + "\n\n\n"
            return context

        except Exception as e:
            logging.info(f"Error in get_context -- {e}")
            raise CustomException(e,sys)
    
    def generate_text(self,text):
        try:
            self.load_model()
            context = self.get_context(question=text)
            message = """
                    Answer the following question using the provided context only. Make sure to answer the questions in more personised way and user friendly way.

                    Question: {text}

                    Context:
                    {context}
                    """.format(text=text, context=context)
            
            prompt = ChatPromptTemplate.from_messages([("human", message)])
            rag_chain = prompt | self.llm
            return rag_chain.invoke({}).content

        except Exception as e:
            logging.info(f"Error in generate_text -- {e}")
            raise CustomException(e,sys)
        
if __name__ == "__main__":

    prediction_obj = PredictionMistral()
    ans = prediction_obj.generate_text(text="Hi. My name is Sanchit.Can you tell the difference in the title of the documents?")
    print(ans)