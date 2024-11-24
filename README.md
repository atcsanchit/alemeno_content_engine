# Content Engine for Alemeno
## Assignment - To build a content engine which will compare the contents of multiple PDF about a particular topic of discussion.
### Overview

The Alemeno Content Engine is a system designed to analyze and compare multiple PDF documents, specifically highlighting their differences and enabling insights generation. It employs Retrieval Augmented Generation (RAG) techniques for effective content retrieval and analysis. A user-friendly chatbot interface built using Streamlit allows users to interact with the system for insights and document comparisons.

### Key Features

- PDF Parsing: Extracts and structures text from PDF documents.

- Embedding Generation: Creates vector representations of document content using a locally running embedding model.

- Vector Store: Manages and queries embeddings efficiently.

- Query Engine: Facilitates retrieval tasks and contextual insights through a local Large Language Model (LLM).

- Interactive Chatbot: Enables querying and comparative insights through a Streamlit-based interface.

### System Architecture

The system is designed to be modular and scalable, comprising the following components:

- Backend Framework: LangChain for creating custom retrieval systems.

- Frontend Framework: Built with Streamlit for an interactive user experience.

- Vector Store: Utilizes a locally hosted vector database (ChromaDB).

### Project Setup

#### Prerequisites

- Python 3.8+

- Streamlit

- Vector store (e.g., ChromaDB, Faiss, or Pinecone)

- Embedding Model (e.g., Sentence Transformers)

- LLM (e.g., LlamaCpp or GPT4All)

### Installation

1. Clone the repository:
`git clone https://github.com/atcsanchit/alemeno_content_engine.git`
`cd alemeno_content_engine`

2. Install dependencies:
`pip install -r requirements.txt`

3. Run all the intermediate pipelines:
`python main.py`

4. Run the Streamlit app:
`streamlit run app1.py`
or
`streamlit run app2.py`

