import os
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()


# API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Set OPENAI_API_KEY in your .env file")


# Repositories
DATA_DIR = "data"
EMB_MODEL = "text-embedding-3-large"


# Vector index
FAISS_INDEX_DIR = "vector_data/faiss_index"     


# Embedding Model
embeddings = OpenAIEmbeddings(model=EMB_MODEL)
