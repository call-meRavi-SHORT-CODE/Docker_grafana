import os
from dotenv import load_dotenv

load_dotenv()



# API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Set OPENAI_API_KEY in your .env file")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Set GROQ_API_KEY in your .env file")
# Optionally strip whitespace/newlines:
GROQ_API_KEY = GROQ_API_KEY.strip()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Set GEMINI_API_KEY in your environment or .env file")


DATA_DIR = 'Rag-API\data'
FAISS_INDEX_DIR = "vector_data/faiss_index"
CHROMA_INDEX_DIR = "vector_data/chroma_index"

# If you have specific embedding objects, import or configure them here:
# e.g. embeddings = OpenAIEmbeddings(...)
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()  # adjust parameters as needed
