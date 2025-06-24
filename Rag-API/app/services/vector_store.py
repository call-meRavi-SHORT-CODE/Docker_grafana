import os
import logging
from typing import Any

from config import FAISS_INDEX_DIR, CHROMA_INDEX_DIR, embeddings

# Import your index-builders here
from vector_stores.faiss_index import build_faiss_index
from vector_stores.chroma_index import build_chroma_index

from langchain_community.vectorstores import FAISS, Chroma


def get_vector_store(store_name: str) -> Any:
    """
    Returns a loaded or newly-built vector store instance 
    based on `store_name` ("faiss", "chroma", "annoy", etc.).
    """
    if store_name.lower() == "faiss":
        if os.path.exists(FAISS_INDEX_DIR):
            logging.info("Loading existing FAISS index…")
            return FAISS.load_local(
                FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True
            )
        else:
            return build_faiss_index()

    elif store_name == "chroma":
        # If the folder exists, simply re-instantiate Chroma pointing at that folder.
        if os.path.exists(CHROMA_INDEX_DIR):
            logging.info("Loading existing Chroma index…")
            return Chroma(
                persist_directory=CHROMA_INDEX_DIR,
                embedding_function=embeddings
            )
        else:
            logging.info("Chroma index not found. Building a new one…")
            new_db = build_chroma_index()
           
            return new_db

    elif store_name == "annoy":
        # Placeholder for Annoy implementation
        raise ValueError("Annoy support not yet implemented")
    else:
        raise ValueError(f"Unsupported vector store: {store_name}")



