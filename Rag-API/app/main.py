from fastapi import FastAPI
from app.routers.ask import router as ask_router
import logging
from app.services.vector_store import get_vector_store
from app.services.llm import get_llm
from app.services.rag_chain import build_rag_retrieval_chain
import dspy
import os 

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
app = FastAPI(
    title="Configurable RAG Agent API",
    version="1.0.0"
)

# Mount the /ask router
app.include_router(ask_router, prefix="", tags=["RAG"])



@app.on_event("startup")
async def preload_indexes_and_chains():

    lm = dspy.LM('openai/gpt-4o-mini')


    dspy.configure(lm=lm)
    """
    This runs once when FastAPI starts. We’ll force building/loading the 
    FAISS and Chroma indexes here, so subsequent /ask calls are fast.
    """
    logging.info("Preloading FAISS index…")
    try:
        # Change “faiss” or “chroma” depending on which you actually need first
        _ = get_vector_store("faiss")
        logging.info("FAISS loaded/built successfully.")
    except Exception as e:
        logging.error(f"Failed building/loading FAISS: {e}")

    logging.info("Preloading Chroma index…")
    try:
        _ = get_vector_store("chroma")
        logging.info("Chroma loaded/built successfully.")
    except Exception as e:
        logging.error(f"Failed building/loading Chroma: {e}")


@app.get("/")
def root():
    return {"message": "Welcome to the RAG Agent API. Visit /docs for usage."}

@app.get("/health")
def health_check():
    return {"status": "ok"}