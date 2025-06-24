from langchain_community.document_loaders import PyPDFDirectoryLoader, WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
from config import DATA_DIR, FAISS_INDEX_DIR, OPENAI_API_KEY, embeddings

logging.basicConfig(level=logging.INFO)

def build_faiss_index():
    logging.info("Building new FAISS index from PDFs and Docker CLI docs…")

    # Load PDFs
    try:
        pdf_loader = PyPDFDirectoryLoader(DATA_DIR)
        pdf_docs = pdf_loader.load()
        logging.info(f"Loaded {len(pdf_docs)} PDF documents.")
    except Exception as e:
        logging.error(f"Failed to load PDFs: {e}")
        pdf_docs = []

    # Load Docker CLI Docs from Web
    cli_urls = [
        "https://docs.docker.com/engine/reference/commandline/ps/",
        "https://docs.docker.com/engine/reference/commandline/logs/",
        "https://docs.docker.com/engine/reference/commandline/stop/",
        "https://docs.docker.com/engine/reference/commandline/images_prune/",
        "https://docs.docker.com/engine/reference/commandline/service_scale/"
    ]
    try:
        cli_loader = WebBaseLoader(web_paths=tuple(cli_urls))
        cli_docs = cli_loader.load()
        logging.info(f"Loaded {len(cli_docs)} CLI documents.")
    except Exception as e:
        logging.warning(f"Failed to load CLI docs: {e}")
        cli_docs = []

    # Combine all documents
    all_docs = pdf_docs + cli_docs

    if not all_docs:
        logging.error("No documents found to index. Aborting FAISS index build.")
        raise ValueError("Cannot build FAISS index with zero documents.")

    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(all_docs)
    logging.info(f"Split into {len(split_docs)} document chunks.")

    # Build and save FAISS index
    try:
        db = FAISS.from_documents(split_docs, embeddings)
        db.save_local(FAISS_INDEX_DIR)
        logging.info(f"FAISS index built and saved to {FAISS_INDEX_DIR}")
        return db
    except Exception as e:
        logging.exception("Failed to build or save FAISS index.")
        raise
