
from langchain_community.document_loaders import PyPDFDirectoryLoader, WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
from config import DATA_DIR,FAISS_INDEX_DIR,OPENAI_API_KEY,embeddings




embeddings = embeddings
logging.basicConfig(level=logging.INFO)

OPENAI_API_KEY = OPENAI_API_KEY



def build_faiss_index():
    logging.info("Building new FAISS index from PDFs and Docker CLI docs…")

    pdf_loader = PyPDFDirectoryLoader(DATA_DIR)
    pdf_docs = pdf_loader.load()

    
    cli_urls = [
            "https://docs.docker.com/engine/reference/commandline/ps/",
            "https://docs.docker.com/engine/reference/commandline/logs/",
            "https://docs.docker.com/engine/reference/commandline/stop/",
            "https://docs.docker.com/engine/reference/commandline/images_prune/",
            "https://docs.docker.com/engine/reference/commandline/service_scale/"
        ]
    cli_loader = WebBaseLoader(web_paths=tuple(cli_urls))
    try:
        cli_docs = cli_loader.load()
    except Exception as e:
        logging.warning(f"Failed to load CLI docs: {e}")
        cli_docs = [] 

    # Combine and split
    all_docs = pdf_docs + cli_docs
    all_docs = pdf_docs
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(all_docs)

    # Build FAISS index
    db = FAISS.from_documents(split_docs, embeddings)
    db.save_local(FAISS_INDEX_DIR)
    logging.info("Index built and saved to %s", FAISS_INDEX_DIR)
    return db