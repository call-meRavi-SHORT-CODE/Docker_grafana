





import logging
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import logging
from config import DATA_DIR,CHROMA_INDEX_DIR,OPENAI_API_KEY,embeddings




embeddings = embeddings
logging.basicConfig(level=logging.INFO)

OPENAI_API_KEY = OPENAI_API_KEY

def build_chroma_index():
    logging.info("Building new Chroma index from PDFs and Docker CLI docs…")

    # 1. Load PDF documents
    pdf_loader = PyPDFDirectoryLoader(DATA_DIR)
    pdf_docs = pdf_loader.load()

    # 2. Define Docker CLI documentation URLs and attempt to load them
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

    # 3. Combine and split documents
    all_docs = pdf_docs + cli_docs
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(all_docs)

   
    db = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=CHROMA_INDEX_DIR
    )
    db.persist()  # Actually write the index files to disk

    logging.info("Chroma index built and persisted to %s", CHROMA_INDEX_DIR)
    return db
