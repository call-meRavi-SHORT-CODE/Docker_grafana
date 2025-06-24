
import os
import json
import re
import logging
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.tools import DuckDuckGoSearchResults
from langgraph.prebuilt import create_react_agent
from langchain.tools import tool
from typing import Any, Dict
from typing import List, Literal
import subprocess


# imports form files
from vector_Store.faiss_index import build_faiss_index
from config import FAISS_INDEX_DIR,OPENAI_API_KEY,embeddings
from Parser.command_Parser import get_parser
from Prompt.prompts import system_prompt,rag_prompt
load_dotenv()



os.environ.setdefault("USER_AGENT", "Doc_search_Agent")
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"



OPENAI_API_KEY = OPENAI_API_KEY
embeddings = embeddings


if os.path.exists(FAISS_INDEX_DIR):
    logging.info("Loading existing FAISS index…")
    db = FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
else:
    db = build_faiss_index()




llm = init_chat_model("gpt-4o-mini", model_provider="openai")


prompt = ChatPromptTemplate.from_template(rag_prompt)


parser = get_parser()


# Creating a Custom Chain 
document_chain = create_stuff_documents_chain(
    llm,
    prompt,
    output_parser=parser,
)

# Retriver
retriever=db.as_retriever()


# combining Chain|Retriver as Retriver chain 
retrieval_chain= create_retrieval_chain(retriever,document_chain)

# TOOLS
@tool
def doc_qa(query: str) -> str:
    """Answer questions based on the Docker documentation context."""
    result: Dict[str, Any] = retrieval_chain.invoke({"input": query})
    return result.get('answer', 'No answer found.')

@tool
def run_command(cmd: str) -> str:
    """
    Execute a shell command (typically a Docker CLI invocation) 
    and return its stdout/stderr.

    Args:
        cmd (str): A fully formed Docker command, e.g. "docker run -d --name my_app nginx".

    Returns:
        str: The combined stdout / stderr of the subprocess. 
             Raises a RuntimeError if the command exits with a nonzero status.
    """
    print(f"\n[>] Executing: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Execution Failed:\n{result.stderr}")
    print(f"Output:\n{result.stdout}")
    return result.stdout


## Tool1 
tools = [doc_qa, run_command]



agent = create_react_agent(model=llm, tools=tools, prompt=system_prompt)


def run_agent(question: str):
    inputs = {"messages": [("user", question)]}
    for step in agent.stream(inputs, stream_mode="values"):
        msg = step['messages'][-1]
        msg.pretty_print()

# Example usage
if __name__ == "__main__":
    run_agent("show all the running docker containers")
  

#Create new container named my-app1 using the nginx:latest image in detached mode