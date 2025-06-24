from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

from Parser.command_Parser import get_parser
from Prompt.prompts import rag_prompt

# Shared parser and prompt
parser = get_parser()
prompt_template = ChatPromptTemplate.from_template(rag_prompt)


def build_rag_retrieval_chain(llm, vector_store):
    """
    Given an LLM instance and a vector store, returns a RAG retrieval chain.
    """
    # 1) Create the chain that “stuff”s docs into the LLM with your prompt
    document_chain = create_stuff_documents_chain(
        llm,
        prompt_template,
        output_parser=parser,
    )

    # 2) Get a retriever from the vector store
    retriever = vector_store.as_retriever()

    # 3) Combine into a single retrieval chain
    return create_retrieval_chain(retriever, document_chain)
