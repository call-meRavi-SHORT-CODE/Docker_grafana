from typing import Any, Dict
from langchain.tools import tool

@tool
def doc_qa_tool(query: str, chain: Any) -> str:
    """
    Runs the RAG retrieval chain on `query` and returns the answer.
    """
    result: Dict[str, Any] = chain.invoke({"input": query})
    return result.get("answer", "No answer found.")
