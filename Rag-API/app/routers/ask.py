from fastapi import APIRouter, HTTPException
from app.models import RAGRequest, RAGResponse
from app.services.vector_store import get_vector_store
from app.services.llm import get_llm, get_llama_index_llm
from app.services.rag_chain import build_rag_retrieval_chain
from app.services.frameworks import get_agent
import logging
import asyncio

router = APIRouter()

@router.post("/ask", response_model=RAGResponse)
def ask(request: RAGRequest):
    try:
        # Initialize component
        vector_store = get_vector_store(request.vector_store)

        if request.framework == "langgraph":
            llm = get_llm(request.llm_model)
            rag_chain = build_rag_retrieval_chain(llm, vector_store)
            agent = get_agent("langgraph", llm, rag_chain)

            inputs = {"messages": [("user", request.query)]}
            response_text = ""
            for step in agent.stream(inputs, stream_mode="values"):
                msg = step["messages"][-1]
                response_text += msg.content

            return RAGResponse(answer=response_text)

        elif request.framework == "llamaindex":
            llm = get_llama_index_llm(request.llm_model)
            rag_chain = build_rag_retrieval_chain(llm, vector_store)
            agent = get_agent("llamaindex", llm, rag_chain)

            async def main():
                return await agent.run(user_msg=request.query)

            response_text = asyncio.run(main())
            return RAGResponse(answer=str(response_text))

        elif request.framework == "dspy":
            llm = get_llm(request.llm_model)
            rag_chain = build_rag_retrieval_chain(llm, vector_store)
            agent = get_agent("dspy", llm, rag_chain)

            pred = agent(question=request.query)
            return RAGResponse(answer=str(pred.answer))

        else:
            raise HTTPException(status_code=400, detail="Invalid framework selected")

    except Exception as e:
        logging.exception("Error inside /ask:")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
