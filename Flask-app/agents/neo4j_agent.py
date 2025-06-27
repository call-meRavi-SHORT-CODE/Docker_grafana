import requests
from agents.base_agent import BaseDockerAgent

class Neo4jDockerAgent(BaseDockerAgent):
    def setup(self):
        self.rag_api_url = "http://localhost:8000"
    
    def run(self, query: str) -> str:
        try:
            response = requests.post(
                f"{self.rag_api_url}/ask",
                json={
                    "framework": "langgraph",  # Fallback
                    "llm_model": "gpt-4o-mini",
                    "vector_store": "faiss",
                    "query": f"Neo4j Graph RAG: {query}"
                },
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return f"Neo4j Graph RAG Response: {result.get('answer', 'No answer found')}"
        except Exception as e:
            return f"Neo4j Error: {str(e)}"