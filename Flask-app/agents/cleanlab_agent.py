import requests
from agents.base_agent import BaseDockerAgent

class CleanlabDockerAgent(BaseDockerAgent):
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
                    "query": f"Cleanlab Confidence-Scored: {query}"
                },
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return f"Cleanlab Confidence-Scored Response: {result.get('answer', 'No answer found')}"
        except Exception as e:
            return f"Cleanlab Error: {str(e)}"