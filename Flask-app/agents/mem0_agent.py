import requests
from agents.base_agent import BaseDockerAgent

class Mem0DockerAgent(BaseDockerAgent):
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
                    "query": f"Mem0 Memory-Enhanced: {query}"
                },
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return f"Mem0 Memory-Enhanced Response: {result.get('answer', 'No answer found')}"
        except Exception as e:
            return f"Mem0 Error: {str(e)}"