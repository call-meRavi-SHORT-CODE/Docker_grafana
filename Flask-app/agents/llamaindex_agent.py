import requests
from agents.base_agent import BaseDockerAgent

class LlamaIndexDockerAgent(BaseDockerAgent):
    def setup(self):
        self.rag_api_url = "http://localhost:8000"
    
    def run(self, query: str) -> str:
        try:
            response = requests.post(
                f"{self.rag_api_url}/ask",
                json={
                    "framework": "llamaindex",
                    "llm_model": "gpt-4o-mini",
                    "vector_store": "faiss", 
                    "query": query
                },
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return result.get('answer', 'No answer found')
        except Exception as e:
            return f"LlamaIndex Error: {str(e)}"