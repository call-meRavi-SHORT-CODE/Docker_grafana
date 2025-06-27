import requests
from agents.base_agent import BaseDockerAgent

class AutoGenDockerAgent(BaseDockerAgent):
    def setup(self):
        self.rag_api_url = "http://localhost:8000"
    
    def run(self, query: str) -> str:
        try:
            # AutoGen implementation would go here
            response = requests.post(
                f"{self.rag_api_url}/ask",
                json={
                    "framework": "langgraph",  # Fallback to langgraph
                    "llm_model": "gpt-4o-mini", 
                    "vector_store": "faiss",
                    "query": f"AutoGen Conversational: {query}"
                },
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return f"AutoGen Conversational Response: {result.get('answer', 'No answer found')}"
        except Exception as e:
            return f"AutoGen Error: {str(e)}"