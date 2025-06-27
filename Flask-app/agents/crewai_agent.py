import requests
from agents.base_agent import BaseDockerAgent

class CrewAIDockerAgent(BaseDockerAgent):
    def setup(self):
        self.rag_api_url = "http://localhost:8000"
    
    def run(self, query: str) -> str:
        try:
            # CrewAI implementation would go here
            # For now, using a placeholder that calls RAG API
            response = requests.post(
                f"{self.rag_api_url}/ask",
                json={
                    "framework": "langgraph",  # Fallback to langgraph
                    "llm_model": "gpt-4o-mini",
                    "vector_store": "faiss",
                    "query": f"CrewAI Multi-Agent: {query}"
                },
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return f"CrewAI Multi-Agent Response: {result.get('answer', 'No answer found')}"
        except Exception as e:
            return f"CrewAI Error: {str(e)}"