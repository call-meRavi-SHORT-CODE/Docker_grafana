from typing import Dict, Any, List
from core.registry import FrameworkAdapter
import time
import requests

class DSPyAdapter(FrameworkAdapter):
    """DSPy framework adapter"""
    
    def get_name(self) -> str:
        return "dspy"
    
    def get_supported_models(self) -> List[str]:
        return [
            'gpt-4o', 'gpt-4o-mini', "gpt-4.1", "gpt-4.1-mini", 
            "gpt-3.5-turbo", 'llama3-8b-8192', 'gemma2-9b-it',
            "llama-3.3-70b-versatile", "gemini-2.0-flash"
        ]
    
    def create_agent(self, config: Dict[str, Any]) -> Any:
        """Create DSPy agent"""
        return {
            'framework': 'dspy',
            'model': config.get('model'),
            'vector_store': config.get('vector_store'),
            'config': config
        }
    
    def execute_query(self, agent: Any, query: str) -> Dict[str, Any]:
        """Execute query using DSPy agent"""
        start_time = time.time()
        
        try:
            payload = {
                "framework": "dspy",
                "llm_model": agent['model'],
                "vector_store": agent['vector_store'],
                "query": query
            }
            
            response = requests.post(
                "http://localhost:8000/ask",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            duration = time.time() - start_time
            
            return {
                'answer': result.get('answer', 'No answer found'),
                'duration': duration,
                'tokens_used': self._estimate_tokens(query + result.get('answer', '')),
                'status': 'success'
            }
            
        except Exception as e:
            duration = time.time() - start_time
            return {
                'answer': f"Error: {str(e)}",
                'duration': duration,
                'tokens_used': 0,
                'status': 'error',
                'error': str(e)
            }
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation"""
        return len(text) // 4