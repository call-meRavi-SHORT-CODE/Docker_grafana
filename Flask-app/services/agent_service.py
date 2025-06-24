from typing import Dict, Any, Optional
from core.registry import registry
from core.tracing import tracing_manager
from config.settings import settings
import structlog

logger = structlog.get_logger()

class AgentService:
    """Service for managing agent interactions with tracing"""
    
    def __init__(self):
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize and register all components"""
        # Import and register adapters
        from adapters import (
            LangGraphAdapter, LlamaIndexAdapter, 
            DSPyAdapter, AutoGenAdapter
        )
        
        registry.register_framework(LangGraphAdapter)
        registry.register_framework(LlamaIndexAdapter)
        registry.register_framework(DSPyAdapter)
        registry.register_framework(AutoGenAdapter)
    
    def execute_query(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a query with full tracing"""
        trace_id = tracing_manager.start_trace(request_data)
        
        try:
            # Get framework adapter
            framework_name = request_data.get('framework', '').lower()
            adapter = registry.get_framework(framework_name)
            
            tracing_manager.add_step(trace_id, 'framework_initialization', {
                'framework': framework_name,
                'adapter_class': adapter.__class__.__name__
            })
            
            # Create agent
            agent_config = {
                'model': request_data.get('model'),
                'vector_store': request_data.get('vector_store'),
                'query': request_data.get('query')
            }
            
            agent = adapter.create_agent(agent_config)
            
            tracing_manager.add_step(trace_id, 'agent_creation', {
                'config': agent_config,
                'agent_type': type(agent).__name__
            })
            
            # Execute query
            query = request_data.get('query', '')
            result = adapter.execute_query(agent, query)
            
            tracing_manager.add_step(trace_id, 'query_execution', {
                'query_length': len(query),
                'response_length': len(result.get('answer', '')),
                'duration': result.get('duration', 0),
                'tokens': result.get('tokens_used', 0),
                'status': result.get('status', 'unknown')
            })
            
            # Clean response
            cleaned_response = self._clean_response(result.get('answer', ''))
            
            final_result = {
                'answer': cleaned_response,
                'trace_id': trace_id,
                'framework': framework_name,
                'model': request_data.get('model'),
                'vector_store': request_data.get('vector_store'),
                'duration': result.get('duration', 0),
                'tokens_used': result.get('tokens_used', 0),
                'status': result.get('status', 'success')
            }
            
            tracing_manager.end_trace(trace_id, 'completed')
            
            logger.info(
                "Query executed successfully",
                trace_id=trace_id,
                framework=framework_name,
                duration=result.get('duration', 0)
            )
            
            return final_result
            
        except Exception as e:
            tracing_manager.end_trace(trace_id, 'failed', str(e))
            
            logger.error(
                "Query execution failed",
                trace_id=trace_id,
                error=str(e),
                framework=request_data.get('framework')
            )
            
            return {
                'answer': f"❌ Error: {str(e)}",
                'trace_id': trace_id,
                'framework': request_data.get('framework'),
                'model': request_data.get('model'),
                'vector_store': request_data.get('vector_store'),
                'duration': 0,
                'tokens_used': 0,
                'status': 'error',
                'error': str(e)
            }
    
    def _clean_response(self, text: str) -> str:
        """Clean and format the response"""
        if not text:
            return "No response generated."
        
        # Remove query repetition
        if text.startswith("Here"):
            start_index = text.find("Here")
            if start_index != -1:
                text = text[start_index:].strip()
        
        # Additional cleaning logic can be added here
        return text.strip()
    
    def get_available_configurations(self) -> Dict[str, Any]:
        """Get all available configurations"""
        return {
            'frameworks': registry.get_available_frameworks(),
            'models': settings.MODELS,
            'vector_stores': settings.VECTORSTORES
        }

# Global service instance
agent_service = AgentService()