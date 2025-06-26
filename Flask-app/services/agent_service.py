from typing import Dict, Any, Optional
from core.registry import registry
from core.tracing import tracing_manager
from config.settings import settings
from .enhanced_metrics_service import enhanced_metrics_service
from .token_calculator import token_calculator
import structlog
import time
import re

logger = structlog.get_logger()

class AgentService:
    """Service for managing agent interactions with enhanced persistent storage"""
    
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
        """Execute a query with full tracing and persistent metrics collection"""
        trace_id = tracing_manager.start_trace(request_data)
        start_time = time.time()
        
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
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Extract response text
            raw_response = result.get('answer', '')
            cleaned_response = self._clean_response(raw_response)
            
            # Try to extract actual token counts from the response
            actual_input_tokens, actual_output_tokens = token_calculator.extract_tokens_from_response(raw_response)
            
            # Calculate accurate tokens and costs
            model = request_data.get('model', 'gpt-4o-mini')
            token_data = token_calculator.calculate_tokens_and_cost(
                query=query,
                response=cleaned_response,
                model=model,
                actual_input_tokens=actual_input_tokens,
                actual_output_tokens=actual_output_tokens
            )
            
            tracing_manager.add_step(trace_id, 'query_execution', {
                'query_length': len(query),
                'response_length': len(cleaned_response),
                'duration': duration,
                'input_tokens': token_data['input_tokens'],
                'output_tokens': token_data['output_tokens'],
                'total_tokens': token_data['total_tokens'],
                'input_cost': token_data['input_cost'],
                'output_cost': token_data['output_cost'],
                'total_cost': token_data['total_cost'],
                'status': 'completed'
            })
            
            final_result = {
                'answer': cleaned_response,
                'trace_id': trace_id,
                'framework': framework_name,
                'model': model,
                'vector_store': request_data.get('vector_store'),
                'duration': duration,
                'input_tokens': token_data['input_tokens'],
                'output_tokens': token_data['output_tokens'],
                'tokens_used': token_data['total_tokens'],
                'input_cost': token_data['input_cost'],
                'output_cost': token_data['output_cost'],
                'total_cost': token_data['total_cost'],
                'status': 'success'
            }
            
            # End trace with response and token data
            tracing_manager.end_trace(trace_id, 'completed', cleaned_response)
            
            logger.info(
                "Query executed successfully",
                trace_id=trace_id,
                framework=framework_name,
                model=model,
                duration=duration,
                tokens=token_data['total_tokens'],
                cost=token_data['total_cost']
            )
            
            return final_result
            
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            
            error_message = str(e)
            
            # Calculate tokens for failed request (query only)
            model = request_data.get('model', 'gpt-4o-mini')
            query = request_data.get('query', '')
            
            token_data = token_calculator.calculate_tokens_and_cost(
                query=query,
                response='',  # No response for failed request
                model=model
            )
            
            # End trace with error
            tracing_manager.end_trace(trace_id, 'failed', '', error_message)
            
            logger.error(
                "Query execution failed",
                trace_id=trace_id,
                error=error_message,
                framework=request_data.get('framework'),
                model=model,
                duration=duration
            )
            
            return {
                'answer': f"❌ Error: {error_message}",
                'trace_id': trace_id,
                'framework': request_data.get('framework'),
                'model': model,
                'vector_store': request_data.get('vector_store'),
                'duration': duration,
                'input_tokens': token_data['input_tokens'],
                'output_tokens': 0,
                'tokens_used': token_data['input_tokens'],
                'input_cost': token_data['input_cost'],
                'output_cost': 0.0,
                'total_cost': token_data['input_cost'],
                'status': 'error',
                'error': error_message
            }
    
    def _clean_response(self, text: str) -> str:
        """Clean and format the response"""
        if not text:
            return "No response generated."
        
        # Remove common prefixes that might be added by the system
        prefixes_to_remove = [
            r"^Here\s+",
            r"^(answer):\s*",
            r"^\(answer\):\s*",
            r"^Response:\s*",
            r"^Output:\s*"
        ]
        
        cleaned = text.strip()
        for prefix in prefixes_to_remove:
            cleaned = re.sub(prefix, "", cleaned, flags=re.IGNORECASE)
        
        # Remove excessive whitespace
        cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)
        cleaned = cleaned.strip()
        
        return cleaned if cleaned else "No response generated."
    
    def get_available_configurations(self) -> Dict[str, Any]:
        """Get all available configurations"""
        return {
            'frameworks': registry.get_available_frameworks(),
            'models': settings.MODELS,
            'vector_stores': settings.VECTORSTORES
        }

# Global service instance
agent_service = AgentService()