import os
import time
import uuid
from typing import Dict, Any, Optional
from datetime import datetime
import structlog
from langtrace_python_sdk import langtrace
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import json
from services.database import db_manager
from services.token_calculator import token_calculator

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Prometheus Metrics
REQUEST_COUNT = Counter(
    'docker_agent_requests_total',
    'Total number of requests',
    ['framework', 'model', 'vector_store', 'status']
)

REQUEST_DURATION = Histogram(
    'docker_agent_request_duration_seconds',
    'Request duration in seconds',
    ['framework', 'model', 'vector_store']
)

ACTIVE_REQUESTS = Gauge(
    'docker_agent_active_requests',
    'Number of active requests'
)

ERROR_COUNT = Counter(
    'docker_agent_errors_total',
    'Total number of errors',
    ['framework', 'model', 'error_type']
)

LLM_TOKEN_USAGE = Counter(
    'docker_agent_llm_tokens_total',
    'Total LLM tokens used',
    ['framework', 'model', 'token_type']
)

LLM_COST_TOTAL = Counter(
    'docker_agent_llm_cost_total',
    'Total LLM cost in USD',
    ['framework', 'model', 'cost_type']
)

class TracingManager:
    def __init__(self, langtrace_api_key: Optional[str] = None):
        self.session_id = str(uuid.uuid4())
        
        # Initialize LangTrace if API key is provided
        if langtrace_api_key:
            try:
                langtrace.init(api_key=langtrace_api_key)
                logger.info("LangTrace initialized successfully")
            except Exception as e:
                logger.error("Failed to initialize LangTrace", error=str(e))
        
        # Start Prometheus metrics server
        try:
            start_http_server(8001)
            logger.info("Prometheus metrics server started on port 8001")
        except Exception as e:
            logger.error("Failed to start Prometheus server", error=str(e))
    
    def start_trace(self, request_data: Dict[str, Any]) -> str:
        """Start a new trace for a request"""
        trace_id = str(uuid.uuid4())
        
        trace = {
            'trace_id': trace_id,
            'session_id': self.session_id,
            'timestamp': datetime.utcnow().isoformat(),
            'request_data': request_data,
            'status': 'started',
            'steps': [],
            'metrics': {
                'start_time': time.time(),
                'tokens_used': 0,
                'api_calls': 0
            },
            'framework': request_data.get('framework', 'unknown'),
            'model': request_data.get('model', 'unknown'),
            'vector_store': request_data.get('vector_store', 'unknown'),
            'query': request_data.get('query', '')
        }
        
        # Save to database
        db_manager.save_trace(trace)
        ACTIVE_REQUESTS.inc()
        
        logger.info(
            "Trace started",
            trace_id=trace_id,
            framework=request_data.get('framework'),
            model=request_data.get('model'),
            vector_store=request_data.get('vector_store')
        )
        
        return trace_id
    
    def add_step(self, trace_id: str, step_name: str, step_data: Dict[str, Any]):
        """Add a step to an existing trace"""
        try:
            # Get existing trace
            trace = db_manager.get_trace_by_id(trace_id)
            if trace:
                step = {
                    'step_name': step_name,
                    'timestamp': datetime.utcnow().isoformat(),
                    'data': step_data,
                    'duration': step_data.get('duration', 0)
                }
                
                # Add step to trace
                steps = trace.get('steps', [])
                steps.append(step)
                trace['steps'] = steps
                
                # Update metrics
                if 'tokens' in step_data:
                    trace['metrics']['tokens_used'] += step_data['tokens']
                    LLM_TOKEN_USAGE.labels(
                        framework=trace.get('framework', 'unknown'),
                        model=trace.get('model', 'unknown'),
                        token_type='total'
                    ).inc(step_data['tokens'])
                
                if step_name in ['llm_call', 'vector_search', 'tool_execution']:
                    trace['metrics']['api_calls'] += 1
                
                # Save updated trace
                db_manager.save_trace(trace)
                
                logger.info(
                    "Step added to trace",
                    trace_id=trace_id,
                    step_name=step_name,
                    duration=step_data.get('duration', 0)
                )
        except Exception as e:
            logger.error(f"Failed to add step to trace {trace_id}: {e}")
    
    def end_trace(self, trace_id: str, status: str = 'completed', 
                  response: str = '', error: Optional[str] = None):
        """End a trace and record metrics"""
        try:
            trace = db_manager.get_trace_by_id(trace_id)
            if trace:
                end_time = time.time()
                duration = end_time - trace['metrics']['start_time']
                
                # Calculate tokens and costs
                query = trace.get('query', '')
                model = trace.get('model', 'gpt-4o-mini')
                
                # Try to extract actual tokens from response
                actual_input, actual_output = token_calculator.extract_tokens_from_response(response)
                
                # Calculate tokens and costs
                token_data = token_calculator.calculate_tokens_and_cost(
                    query=query,
                    response=response,
                    model=model,
                    actual_input_tokens=actual_input,
                    actual_output_tokens=actual_output
                )
                
                # Update trace
                trace.update({
                    'status': status,
                    'end_time': datetime.utcnow().isoformat(),
                    'total_duration': duration,
                    'response': response,
                    'input_tokens': token_data['input_tokens'],
                    'output_tokens': token_data['output_tokens'],
                    'total_tokens': token_data['total_tokens'],
                    'input_cost': token_data['input_cost'],
                    'output_cost': token_data['output_cost'],
                    'total_cost': token_data['total_cost'],
                    'error_message': error
                })
                
                # Save trace
                db_manager.save_trace(trace)
                
                # Save metrics
                metrics_data = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'trace_id': trace_id,
                    'framework': trace.get('framework', 'unknown'),
                    'model': trace.get('model', 'unknown'),
                    'vector_store': trace.get('vector_store', 'unknown'),
                    'input_tokens': token_data['input_tokens'],
                    'output_tokens': token_data['output_tokens'],
                    'total_tokens': token_data['total_tokens'],
                    'input_cost': token_data['input_cost'],
                    'output_cost': token_data['output_cost'],
                    'total_cost': token_data['total_cost'],
                    'latency_ms': duration * 1000,
                    'status': status,
                    'error_message': error
                }
                db_manager.save_metrics(metrics_data)
                
                # Update Prometheus metrics
                REQUEST_COUNT.labels(
                    framework=trace.get('framework', 'unknown'),
                    model=trace.get('model', 'unknown'),
                    vector_store=trace.get('vector_store', 'unknown'),
                    status=status
                ).inc()
                
                REQUEST_DURATION.labels(
                    framework=trace.get('framework', 'unknown'),
                    model=trace.get('model', 'unknown'),
                    vector_store=trace.get('vector_store', 'unknown')
                ).observe(duration)
                
                # Token metrics
                LLM_TOKEN_USAGE.labels(
                    framework=trace.get('framework', 'unknown'),
                    model=trace.get('model', 'unknown'),
                    token_type='input'
                ).inc(token_data['input_tokens'])
                
                LLM_TOKEN_USAGE.labels(
                    framework=trace.get('framework', 'unknown'),
                    model=trace.get('model', 'unknown'),
                    token_type='output'
                ).inc(token_data['output_tokens'])
                
                # Cost metrics
                LLM_COST_TOTAL.labels(
                    framework=trace.get('framework', 'unknown'),
                    model=trace.get('model', 'unknown'),
                    cost_type='input'
                ).inc(token_data['input_cost'])
                
                LLM_COST_TOTAL.labels(
                    framework=trace.get('framework', 'unknown'),
                    model=trace.get('model', 'unknown'),
                    cost_type='output'
                ).inc(token_data['output_cost'])
                
                if error:
                    ERROR_COUNT.labels(
                        framework=trace.get('framework', 'unknown'),
                        model=trace.get('model', 'unknown'),
                        error_type=type(error).__name__ if isinstance(error, Exception) else 'unknown'
                    ).inc()
                
                ACTIVE_REQUESTS.dec()
                
                logger.info(
                    "Trace completed",
                    trace_id=trace_id,
                    status=status,
                    duration=duration,
                    tokens_used=token_data['total_tokens'],
                    total_cost=token_data['total_cost'],
                    api_calls=trace['metrics']['api_calls']
                )
        except Exception as e:
            logger.error(f"Failed to end trace {trace_id}: {e}")
    
    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific trace by ID"""
        return db_manager.get_trace_by_id(trace_id)
    
    def get_all_traces(self) -> list:
        """Get all traces"""
        return db_manager.get_traces()
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of metrics from database"""
        try:
            summary = db_manager.get_metrics_summary(24)  # Last 24 hours
            
            # Add session info
            summary['session_id'] = self.session_id
            summary['active_requests'] = 0  # Real-time active requests would need separate tracking
            
            return summary
        except Exception as e:
            logger.error(f"Failed to get metrics summary: {e}")
            return {
                'session_id': self.session_id,
                'total_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'success_rate': 0,
                'average_duration': 0,
                'total_tokens_used': 0,
                'active_requests': 0
            }

# Global tracing manager instance
tracing_manager = TracingManager(os.getenv('LANGTRACE_API_KEY'))