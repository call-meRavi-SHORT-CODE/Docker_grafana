import os
import time
import uuid
from typing import Dict, Any, Optional
from datetime import datetime
import structlog
from langtrace_python_sdk import langtrace
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import json

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

class TracingManager:
    def __init__(self, langtrace_api_key: Optional[str] = None):
        self.session_id = str(uuid.uuid4())
        self.traces = []
        
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
            }
        }
        
        self.traces.append(trace)
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
        trace = self._find_trace(trace_id)
        if trace:
            step = {
                'step_name': step_name,
                'timestamp': datetime.utcnow().isoformat(),
                'data': step_data,
                'duration': step_data.get('duration', 0)
            }
            trace['steps'].append(step)
            
            # Update metrics
            if 'tokens' in step_data:
                trace['metrics']['tokens_used'] += step_data['tokens']
                LLM_TOKEN_USAGE.labels(
                    framework=trace['request_data'].get('framework', 'unknown'),
                    model=trace['request_data'].get('model', 'unknown'),
                    token_type='total'
                ).inc(step_data['tokens'])
            
            if step_name in ['llm_call', 'vector_search', 'tool_execution']:
                trace['metrics']['api_calls'] += 1
            
            logger.info(
                "Step added to trace",
                trace_id=trace_id,
                step_name=step_name,
                duration=step_data.get('duration', 0)
            )
    
    def end_trace(self, trace_id: str, status: str = 'completed', error: Optional[str] = None):
        """End a trace and record metrics"""
        trace = self._find_trace(trace_id)
        if trace:
            end_time = time.time()
            duration = end_time - trace['metrics']['start_time']
            
            trace['status'] = status
            trace['end_time'] = datetime.utcnow().isoformat()
            trace['total_duration'] = duration
            
            if error:
                trace['error'] = error
                ERROR_COUNT.labels(
                    framework=trace['request_data'].get('framework', 'unknown'),
                    model=trace['request_data'].get('model', 'unknown'),
                    error_type=type(error).__name__ if isinstance(error, Exception) else 'unknown'
                ).inc()
            
            # Record metrics
            REQUEST_COUNT.labels(
                framework=trace['request_data'].get('framework', 'unknown'),
                model=trace['request_data'].get('model', 'unknown'),
                vector_store=trace['request_data'].get('vector_store', 'unknown'),
                status=status
            ).inc()
            
            REQUEST_DURATION.labels(
                framework=trace['request_data'].get('framework', 'unknown'),
                model=trace['request_data'].get('model', 'unknown'),
                vector_store=trace['request_data'].get('vector_store', 'unknown')
            ).observe(duration)
            
            ACTIVE_REQUESTS.dec()
            
            logger.info(
                "Trace completed",
                trace_id=trace_id,
                status=status,
                duration=duration,
                tokens_used=trace['metrics']['tokens_used'],
                api_calls=trace['metrics']['api_calls']
            )
    
    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific trace by ID"""
        return self._find_trace(trace_id)
    
    def get_all_traces(self) -> list:
        """Get all traces for the current session"""
        return self.traces
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of metrics"""
        total_requests = len(self.traces)
        completed_requests = len([t for t in self.traces if t['status'] == 'completed'])
        failed_requests = len([t for t in self.traces if t['status'] == 'failed'])
        
        avg_duration = 0
        total_tokens = 0
        
        if completed_requests > 0:
            durations = [t.get('total_duration', 0) for t in self.traces if t['status'] == 'completed']
            avg_duration = sum(durations) / len(durations)
            total_tokens = sum([t['metrics']['tokens_used'] for t in self.traces])
        
        return {
            'session_id': self.session_id,
            'total_requests': total_requests,
            'completed_requests': completed_requests,
            'failed_requests': failed_requests,
            'success_rate': (completed_requests / total_requests * 100) if total_requests > 0 else 0,
            'average_duration': avg_duration,
            'total_tokens_used': total_tokens,
            'active_requests': len([t for t in self.traces if t['status'] == 'started'])
        }
    
    def _find_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Find a trace by ID"""
        for trace in self.traces:
            if trace['trace_id'] == trace_id:
                return trace
        return None
    
    def export_traces_to_grafana(self, grafana_url: str, api_key: str):
        """Export traces to Grafana Cloud (placeholder for actual implementation)"""
        try:
            # This would be implemented based on your Grafana Cloud setup
            # For now, we'll log the traces in a format suitable for Grafana
            for trace in self.traces:
                logger.info(
                    "Grafana export",
                    trace_id=trace['trace_id'],
                    framework=trace['request_data'].get('framework'),
                    model=trace['request_data'].get('model'),
                    duration=trace.get('total_duration', 0),
                    status=trace['status'],
                    tokens_used=trace['metrics']['tokens_used']
                )
        except Exception as e:
            logger.error("Failed to export to Grafana", error=str(e))

# Global tracing manager instance
tracing_manager = TracingManager(os.getenv('LANGTRACE_API_KEY'))