from typing import Dict, Any, List
from datetime import datetime, timedelta
import structlog
from .metrics_collector import metrics_collector, MetricData

logger = structlog.get_logger()

class EnhancedMetricsService:
    """Enhanced metrics service with real-time data collection"""
    
    def __init__(self):
        self.collector = metrics_collector
    
    def record_trace_metrics(self, trace_data: Dict[str, Any]) -> MetricData:
        """Record metrics for a trace execution"""
        try:
            return self.collector.record_metrics(trace_data)
        except Exception as e:
            logger.error(f"Failed to record trace metrics: {e}")
            # Return a default metric to avoid breaking the flow
            return MetricData(
                timestamp=datetime.now(),
                trace_id=trace_data.get('trace_id', ''),
                framework=trace_data.get('framework', 'unknown'),
                model=trace_data.get('model', 'unknown'),
                vector_store=trace_data.get('vector_store', 'unknown'),
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                input_cost=0.0,
                output_cost=0.0,
                total_cost=0.0,
                latency_ms=0.0,
                status='failed',
                error_message=str(e)
            )
    
    def get_real_time_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """Get real-time metrics for the specified time period"""
        try:
            return self.collector.get_real_time_metrics(hours)
        except Exception as e:
            logger.error(f"Failed to get real-time metrics: {e}")
            return self._get_fallback_metrics()
    
    def get_enhanced_metrics(self, days: int = 7) -> Dict[str, Any]:
        """Get enhanced metrics with backward compatibility"""
        try:
            # Get real-time data
            real_time_data = self.collector.get_real_time_metrics(days * 24)
            
            # Convert to the expected format for backward compatibility
            summary = real_time_data['summary']
            time_series = real_time_data['time_series']
            
            return {
                'session_id': 'real-time-session',
                'total_requests': summary['total_requests'],
                'completed_requests': summary['successful_requests'],
                'failed_requests': summary['failed_requests'],
                'success_rate': summary['success_rate'],
                'average_duration': summary['avg_latency_ms'] / 1000,  # Convert to seconds
                'total_tokens_used': summary['total_tokens'],
                'active_requests': 0,  # Real-time active requests would need separate tracking
                'token_usage': {
                    'total_input_tokens': summary['total_input_tokens'],
                    'total_output_tokens': summary['total_output_tokens'],
                    'total_tokens': summary['total_tokens'],
                    'daily_data': {
                        'labels': time_series['labels'],
                        'input_tokens': time_series['input_tokens'],
                        'output_tokens': time_series['output_tokens'],
                        'total_tokens': time_series['total_tokens']
                    }
                },
                'cost_analysis': {
                    'total_input_cost': summary['total_input_cost'],
                    'total_output_cost': summary['total_output_cost'],
                    'total_cost': summary['total_cost'],
                    'daily_data': {
                        'labels': time_series['labels'],
                        'input_costs': time_series['input_costs'],
                        'output_costs': time_series['output_costs'],
                        'total_costs': time_series['total_costs']
                    }
                },
                'latency_analysis': {
                    'avg_latency_ms': summary['avg_latency_ms'],
                    'max_latency_ms': max(time_series['latencies']) if time_series['latencies'] else 0,
                    'min_latency_ms': min(time_series['latencies']) if time_series['latencies'] else 0,
                    'daily_data': {
                        'labels': time_series['labels'],
                        'latencies': time_series['latencies']
                    }
                },
                'time_range_days': days
            }
        except Exception as e:
            logger.error(f"Failed to get enhanced metrics: {e}")
            return self._get_fallback_enhanced_metrics(days)
    
    def get_token_usage_data(self, days: int = 7) -> Dict[str, Any]:
        """Get token usage data"""
        try:
            real_time_data = self.collector.get_real_time_metrics(days * 24)
            time_series = real_time_data['time_series']
            
            return {
                'labels': time_series['labels'],
                'input_tokens': time_series['input_tokens'],
                'output_tokens': time_series['output_tokens'],
                'total_tokens': time_series['total_tokens']
            }
        except Exception as e:
            logger.error(f"Failed to get token usage data: {e}")
            return {'labels': [], 'input_tokens': [], 'output_tokens': [], 'total_tokens': []}
    
    def get_cost_data(self, days: int = 7, model: str = 'gpt-4o-mini') -> Dict[str, Any]:
        """Get cost data"""
        try:
            real_time_data = self.collector.get_real_time_metrics(days * 24)
            time_series = real_time_data['time_series']
            
            return {
                'labels': time_series['labels'],
                'input_costs': time_series['input_costs'],
                'output_costs': time_series['output_costs'],
                'total_costs': time_series['total_costs'],
                'model': model,
                'pricing': self.collector.token_costs.get(model, self.collector.token_costs['gpt-4o-mini'])
            }
        except Exception as e:
            logger.error(f"Failed to get cost data: {e}")
            return {
                'labels': [], 'input_costs': [], 'output_costs': [], 'total_costs': [],
                'model': model, 'pricing': {'input': 0.001, 'output': 0.002}
            }
    
    def get_latency_data(self, days: int = 7) -> Dict[str, Any]:
        """Get latency data"""
        try:
            real_time_data = self.collector.get_real_time_metrics(days * 24)
            time_series = real_time_data['time_series']
            
            latencies = time_series['latencies']
            return {
                'labels': time_series['labels'],
                'latencies': latencies,
                'avg_latency': sum(latencies) / len(latencies) if latencies else 0,
                'max_latency': max(latencies) if latencies else 0,
                'min_latency': min(latencies) if latencies else 0
            }
        except Exception as e:
            logger.error(f"Failed to get latency data: {e}")
            return {'labels': [], 'latencies': [], 'avg_latency': 0, 'max_latency': 0, 'min_latency': 0}
    
    def get_model_usage_breakdown(self) -> Dict[str, Any]:
        """Get model usage breakdown from real data"""
        try:
            # Get recent data for model breakdown
            real_time_data = self.collector.get_real_time_metrics(24 * 7)  # Last 7 days
            recent_traces = real_time_data['recent_traces']
            
            model_stats = {}
            for trace in recent_traces:
                model = trace['model']
                if model not in model_stats:
                    model_stats[model] = {
                        'requests': 0,
                        'tokens': 0,
                        'cost': 0.0,
                        'total_latency': 0.0,
                        'count': 0
                    }
                
                stats = model_stats[model]
                stats['requests'] += 1
                stats['tokens'] += trace['total_tokens']
                stats['cost'] += trace['total_cost']
                stats['total_latency'] += trace['latency_ms']
                stats['count'] += 1
            
            # Calculate averages
            for model, stats in model_stats.items():
                if stats['count'] > 0:
                    stats['avg_latency'] = int(stats['total_latency'] / stats['count'])
                else:
                    stats['avg_latency'] = 0
                del stats['total_latency']
                del stats['count']
                stats['cost'] = round(stats['cost'], 4)
            
            return model_stats
        except Exception as e:
            logger.error(f"Failed to get model usage breakdown: {e}")
            return {}
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus formatted metrics"""
        try:
            return self.collector.get_prometheus_metrics()
        except Exception as e:
            logger.error(f"Failed to get Prometheus metrics: {e}")
            return ""
    
    def cleanup_old_data(self, days: int = 30) -> int:
        """Clean up old metrics data"""
        try:
            return self.collector.cleanup_old_metrics(days)
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            return 0
    
    def _get_fallback_metrics(self) -> Dict[str, Any]:
        """Fallback metrics when real-time collection fails"""
        return {
            'summary': {
                'total_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'success_rate': 0,
                'total_input_tokens': 0,
                'total_output_tokens': 0,
                'total_tokens': 0,
                'total_input_cost': 0.0,
                'total_output_cost': 0.0,
                'total_cost': 0.0,
                'avg_latency_ms': 0.0
            },
            'time_series': {
                'labels': [],
                'input_tokens': [],
                'output_tokens': [],
                'total_tokens': [],
                'input_costs': [],
                'output_costs': [],
                'total_costs': [],
                'latencies': [],
                'request_counts': []
            },
            'recent_traces': []
        }
    
    def _get_fallback_enhanced_metrics(self, days: int) -> Dict[str, Any]:
        """Fallback enhanced metrics for backward compatibility"""
        return {
            'session_id': 'fallback-session',
            'total_requests': 0,
            'completed_requests': 0,
            'failed_requests': 0,
            'success_rate': 0,
            'average_duration': 0,
            'total_tokens_used': 0,
            'active_requests': 0,
            'token_usage': {
                'total_input_tokens': 0,
                'total_output_tokens': 0,
                'total_tokens': 0,
                'daily_data': {'labels': [], 'input_tokens': [], 'output_tokens': [], 'total_tokens': []}
            },
            'cost_analysis': {
                'total_input_cost': 0.0,
                'total_output_cost': 0.0,
                'total_cost': 0.0,
                'daily_data': {'labels': [], 'input_costs': [], 'output_costs': [], 'total_costs': []}
            },
            'latency_analysis': {
                'avg_latency_ms': 0,
                'max_latency_ms': 0,
                'min_latency_ms': 0,
                'daily_data': {'labels': [], 'latencies': []}
            },
            'time_range_days': days
        }

# Global enhanced metrics service instance
enhanced_metrics_service = EnhancedMetricsService()