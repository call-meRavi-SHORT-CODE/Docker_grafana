from typing import Dict, Any, List
from datetime import datetime, timedelta
import structlog
from services.database import db_manager

logger = structlog.get_logger()

class EnhancedMetricsService:
    """Enhanced metrics service with persistent database storage"""
    
    def __init__(self):
        self.db = db_manager
    
    def get_real_time_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """Get real-time metrics for the specified time period"""
        try:
            # Get summary data
            summary = self.db.get_metrics_summary(hours)
            
            # Get time series data
            time_series = self.db.get_time_series_data(hours)
            
            # Get recent traces
            recent_traces = self.db.get_recent_traces(10)
            
            return {
                'summary': summary,
                'time_series': time_series,
                'recent_traces': recent_traces
            }
        except Exception as e:
            logger.error(f"Failed to get real-time metrics: {e}")
            return self._get_fallback_metrics()
    
    def get_enhanced_metrics(self, days: int = 7) -> Dict[str, Any]:
        """Get enhanced metrics with backward compatibility"""
        try:
            # Get real-time data
            real_time_data = self.get_real_time_metrics(days * 24)
            
            # Convert to the expected format for backward compatibility
            summary = real_time_data['summary']
            time_series = real_time_data['time_series']
            
            return {
                'session_id': 'persistent-session',
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
            time_series = self.db.get_time_series_data(days * 24)
            
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
            time_series = self.db.get_time_series_data(days * 24)
            
            # Get pricing info
            from services.token_calculator import token_calculator
            pricing = token_calculator.get_model_pricing(model)
            
            return {
                'labels': time_series['labels'],
                'input_costs': time_series['input_costs'],
                'output_costs': time_series['output_costs'],
                'total_costs': time_series['total_costs'],
                'model': model,
                'pricing': pricing
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
            time_series = self.db.get_time_series_data(days * 24)
            
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
        """Get model usage breakdown from database"""
        try:
            # Get data from last 7 days
            since = (datetime.now() - timedelta(days=7)).isoformat()
            
            with self.db.get_connection() as conn:
                rows = conn.execute("""
                    SELECT 
                        model,
                        COUNT(*) as requests,
                        SUM(total_tokens) as tokens,
                        SUM(total_cost) as cost,
                        AVG(latency_ms) as avg_latency
                    FROM metrics 
                    WHERE timestamp >= ?
                    GROUP BY model
                    ORDER BY requests DESC
                """, (since,)).fetchall()
                
                model_stats = {}
                for row in rows:
                    model_stats[row['model']] = {
                        'requests': row['requests'],
                        'tokens': row['tokens'] or 0,
                        'cost': round(row['cost'] or 0.0, 4),
                        'avg_latency': int(row['avg_latency'] or 0)
                    }
                
                return model_stats
        except Exception as e:
            logger.error(f"Failed to get model usage breakdown: {e}")
            return {}
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus formatted metrics"""
        try:
            # This would generate Prometheus format from database
            # For now, return basic format
            summary = self.db.get_metrics_summary(24)
            
            metrics = []
            metrics.append(f"# HELP docker_agent_total_requests Total requests processed")
            metrics.append(f"# TYPE docker_agent_total_requests counter")
            metrics.append(f"docker_agent_total_requests {summary.get('total_requests', 0)}")
            
            metrics.append(f"# HELP docker_agent_total_tokens Total tokens used")
            metrics.append(f"# TYPE docker_agent_total_tokens counter")
            metrics.append(f"docker_agent_total_tokens {summary.get('total_tokens', 0)}")
            
            metrics.append(f"# HELP docker_agent_total_cost Total cost in USD")
            metrics.append(f"# TYPE docker_agent_total_cost counter")
            metrics.append(f"docker_agent_total_cost {summary.get('total_cost', 0.0)}")
            
            return "\n".join(metrics)
        except Exception as e:
            logger.error(f"Failed to get Prometheus metrics: {e}")
            return ""
    
    def cleanup_old_data(self, days: int = 30) -> int:
        """Clean up old metrics data"""
        try:
            return self.db.cleanup_old_data(days)
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            return 0
    
    def _get_fallback_metrics(self) -> Dict[str, Any]:
        """Fallback metrics when database fails"""
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