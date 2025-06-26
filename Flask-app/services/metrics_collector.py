import sqlite3
import time
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from threading import Lock
import requests
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
import structlog

logger = structlog.get_logger()

@dataclass
class MetricData:
    timestamp: datetime
    trace_id: str
    framework: str
    model: str
    vector_store: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    latency_ms: float
    status: str
    error_message: Optional[str] = None

class MetricsCollector:
    """Real-time metrics collector with SQLite persistence and Prometheus export"""
    
    def __init__(self, db_path: str = "metrics.db", langtrace_api_key: Optional[str] = None):
        self.db_path = db_path
        self.langtrace_api_key = langtrace_api_key
        self.lock = Lock()
        
        # Token costs per 1K tokens
        self.token_costs = {
            'gpt-4o': {'input': 0.005, 'output': 0.015},
            'gpt-4o-mini': {'input': 0.00015, 'output': 0.0006},
            'gpt-3.5-turbo': {'input': 0.0015, 'output': 0.002},
            'llama3-8b-8192': {'input': 0.0005, 'output': 0.0008},
            'gemma2-9b-it': {'input': 0.0002, 'output': 0.0002},
            'llama-3.3-70b-versatile': {'input': 0.0009, 'output': 0.0009},
            'gemini-2.0-flash': {'input': 0.00075, 'output': 0.003}
        }
        
        # Initialize database
        self._init_database()
        
        # Initialize Prometheus metrics
        self.registry = CollectorRegistry()
        self._init_prometheus_metrics()
        
    def _init_database(self):
        """Initialize SQLite database with metrics table"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    trace_id TEXT NOT NULL,
                    framework TEXT NOT NULL,
                    model TEXT NOT NULL,
                    vector_store TEXT NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    input_cost REAL NOT NULL,
                    output_cost REAL NOT NULL,
                    total_cost REAL NOT NULL,
                    latency_ms REAL NOT NULL,
                    status TEXT NOT NULL,
                    error_message TEXT
                )
            """)
            conn.commit()
            
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics"""
        self.prom_request_count = Counter(
            'llm_requests_total',
            'Total LLM requests',
            ['framework', 'model', 'vector_store', 'status'],
            registry=self.registry
        )
        
        self.prom_token_count = Counter(
            'llm_tokens_total',
            'Total LLM tokens used',
            ['framework', 'model', 'token_type'],
            registry=self.registry
        )
        
        self.prom_cost_total = Counter(
            'llm_cost_total',
            'Total LLM cost in USD',
            ['framework', 'model', 'cost_type'],
            registry=self.registry
        )
        
        self.prom_latency = Histogram(
            'llm_latency_seconds',
            'LLM request latency',
            ['framework', 'model'],
            registry=self.registry
        )
        
        self.prom_active_requests = Gauge(
            'llm_active_requests',
            'Currently active LLM requests',
            registry=self.registry
        )
        
    def collect_from_langtrace(self, trace_id: str) -> Optional[MetricData]:
        """Collect metrics from Langtrace API"""
        if not self.langtrace_api_key:
            return None
            
        try:
            headers = {
                'Authorization': f'Bearer {self.langtrace_api_key}',
                'Content-Type': 'application/json'
            }
            
            response = requests.get(
                f'https://api.langtrace.ai/v1/traces/{trace_id}',
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_langtrace_data(data)
            else:
                logger.warning(f"Langtrace API returned {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to collect from Langtrace: {e}")
            return None
            
    def _parse_langtrace_data(self, data: Dict[str, Any]) -> Optional[MetricData]:
        """Parse Langtrace API response into MetricData"""
        try:
            # Extract relevant fields from Langtrace response
            # This is a simplified parser - adjust based on actual Langtrace API structure
            spans = data.get('spans', [])
            if not spans:
                return None
                
            main_span = spans[0]
            attributes = main_span.get('attributes', {})
            
            input_tokens = attributes.get('llm.usage.prompt_tokens', 0)
            output_tokens = attributes.get('llm.usage.completion_tokens', 0)
            model = attributes.get('llm.model', 'unknown')
            
            pricing = self.token_costs.get(model, self.token_costs['gpt-4o-mini'])
            input_cost = (input_tokens / 1000) * pricing['input']
            output_cost = (output_tokens / 1000) * pricing['output']
            
            return MetricData(
                timestamp=datetime.now(),
                trace_id=data.get('trace_id', ''),
                framework=attributes.get('framework', 'unknown'),
                model=model,
                vector_store=attributes.get('vector_store', 'unknown'),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                input_cost=input_cost,
                output_cost=output_cost,
                total_cost=input_cost + output_cost,
                latency_ms=main_span.get('duration_ms', 0),
                status='completed' if main_span.get('status') == 'OK' else 'failed'
            )
            
        except Exception as e:
            logger.error(f"Failed to parse Langtrace data: {e}")
            return None
            
    def collect_manual_metrics(self, trace_data: Dict[str, Any]) -> MetricData:
        """Manually calculate metrics from trace data (fallback)"""
        try:
            model = trace_data.get('model', 'gpt-4o-mini')
            pricing = self.token_costs.get(model, self.token_costs['gpt-4o-mini'])
            
            # Estimate tokens if not provided
            query = trace_data.get('query', '')
            response = trace_data.get('response', '')
            
            input_tokens = trace_data.get('input_tokens', len(query.split()) * 1.3)  # Rough estimation
            output_tokens = trace_data.get('output_tokens', len(response.split()) * 1.3)
            
            input_cost = (input_tokens / 1000) * pricing['input']
            output_cost = (output_tokens / 1000) * pricing['output']
            
            return MetricData(
                timestamp=datetime.now(),
                trace_id=trace_data.get('trace_id', ''),
                framework=trace_data.get('framework', 'unknown'),
                model=model,
                vector_store=trace_data.get('vector_store', 'unknown'),
                input_tokens=int(input_tokens),
                output_tokens=int(output_tokens),
                total_tokens=int(input_tokens + output_tokens),
                input_cost=input_cost,
                output_cost=output_cost,
                total_cost=input_cost + output_cost,
                latency_ms=trace_data.get('duration', 0) * 1000,
                status=trace_data.get('status', 'completed'),
                error_message=trace_data.get('error')
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate manual metrics: {e}")
            # Return default metrics to avoid breaking the system
            return MetricData(
                timestamp=datetime.now(),
                trace_id=trace_data.get('trace_id', ''),
                framework=trace_data.get('framework', 'unknown'),
                model=trace_data.get('model', 'gpt-4o-mini'),
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
            
    def record_metrics(self, trace_data: Dict[str, Any]) -> MetricData:
        """Record metrics with Langtrace fallback to manual calculation"""
        with self.lock:
            trace_id = trace_data.get('trace_id', '')
            
            # Try Langtrace first
            metrics = self.collect_from_langtrace(trace_id)
            
            # Fallback to manual calculation
            if metrics is None:
                metrics = self.collect_manual_metrics(trace_data)
                logger.info(f"Using manual metrics calculation for trace {trace_id}")
            else:
                logger.info(f"Collected metrics from Langtrace for trace {trace_id}")
            
            # Store in database
            self._store_metrics(metrics)
            
            # Update Prometheus metrics
            self._update_prometheus_metrics(metrics)
            
            return metrics
            
    def _store_metrics(self, metrics: MetricData):
        """Store metrics in SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO metrics (
                    timestamp, trace_id, framework, model, vector_store,
                    input_tokens, output_tokens, total_tokens,
                    input_cost, output_cost, total_cost,
                    latency_ms, status, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.timestamp.isoformat(),
                metrics.trace_id,
                metrics.framework,
                metrics.model,
                metrics.vector_store,
                metrics.input_tokens,
                metrics.output_tokens,
                metrics.total_tokens,
                metrics.input_cost,
                metrics.output_cost,
                metrics.total_cost,
                metrics.latency_ms,
                metrics.status,
                metrics.error_message
            ))
            conn.commit()
            
    def _update_prometheus_metrics(self, metrics: MetricData):
        """Update Prometheus metrics"""
        labels = {
            'framework': metrics.framework,
            'model': metrics.model,
            'vector_store': metrics.vector_store
        }
        
        # Request count
        self.prom_request_count.labels(
            **labels,
            status=metrics.status
        ).inc()
        
        # Token counts
        self.prom_token_count.labels(
            **{k: v for k, v in labels.items() if k != 'vector_store'},
            token_type='input'
        ).inc(metrics.input_tokens)
        
        self.prom_token_count.labels(
            **{k: v for k, v in labels.items() if k != 'vector_store'},
            token_type='output'
        ).inc(metrics.output_tokens)
        
        # Costs
        self.prom_cost_total.labels(
            **{k: v for k, v in labels.items() if k != 'vector_store'},
            cost_type='input'
        ).inc(metrics.input_cost)
        
        self.prom_cost_total.labels(
            **{k: v for k, v in labels.items() if k != 'vector_store'},
            cost_type='output'
        ).inc(metrics.output_cost)
        
        # Latency
        self.prom_latency.labels(
            **{k: v for k, v in labels.items() if k != 'vector_store'}
        ).observe(metrics.latency_ms / 1000)  # Convert to seconds
        
    def get_real_time_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """Get real-time metrics from database"""
        since = datetime.now() - timedelta(hours=hours)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Get all metrics within time range
            cursor = conn.execute("""
                SELECT * FROM metrics 
                WHERE timestamp >= ? 
                ORDER BY timestamp DESC
            """, (since.isoformat(),))
            
            rows = cursor.fetchall()
            
            if not rows:
                return self._empty_metrics_response()
            
            # Calculate aggregated metrics
            total_requests = len(rows)
            successful_requests = len([r for r in rows if r['status'] == 'completed'])
            failed_requests = total_requests - successful_requests
            
            total_input_tokens = sum(r['input_tokens'] for r in rows)
            total_output_tokens = sum(r['output_tokens'] for r in rows)
            total_tokens = total_input_tokens + total_output_tokens
            
            total_input_cost = sum(r['input_cost'] for r in rows)
            total_output_cost = sum(r['output_cost'] for r in rows)
            total_cost = total_input_cost + total_output_cost
            
            avg_latency = sum(r['latency_ms'] for r in rows) / total_requests if total_requests > 0 else 0
            
            # Time series data for charts
            time_series = self._generate_time_series(rows, hours)
            
            return {
                'summary': {
                    'total_requests': total_requests,
                    'successful_requests': successful_requests,
                    'failed_requests': failed_requests,
                    'success_rate': (successful_requests / total_requests * 100) if total_requests > 0 else 0,
                    'total_input_tokens': total_input_tokens,
                    'total_output_tokens': total_output_tokens,
                    'total_tokens': total_tokens,
                    'total_input_cost': round(total_input_cost, 4),
                    'total_output_cost': round(total_output_cost, 4),
                    'total_cost': round(total_cost, 4),
                    'avg_latency_ms': round(avg_latency, 2)
                },
                'time_series': time_series,
                'recent_traces': [dict(row) for row in rows[:10]]  # Last 10 traces
            }
            
    def _empty_metrics_response(self) -> Dict[str, Any]:
        """Return empty metrics response when no data available"""
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
        
    def _generate_time_series(self, rows: List[sqlite3.Row], hours: int) -> Dict[str, List]:
        """Generate time series data for charts"""
        # Group data by hour
        time_buckets = {}
        now = datetime.now()
        
        # Initialize buckets
        for i in range(hours):
            bucket_time = now - timedelta(hours=i)
            bucket_key = bucket_time.strftime('%Y-%m-%d %H:00')
            time_buckets[bucket_key] = {
                'input_tokens': 0,
                'output_tokens': 0,
                'input_cost': 0.0,
                'output_cost': 0.0,
                'latencies': [],
                'request_count': 0
            }
        
        # Fill buckets with data
        for row in rows:
            timestamp = datetime.fromisoformat(row['timestamp'])
            bucket_key = timestamp.strftime('%Y-%m-%d %H:00')
            
            if bucket_key in time_buckets:
                bucket = time_buckets[bucket_key]
                bucket['input_tokens'] += row['input_tokens']
                bucket['output_tokens'] += row['output_tokens']
                bucket['input_cost'] += row['input_cost']
                bucket['output_cost'] += row['output_cost']
                bucket['latencies'].append(row['latency_ms'])
                bucket['request_count'] += 1
        
        # Convert to chart format
        sorted_keys = sorted(time_buckets.keys())
        
        return {
            'labels': [key.split(' ')[1] for key in sorted_keys],  # Just the hour part
            'input_tokens': [time_buckets[key]['input_tokens'] for key in sorted_keys],
            'output_tokens': [time_buckets[key]['output_tokens'] for key in sorted_keys],
            'total_tokens': [
                time_buckets[key]['input_tokens'] + time_buckets[key]['output_tokens'] 
                for key in sorted_keys
            ],
            'input_costs': [round(time_buckets[key]['input_cost'], 4) for key in sorted_keys],
            'output_costs': [round(time_buckets[key]['output_cost'], 4) for key in sorted_keys],
            'total_costs': [
                round(time_buckets[key]['input_cost'] + time_buckets[key]['output_cost'], 4) 
                for key in sorted_keys
            ],
            'latencies': [
                round(sum(time_buckets[key]['latencies']) / len(time_buckets[key]['latencies']), 2)
                if time_buckets[key]['latencies'] else 0
                for key in sorted_keys
            ],
            'request_counts': [time_buckets[key]['request_count'] for key in sorted_keys]
        }
        
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus formatted metrics"""
        return generate_latest(self.registry).decode('utf-8')
        
    def cleanup_old_metrics(self, days: int = 30):
        """Clean up metrics older than specified days"""
        cutoff = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM metrics WHERE timestamp < ?",
                (cutoff.isoformat(),)
            )
            deleted_count = cursor.rowcount
            conn.commit()
            
        logger.info(f"Cleaned up {deleted_count} old metric records")
        return deleted_count

# Global metrics collector instance
metrics_collector = MetricsCollector()