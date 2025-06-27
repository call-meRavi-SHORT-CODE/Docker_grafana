import sqlite3
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from contextlib import contextmanager
import json
import threading

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Centralized database manager for persistent storage"""
    
    def __init__(self, db_path: str = "docker_agent.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_database()
    
    def _init_database(self):
        """Initialize all required tables"""
        with self.get_connection() as conn:
            # Traces table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS traces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT UNIQUE NOT NULL,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    end_time TEXT,
                    status TEXT NOT NULL DEFAULT 'started',
                    framework TEXT NOT NULL,
                    model TEXT NOT NULL,
                    vector_store TEXT NOT NULL,
                    query TEXT NOT NULL,
                    response TEXT,
                    total_duration REAL,
                    input_tokens INTEGER DEFAULT 0,
                    output_tokens INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    input_cost REAL DEFAULT 0.0,
                    output_cost REAL DEFAULT 0.0,
                    total_cost REAL DEFAULT 0.0,
                    error_message TEXT,
                    request_data TEXT,
                    steps TEXT,
                    metrics TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Metrics table for detailed tracking
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    trace_id TEXT NOT NULL,
                    framework TEXT NOT NULL,
                    model TEXT NOT NULL,
                    vector_store TEXT NOT NULL,
                    input_tokens INTEGER NOT NULL DEFAULT 0,
                    output_tokens INTEGER NOT NULL DEFAULT 0,
                    total_tokens INTEGER NOT NULL DEFAULT 0,
                    input_cost REAL NOT NULL DEFAULT 0.0,
                    output_cost REAL NOT NULL DEFAULT 0.0,
                    total_cost REAL NOT NULL DEFAULT 0.0,
                    latency_ms REAL NOT NULL DEFAULT 0.0,
                    status TEXT NOT NULL,
                    error_message TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (trace_id) REFERENCES traces (trace_id)
                )
            """)
            
            # Sessions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    last_activity TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Framework health table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS framework_health (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    framework_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    last_check TEXT NOT NULL,
                    error_message TEXT,
                    test_passed BOOLEAN DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for better performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_traces_timestamp ON traces(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_traces_status ON traces(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_traces_framework ON traces(framework)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_trace_id ON metrics(trace_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_framework_health_name ON framework_health(framework_name)")
            
            conn.commit()
            logger.info("Database initialized successfully")
    
    @contextmanager
    def get_connection(self):
        """Get database connection with proper error handling"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.row_factory = sqlite3.Row
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def save_trace(self, trace_data: Dict[str, Any]) -> bool:
        """Save or update trace data"""
        with self.lock:
            try:
                with self.get_connection() as conn:
                    # Check if trace exists
                    existing = conn.execute(
                        "SELECT id FROM traces WHERE trace_id = ?",
                        (trace_data['trace_id'],)
                    ).fetchone()
                    
                    if existing:
                        # Update existing trace
                        conn.execute("""
                            UPDATE traces SET
                                end_time = ?,
                                status = ?,
                                response = ?,
                                total_duration = ?,
                                input_tokens = ?,
                                output_tokens = ?,
                                total_tokens = ?,
                                input_cost = ?,
                                output_cost = ?,
                                total_cost = ?,
                                error_message = ?,
                                steps = ?,
                                metrics = ?
                            WHERE trace_id = ?
                        """, (
                            trace_data.get('end_time'),
                            trace_data.get('status', 'started'),
                            trace_data.get('response'),
                            trace_data.get('total_duration'),
                            trace_data.get('input_tokens', 0),
                            trace_data.get('output_tokens', 0),
                            trace_data.get('total_tokens', 0),
                            trace_data.get('input_cost', 0.0),
                            trace_data.get('output_cost', 0.0),
                            trace_data.get('total_cost', 0.0),
                            trace_data.get('error_message'),
                            json.dumps(trace_data.get('steps', [])),
                            json.dumps(trace_data.get('metrics', {})),
                            trace_data['trace_id']
                        ))
                    else:
                        # Insert new trace
                        conn.execute("""
                            INSERT INTO traces (
                                trace_id, session_id, timestamp, status, framework,
                                model, vector_store, query, response, total_duration,
                                input_tokens, output_tokens, total_tokens,
                                input_cost, output_cost, total_cost,
                                error_message, request_data, steps, metrics
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            trace_data['trace_id'],
                            trace_data.get('session_id', 'default'),
                            trace_data.get('timestamp', datetime.now().isoformat()),
                            trace_data.get('status', 'started'),
                            trace_data.get('framework', 'unknown'),
                            trace_data.get('model', 'unknown'),
                            trace_data.get('vector_store', 'unknown'),
                            trace_data.get('query', ''),
                            trace_data.get('response'),
                            trace_data.get('total_duration'),
                            trace_data.get('input_tokens', 0),
                            trace_data.get('output_tokens', 0),
                            trace_data.get('total_tokens', 0),
                            trace_data.get('input_cost', 0.0),
                            trace_data.get('output_cost', 0.0),
                            trace_data.get('total_cost', 0.0),
                            trace_data.get('error_message'),
                            json.dumps(trace_data.get('request_data', {})),
                            json.dumps(trace_data.get('steps', [])),
                            json.dumps(trace_data.get('metrics', {}))
                        ))
                    
                    conn.commit()
                    return True
            except Exception as e:
                logger.error(f"Failed to save trace: {e}")
                return False
    
    def save_metrics(self, metrics_data: Dict[str, Any]) -> bool:
        """Save metrics data"""
        with self.lock:
            try:
                with self.get_connection() as conn:
                    conn.execute("""
                        INSERT INTO metrics (
                            timestamp, trace_id, framework, model, vector_store,
                            input_tokens, output_tokens, total_tokens,
                            input_cost, output_cost, total_cost,
                            latency_ms, status, error_message
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        metrics_data.get('timestamp', datetime.now().isoformat()),
                        metrics_data.get('trace_id', ''),
                        metrics_data.get('framework', 'unknown'),
                        metrics_data.get('model', 'unknown'),
                        metrics_data.get('vector_store', 'unknown'),
                        metrics_data.get('input_tokens', 0),
                        metrics_data.get('output_tokens', 0),
                        metrics_data.get('total_tokens', 0),
                        metrics_data.get('input_cost', 0.0),
                        metrics_data.get('output_cost', 0.0),
                        metrics_data.get('total_cost', 0.0),
                        metrics_data.get('latency_ms', 0.0),
                        metrics_data.get('status', 'completed'),
                        metrics_data.get('error_message')
                    ))
                    conn.commit()
                    return True
            except Exception as e:
                logger.error(f"Failed to save metrics: {e}")
                return False
    
    def save_framework_health(self, health_data: Dict[str, Any]) -> bool:
        """Save framework health data"""
        with self.lock:
            try:
                with self.get_connection() as conn:
                    for framework_name, health_info in health_data.items():
                        conn.execute("""
                            INSERT INTO framework_health (
                                framework_name, status, last_check, error_message, test_passed
                            ) VALUES (?, ?, ?, ?, ?)
                        """, (
                            framework_name,
                            health_info.get('status', 'unknown'),
                            datetime.now().isoformat(),
                            health_info.get('error'),
                            health_info.get('test_passed', False)
                        ))
                    conn.commit()
                    return True
            except Exception as e:
                logger.error(f"Failed to save framework health: {e}")
                return False
    
    def get_traces(self, limit: int = 100, status: str = None) -> List[Dict[str, Any]]:
        """Get traces with optional filtering"""
        try:
            with self.get_connection() as conn:
                query = "SELECT * FROM traces"
                params = []
                
                if status:
                    query += " WHERE status = ?"
                    params.append(status)
                
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                rows = conn.execute(query, params).fetchall()
                
                traces = []
                for row in rows:
                    trace = dict(row)
                    # Parse JSON fields
                    trace['request_data'] = json.loads(trace['request_data'] or '{}')
                    trace['steps'] = json.loads(trace['steps'] or '[]')
                    trace['metrics'] = json.loads(trace['metrics'] or '{}')
                    traces.append(trace)
                
                return traces
        except Exception as e:
            logger.error(f"Failed to get traces: {e}")
            return []
    
    def get_trace_by_id(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Get specific trace by ID"""
        try:
            with self.get_connection() as conn:
                row = conn.execute(
                    "SELECT * FROM traces WHERE trace_id = ?",
                    (trace_id,)
                ).fetchone()
                
                if row:
                    trace = dict(row)
                    trace['request_data'] = json.loads(trace['request_data'] or '{}')
                    trace['steps'] = json.loads(trace['steps'] or '[]')
                    trace['metrics'] = json.loads(trace['metrics'] or '{}')
                    return trace
                return None
        except Exception as e:
            logger.error(f"Failed to get trace {trace_id}: {e}")
            return None
    
    def get_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get metrics summary for specified time period"""
        try:
            since = (datetime.now() - timedelta(hours=hours)).isoformat()
            
            with self.get_connection() as conn:
                # Get basic counts
                summary_row = conn.execute("""
                    SELECT 
                        COUNT(*) as total_requests,
                        COUNT(CASE WHEN status = 'completed' THEN 1 END) as successful_requests,
                        COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_requests,
                        COALESCE(SUM(input_tokens), 0) as total_input_tokens,
                        COALESCE(SUM(output_tokens), 0) as total_output_tokens,
                        COALESCE(SUM(total_tokens), 0) as total_tokens,
                        COALESCE(SUM(input_cost), 0.0) as total_input_cost,
                        COALESCE(SUM(output_cost), 0.0) as total_output_cost,
                        COALESCE(SUM(total_cost), 0.0) as total_cost,
                        COALESCE(AVG(latency_ms), 0.0) as avg_latency_ms
                    FROM metrics 
                    WHERE timestamp >= ?
                """, (since,)).fetchone()
                
                if summary_row:
                    summary = dict(summary_row)
                    summary['success_rate'] = (
                        (summary['successful_requests'] / summary['total_requests'] * 100) 
                        if summary['total_requests'] > 0 else 0
                    )
                    return summary
                
                return {
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
                }
        except Exception as e:
            logger.error(f"Failed to get metrics summary: {e}")
            return {}
    
    def get_time_series_data(self, hours: int = 24) -> Dict[str, List]:
        """Get time series data for charts"""
        try:
            since = (datetime.now() - timedelta(hours=hours)).isoformat()
            
            with self.get_connection() as conn:
                # Group by hour
                rows = conn.execute("""
                    SELECT 
                        strftime('%Y-%m-%d %H:00', timestamp) as hour_bucket,
                        COUNT(*) as request_count,
                        COALESCE(SUM(input_tokens), 0) as input_tokens,
                        COALESCE(SUM(output_tokens), 0) as output_tokens,
                        COALESCE(SUM(total_tokens), 0) as total_tokens,
                        COALESCE(SUM(input_cost), 0.0) as input_cost,
                        COALESCE(SUM(output_cost), 0.0) as output_cost,
                        COALESCE(SUM(total_cost), 0.0) as total_cost,
                        COALESCE(AVG(latency_ms), 0.0) as avg_latency
                    FROM metrics 
                    WHERE timestamp >= ?
                    GROUP BY hour_bucket
                    ORDER BY hour_bucket
                """, (since,)).fetchall()
                
                # Convert to chart format
                labels = []
                request_counts = []
                input_tokens = []
                output_tokens = []
                total_tokens = []
                input_costs = []
                output_costs = []
                total_costs = []
                latencies = []
                
                for row in rows:
                    labels.append(row['hour_bucket'].split(' ')[1])  # Just hour part
                    request_counts.append(row['request_count'])
                    input_tokens.append(row['input_tokens'])
                    output_tokens.append(row['output_tokens'])
                    total_tokens.append(row['total_tokens'])
                    input_costs.append(round(row['input_cost'], 4))
                    output_costs.append(round(row['output_cost'], 4))
                    total_costs.append(round(row['total_cost'], 4))
                    latencies.append(round(row['avg_latency'], 2))
                
                return {
                    'labels': labels,
                    'request_counts': request_counts,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'total_tokens': total_tokens,
                    'input_costs': input_costs,
                    'output_costs': output_costs,
                    'total_costs': total_costs,
                    'latencies': latencies
                }
        except Exception as e:
            logger.error(f"Failed to get time series data: {e}")
            return {
                'labels': [],
                'request_counts': [],
                'input_tokens': [],
                'output_tokens': [],
                'total_tokens': [],
                'input_costs': [],
                'output_costs': [],
                'total_costs': [],
                'latencies': []
            }
    
    def get_recent_traces(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent traces for activity display"""
        try:
            with self.get_connection() as conn:
                rows = conn.execute("""
                    SELECT trace_id, timestamp, framework, model, total_tokens, 
                           total_cost, latency_ms, status
                    FROM metrics 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (limit,)).fetchall()
                
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get recent traces: {e}")
            return []
    
    def get_framework_health_history(self, hours: int = 24) -> Dict[str, Any]:
        """Get framework health history"""
        try:
            since = (datetime.now() - timedelta(hours=hours)).isoformat()
            
            with self.get_connection() as conn:
                rows = conn.execute("""
                    SELECT framework_name, status, last_check, test_passed, error_message
                    FROM framework_health 
                    WHERE last_check >= ?
                    ORDER BY last_check DESC
                """, (since,)).fetchall()
                
                health_data = {}
                for row in rows:
                    framework = row['framework_name']
                    if framework not in health_data:
                        health_data[framework] = []
                    
                    health_data[framework].append({
                        'status': row['status'],
                        'timestamp': row['last_check'],
                        'test_passed': bool(row['test_passed']),
                        'error': row['error_message']
                    })
                
                return health_data
        except Exception as e:
            logger.error(f"Failed to get framework health history: {e}")
            return {}
    
    def cleanup_old_data(self, days: int = 30) -> int:
        """Clean up old data"""
        try:
            cutoff = (datetime.now() - timedelta(days=days)).isoformat()
            
            with self.get_connection() as conn:
                # Delete old metrics
                cursor = conn.execute(
                    "DELETE FROM metrics WHERE timestamp < ?",
                    (cutoff,)
                )
                metrics_deleted = cursor.rowcount
                
                # Delete old traces
                cursor = conn.execute(
                    "DELETE FROM traces WHERE timestamp < ?",
                    (cutoff,)
                )
                traces_deleted = cursor.rowcount
                
                # Delete old framework health
                cursor = conn.execute(
                    "DELETE FROM framework_health WHERE last_check < ?",
                    (cutoff,)
                )
                health_deleted = cursor.rowcount
                
                conn.commit()
                
                total_deleted = metrics_deleted + traces_deleted + health_deleted
                logger.info(f"Cleaned up {total_deleted} old records")
                return total_deleted
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            return 0

# Global database manager instance
db_manager = DatabaseManager()