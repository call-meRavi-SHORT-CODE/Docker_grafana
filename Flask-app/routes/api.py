from flask import Blueprint, request, jsonify
from services.agent_service import agent_service
from core.tracing import tracing_manager
import structlog

logger = structlog.get_logger()

api_bp = Blueprint('api', __name__, url_prefix='/api')

@api_bp.route('/generate', methods=['POST'])
def generate():
    """Generate response using selected configuration"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = ['framework', 'model', 'vector_store', 'query']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Execute query
        result = agent_service.execute_query(data)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error("API generate error", error=str(e))
        return jsonify({'error': str(e)}), 500

@api_bp.route('/configurations', methods=['GET'])
def get_configurations():
    """Get available configurations"""
    try:
        configs = agent_service.get_available_configurations()
        return jsonify(configs)
    except Exception as e:
        logger.error("API configurations error", error=str(e))
        return jsonify({'error': str(e)}), 500

@api_bp.route('/traces', methods=['GET'])
def get_traces():
    """Get all traces"""
    try:
        traces = tracing_manager.get_all_traces()
        return jsonify({'traces': traces})
    except Exception as e:
        logger.error("API traces error", error=str(e))
        return jsonify({'error': str(e)}), 500

@api_bp.route('/traces/<trace_id>', methods=['GET'])
def get_trace(trace_id):
    """Get specific trace"""
    try:
        trace = tracing_manager.get_trace(trace_id)
        if not trace:
            return jsonify({'error': 'Trace not found'}), 404
        return jsonify(trace)
    except Exception as e:
        logger.error("API trace error", error=str(e), trace_id=trace_id)
        return jsonify({'error': str(e)}), 500

@api_bp.route('/metrics', methods=['GET'])
def get_metrics():
    """Get metrics summary"""
    try:
        metrics = tracing_manager.get_metrics_summary()
        return jsonify(metrics)
    except Exception as e:
        logger.error("API metrics error", error=str(e))
        return jsonify({'error': str(e)}), 500

@api_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'docker-agent-flask'})