from flask import Blueprint, render_template, request, redirect, url_for, flash
from services.agent_service import agent_service
from core.tracing import tracing_manager
from config.settings import settings
import structlog

logger = structlog.get_logger()

web_bp = Blueprint('web', __name__)

@web_bp.route('/')
def index():
    """Main playground page"""
    configs = agent_service.get_available_configurations()
    
    return render_template(
        'playground.html',
        frameworks=configs['frameworks'],
        models=configs['models'],
        vectorstores=configs['vector_stores'],
        selected_fw=configs['frameworks'][0] if configs['frameworks'] else '',
        selected_m=configs['models'][0] if configs['models'] else '',
        selected_vs=configs['vector_stores'][0] if configs['vector_stores'] else '',
        prompt_text='',
        response=None
    )

@web_bp.route('/generate', methods=['POST'])
def generate():
    """Handle form submission"""
    try:
        # Get form data
        framework = request.form.get('framework')
        model = request.form.get('model')
        vector_store = request.form.get('vector_store')
        query = request.form.get('prompt_text', '').strip()
        
        if not query:
            flash('Please enter a query', 'error')
            return redirect(url_for('web.index'))
        
        # Prepare request data
        request_data = {
            'framework': framework,
            'model': model,
            'vector_store': vector_store,
            'query': query
        }
        
        # Execute query
        result = agent_service.execute_query(request_data)
        
        # Get configurations for form
        configs = agent_service.get_available_configurations()
        
        return render_template(
            'playground.html',
            frameworks=configs['frameworks'],
            models=configs['models'],
            vectorstores=configs['vector_stores'],
            selected_fw=framework,
            selected_m=model,
            selected_vs=vector_store,
            prompt_text=query,
            response=result['answer'],
            trace_id=result.get('trace_id'),
            duration=result.get('duration', 0),
            tokens_used=result.get('tokens_used', 0),
            status=result.get('status', 'unknown')
        )
        
    except Exception as e:
        logger.error("Web generate error", error=str(e))
        flash(f'Error: {str(e)}', 'error')
        return redirect(url_for('web.index'))

@web_bp.route('/traces')
def traces():
    """Traces page"""
    traces = tracing_manager.get_all_traces()
    return render_template('traces.html', traces=traces)

@web_bp.route('/traces/<trace_id>')
def trace_detail(trace_id):
    """Trace detail page"""
    trace = tracing_manager.get_trace(trace_id)
    if not trace:
        flash('Trace not found', 'error')
        return redirect(url_for('web.traces'))
    
    return render_template('trace_detail.html', trace=trace)

@web_bp.route('/metrics')
def metrics():
    """Metrics page"""
    metrics = tracing_manager.get_metrics_summary()
    configs = agent_service.get_available_configurations()
    
    return render_template(
        'metrics.html',
        metrics=metrics,
        frameworks=configs['frameworks'],
        models=configs['models'],
        vectorstores=configs['vector_stores']
    )

@web_bp.route('/logs')
def logs():
    """Logs page"""
    # Get recent traces for log display
    traces = tracing_manager.get_all_traces()
    recent_traces = sorted(traces, key=lambda x: x.get('timestamp', ''), reverse=True)[:50]
    
    return render_template('logs.html', traces=recent_traces)