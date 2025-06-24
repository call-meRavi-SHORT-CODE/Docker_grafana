// Global application JavaScript

// Initialize application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    // Check API health
    checkApiHealth();
    
    // Initialize real-time updates
    initializeRealTimeUpdates();
    
    // Initialize interactive elements
    initializeInteractiveElements();
    
    console.log('Docker Agent application initialized');
}

// API Health Check
async function checkApiHealth() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        
        const statusIndicator = document.getElementById('status-indicator');
        if (statusIndicator) {
            if (data.status === 'healthy') {
                statusIndicator.innerHTML = '<i class="fas fa-circle"></i> Online';
                statusIndicator.style.color = '#10b981';
            } else {
                statusIndicator.innerHTML = '<i class="fas fa-exclamation-triangle"></i> Issues';
                statusIndicator.style.color = '#f59e0b';
            }
        }
    } catch (error) {
        console.error('Health check failed:', error);
        const statusIndicator = document.getElementById('status-indicator');
        if (statusIndicator) {
            statusIndicator.innerHTML = '<i class="fas fa-times-circle"></i> Offline';
            statusIndicator.style.color = '#ef4444';
        }
    }
}

// Real-time Updates
function initializeRealTimeUpdates() {
    // Update metrics every 30 seconds on metrics page
    if (window.location.pathname.includes('/metrics')) {
        setInterval(updateMetrics, 30000);
    }
    
    // Update traces every 15 seconds on traces page
    if (window.location.pathname.includes('/traces')) {
        setInterval(updateTraces, 15000);
    }
    
    // Update logs every 10 seconds on logs page
    if (window.location.pathname.includes('/logs')) {
        setInterval(updateLogs, 10000);
    }
}

// Update functions
async function updateMetrics() {
    try {
        const response = await fetch('/api/metrics');
        const data = await response.json();
        
        // Update metric values in the DOM
        updateMetricValues(data);
    } catch (error) {
        console.error('Failed to update metrics:', error);
    }
}

async function updateTraces() {
    try {
        const response = await fetch('/api/traces');
        const data = await response.json();
        
        // Update traces display
        updateTracesDisplay(data.traces);
    } catch (error) {
        console.error('Failed to update traces:', error);
    }
}

async function updateLogs() {
    // Refresh logs page
    if (window.location.pathname.includes('/logs')) {
        location.reload();
    }
}

function updateMetricValues(metrics) {
    // Update summary stats
    const elements = {
        'total-requests': metrics.total_requests,
        'success-rate': metrics.success_rate.toFixed(1) + '%',
        'avg-duration': metrics.average_duration.toFixed(2) + 's',
        'total-tokens': metrics.total_tokens_used,
        'active-requests': metrics.active_requests,
        'completed-requests': metrics.completed_requests,
        'failed-requests': metrics.failed_requests
    };
    
    Object.entries(elements).forEach(([id, value]) => {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = value;
        }
    });
}

function updateTracesDisplay(traces) {
    // This would update the traces grid without full page reload
    console.log('Traces updated:', traces.length);
}

// Interactive Elements
function initializeInteractiveElements() {
    // Add click handlers for expandable elements
    document.querySelectorAll('.expandable').forEach(element => {
        element.addEventListener('click', function() {
            this.classList.toggle('expanded');
        });
    });
    
    // Add copy functionality to code blocks
    document.querySelectorAll('pre').forEach(pre => {
        const button = document.createElement('button');
        button.className = 'copy-button';
        button.innerHTML = '<i class="fas fa-copy"></i>';
        button.title = 'Copy to clipboard';
        button.style.cssText = `
            position: absolute;
            top: 0.5rem;
            right: 0.5rem;
            background: rgba(0,0,0,0.7);
            color: white;
            border: none;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.8rem;
        `;
        
        button.addEventListener('click', function() {
            navigator.clipboard.writeText(pre.textContent).then(() => {
                button.innerHTML = '<i class="fas fa-check"></i>';
                setTimeout(() => {
                    button.innerHTML = '<i class="fas fa-copy"></i>';
                }, 2000);
            });
        });
        
        pre.style.position = 'relative';
        pre.appendChild(button);
    });
}

// Utility Functions
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `flash-message flash-${type}`;
    notification.innerHTML = `
        <i class="fas fa-${type === 'error' ? 'exclamation-triangle' : 'info-circle'}"></i>
        ${message}
        <button class="flash-close" onclick="this.parentElement.remove()">×</button>
    `;
    
    let container = document.querySelector('.flash-messages');
    if (!container) {
        container = document.createElement('div');
        container.className = 'flash-messages';
        document.body.appendChild(container);
    }
    
    container.appendChild(notification);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (notification.parentElement) {
            notification.remove();
        }
    }, 5000);
}

function formatDuration(seconds) {
    if (seconds < 1) {
        return (seconds * 1000).toFixed(0) + 'ms';
    } else if (seconds < 60) {
        return seconds.toFixed(2) + 's';
    } else {
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = (seconds % 60).toFixed(0);
        return `${minutes}m ${remainingSeconds}s`;
    }
}

function formatTimestamp(timestamp) {
    const date = new Date(timestamp);
    return date.toLocaleString();
}

// Export functions for global use
window.DockerAgent = {
    showNotification,
    formatDuration,
    formatTimestamp,
    checkApiHealth,
    updateMetrics,
    updateTraces,
    updateLogs
};

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + Enter to submit form
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        const form = document.querySelector('form');
        if (form) {
            form.submit();
        }
    }
    
    // Escape to close modals/overlays
    if (e.key === 'Escape') {
        const overlay = document.querySelector('.loading-overlay');
        if (overlay && overlay.style.display === 'flex') {
            overlay.style.display = 'none';
        }
    }
});

// Handle form submissions with loading states
document.addEventListener('submit', function(e) {
    const form = e.target;
    const submitButton = form.querySelector('button[type="submit"]');
    
    if (submitButton) {
        submitButton.disabled = true;
        const originalText = submitButton.innerHTML;
        submitButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
        
        // Show loading overlay if it exists
        const overlay = document.getElementById('loading-overlay');
        if (overlay) {
            overlay.style.display = 'flex';
        }
        
        // Reset button after 30 seconds as fallback
        setTimeout(() => {
            submitButton.disabled = false;
            submitButton.innerHTML = originalText;
            if (overlay) {
                overlay.style.display = 'none';
            }
        }, 30000);
    }
});

// Handle network errors gracefully
window.addEventListener('online', function() {
    showNotification('Connection restored', 'success');
    checkApiHealth();
});

window.addEventListener('offline', function() {
    showNotification('Connection lost. Some features may not work.', 'error');
});

// Smooth scrolling for anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});