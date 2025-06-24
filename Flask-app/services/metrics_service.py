from typing import Dict, Any, List
from datetime import datetime, timedelta
import random
from core.tracing import tracing_manager

class MetricsService:
    """Service for generating and managing performance metrics"""
    
    def __init__(self):
        self.token_costs = {
            'gpt-4o': {'input': 0.005, 'output': 0.015},  # per 1K tokens
            'gpt-4o-mini': {'input': 0.00015, 'output': 0.0006},
            'gpt-3.5-turbo': {'input': 0.0015, 'output': 0.002},
            'llama3-8b-8192': {'input': 0.0005, 'output': 0.0008},
            'gemma2-9b-it': {'input': 0.0002, 'output': 0.0002},
            'llama-3.3-70b-versatile': {'input': 0.0009, 'output': 0.0009},
            'gemini-2.0-flash': {'input': 0.00075, 'output': 0.003}
        }
    
    def get_token_usage_data(self, days: int = 7) -> Dict[str, Any]:
        """Generate token usage data for the specified number of days"""
        labels = []
        input_tokens = []
        output_tokens = []
        total_tokens = []
        
        for i in range(days - 1, -1, -1):
            date = datetime.now() - timedelta(days=i)
            labels.append(date.strftime('%Y-%m-%d'))
            
            # Generate realistic sample data based on actual usage patterns
            base_input = random.randint(800, 5000)
            base_output = random.randint(300, 2500)
            
            # Add some variance based on day of week (higher on weekdays)
            if date.weekday() < 5:  # Monday to Friday
                multiplier = random.uniform(1.2, 1.8)
            else:  # Weekend
                multiplier = random.uniform(0.6, 1.0)
            
            daily_input = int(base_input * multiplier)
            daily_output = int(base_output * multiplier)
            
            input_tokens.append(daily_input)
            output_tokens.append(daily_output)
            total_tokens.append(daily_input + daily_output)
        
        return {
            'labels': labels,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': total_tokens
        }
    
    def get_cost_data(self, days: int = 7, model: str = 'gpt-4o-mini') -> Dict[str, Any]:
        """Generate cost data based on token usage and model pricing"""
        token_data = self.get_token_usage_data(days)
        
        # Get pricing for the model
        pricing = self.token_costs.get(model, self.token_costs['gpt-4o-mini'])
        
        input_costs = []
        output_costs = []
        total_costs = []
        
        for i in range(len(token_data['labels'])):
            input_cost = (token_data['input_tokens'][i] / 1000) * pricing['input']
            output_cost = (token_data['output_tokens'][i] / 1000) * pricing['output']
            
            input_costs.append(round(input_cost, 4))
            output_costs.append(round(output_cost, 4))
            total_costs.append(round(input_cost + output_cost, 4))
        
        return {
            'labels': token_data['labels'],
            'input_costs': input_costs,
            'output_costs': output_costs,
            'total_costs': total_costs,
            'model': model,
            'pricing': pricing
        }
    
    def get_latency_data(self, days: int = 7) -> Dict[str, Any]:
        """Generate latency data for the specified number of days"""
        labels = []
        latencies = []
        
        for i in range(days - 1, -1, -1):
            date = datetime.now() - timedelta(days=i)
            labels.append(date.strftime('%Y-%m-%d'))
            
            # Generate realistic latency data (100-3000ms)
            base_latency = random.randint(200, 1500)
            
            # Add some variance and occasional spikes
            if random.random() < 0.1:  # 10% chance of spike
                latency = base_latency * random.uniform(2, 4)
            else:
                latency = base_latency * random.uniform(0.8, 1.2)
            
            latencies.append(int(latency))
        
        return {
            'labels': labels,
            'latencies': latencies,
            'avg_latency': sum(latencies) / len(latencies),
            'max_latency': max(latencies),
            'min_latency': min(latencies)
        }
    
    def get_enhanced_metrics(self, days: int = 7) -> Dict[str, Any]:
        """Get comprehensive metrics including token usage, costs, and latency"""
        # Get base metrics from tracing manager
        base_metrics = tracing_manager.get_metrics_summary()
        
        # Get enhanced data
        token_data = self.get_token_usage_data(days)
        cost_data = self.get_cost_data(days)
        latency_data = self.get_latency_data(days)
        
        # Calculate totals
        total_input_tokens = sum(token_data['input_tokens'])
        total_output_tokens = sum(token_data['output_tokens'])
        total_input_cost = sum(cost_data['input_costs'])
        total_output_cost = sum(cost_data['output_costs'])
        
        return {
            **base_metrics,
            'token_usage': {
                'total_input_tokens': total_input_tokens,
                'total_output_tokens': total_output_tokens,
                'total_tokens': total_input_tokens + total_output_tokens,
                'daily_data': token_data
            },
            'cost_analysis': {
                'total_input_cost': round(total_input_cost, 2),
                'total_output_cost': round(total_output_cost, 2),
                'total_cost': round(total_input_cost + total_output_cost, 2),
                'daily_data': cost_data
            },
            'latency_analysis': {
                'avg_latency_ms': round(latency_data['avg_latency']),
                'max_latency_ms': latency_data['max_latency'],
                'min_latency_ms': latency_data['min_latency'],
                'daily_data': latency_data
            },
            'time_range_days': days
        }
    
    def get_model_usage_breakdown(self) -> Dict[str, Any]:
        """Get breakdown of usage by model"""
        # This would typically come from actual usage data
        # For now, generating sample data
        models = ['gpt-4o-mini', 'gpt-4o', 'llama3-8b-8192', 'gemma2-9b-it']
        usage_data = {}
        
        total_requests = random.randint(100, 1000)
        
        for model in models:
            requests = random.randint(10, total_requests // 2)
            tokens = random.randint(1000, 50000)
            cost = (tokens / 1000) * self.token_costs.get(model, {}).get('input', 0.001)
            
            usage_data[model] = {
                'requests': requests,
                'tokens': tokens,
                'cost': round(cost, 2),
                'avg_latency': random.randint(200, 2000)
            }
        
        return usage_data

# Global service instance
metrics_service = MetricsService()