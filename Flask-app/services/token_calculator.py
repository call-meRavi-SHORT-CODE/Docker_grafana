import re
import tiktoken
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class TokenCalculator:
    """Accurate token calculation for different models"""
    
    def __init__(self):
        # Token costs per 1K tokens (input, output)
        self.token_costs = {
            'gpt-4o': {'input': 0.005, 'output': 0.015},
            'gpt-4o-mini': {'input': 0.00015, 'output': 0.0006},
            'gpt-3.5-turbo': {'input': 0.0015, 'output': 0.002},
            'gpt-4': {'input': 0.03, 'output': 0.06},
            'gpt-4-32k': {'input': 0.06, 'output': 0.12},
            'llama3-8b-8192': {'input': 0.0005, 'output': 0.0008},
            'gemma2-9b-it': {'input': 0.0002, 'output': 0.0002},
            'llama-3.3-70b-versatile': {'input': 0.0009, 'output': 0.0009},
            'gemini-2.0-flash': {'input': 0.00075, 'output': 0.003}
        }
        
        # Model to encoding mapping
        self.model_encodings = {
            'gpt-4o': 'cl100k_base',
            'gpt-4o-mini': 'cl100k_base',
            'gpt-3.5-turbo': 'cl100k_base',
            'gpt-4': 'cl100k_base',
            'gpt-4-32k': 'cl100k_base',
        }
    
    def count_tokens(self, text: str, model: str = 'gpt-4o-mini') -> int:
        """Count tokens in text for specific model"""
        if not text:
            return 0
            
        try:
            # Use tiktoken for OpenAI models
            if model in self.model_encodings:
                encoding = tiktoken.get_encoding(self.model_encodings[model])
                return len(encoding.encode(text))
            else:
                # Fallback estimation for other models
                return self._estimate_tokens(text)
        except Exception as e:
            logger.warning(f"Failed to count tokens for {model}: {e}")
            return self._estimate_tokens(text)
    
    def _estimate_tokens(self, text: str) -> int:
        """Fallback token estimation"""
        # Rough estimation: 1 token ≈ 4 characters for English text
        # This is a conservative estimate
        return max(1, len(text) // 4)
    
    def extract_tokens_from_response(self, response_text: str) -> Tuple[Optional[int], Optional[int]]:
        """Extract token counts from LLM response if available"""
        try:
            # Look for common token usage patterns in responses
            patterns = [
                r'input[_\s]*tokens?[:\s]*(\d+)',
                r'prompt[_\s]*tokens?[:\s]*(\d+)',
                r'completion[_\s]*tokens?[:\s]*(\d+)',
                r'output[_\s]*tokens?[:\s]*(\d+)',
                r'total[_\s]*tokens?[:\s]*(\d+)',
            ]
            
            input_tokens = None
            output_tokens = None
            
            for pattern in patterns:
                matches = re.findall(pattern, response_text, re.IGNORECASE)
                if matches:
                    if 'input' in pattern or 'prompt' in pattern:
                        input_tokens = int(matches[0])
                    elif 'output' in pattern or 'completion' in pattern:
                        output_tokens = int(matches[0])
            
            return input_tokens, output_tokens
        except Exception as e:
            logger.debug(f"Could not extract tokens from response: {e}")
            return None, None
    
    def calculate_tokens_and_cost(self, 
                                  query: str, 
                                  response: str, 
                                  model: str,
                                  actual_input_tokens: Optional[int] = None,
                                  actual_output_tokens: Optional[int] = None) -> Dict[str, any]:
        """Calculate tokens and costs with fallback to estimation"""
        
        # Use actual tokens if provided, otherwise estimate
        if actual_input_tokens is not None:
            input_tokens = actual_input_tokens
        else:
            input_tokens = self.count_tokens(query, model)
        
        if actual_output_tokens is not None:
            output_tokens = actual_output_tokens
        else:
            output_tokens = self.count_tokens(response, model)
        
        total_tokens = input_tokens + output_tokens
        
        # Calculate costs
        pricing = self.token_costs.get(model, self.token_costs['gpt-4o-mini'])
        input_cost = (input_tokens / 1000) * pricing['input']
        output_cost = (output_tokens / 1000) * pricing['output']
        total_cost = input_cost + output_cost
        
        return {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': total_tokens,
            'input_cost': round(input_cost, 6),
            'output_cost': round(output_cost, 6),
            'total_cost': round(total_cost, 6),
            'model': model,
            'pricing': pricing
        }
    
    def get_model_pricing(self, model: str) -> Dict[str, float]:
        """Get pricing for a specific model"""
        return self.token_costs.get(model, self.token_costs['gpt-4o-mini'])

# Global token calculator instance
token_calculator = TokenCalculator()