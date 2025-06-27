# Note: Cleanlab is primarily for data quality - simulating confidence scoring approach
from openai import OpenAI
import numpy as np
from agents.base_agent import BaseDockerAgent
from Tool.agent_tool import doc_qa_tool, run_command_tool
from config import OPENAI_API_KEY

class CleanlabDockerAgent(BaseDockerAgent):
    def setup(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Simulate confidence thresholds
        self.confidence_threshold = 0.8
        self.safe_commands = ["docker ps", "docker images", "docker version", "docker info"]
    
    def _calculate_confidence(self, query: str, response: str) -> float:
        """Simulate confidence calculation based on response quality"""
        confidence_factors = []
        
        # Factor 1: Response length (reasonable responses are neither too short nor too long)
        length_score = min(len(response) / 500, 1.0) if len(response) > 50 else 0.3
        confidence_factors.append(length_score)
        
        # Factor 2: Contains Docker-specific terms
        docker_terms = ["docker", "container", "image", "dockerfile", "compose"]
        docker_score = sum(1 for term in docker_terms if term.lower() in response.lower()) / len(docker_terms)
        confidence_factors.append(docker_score)
        
        # Factor 3: Query-response relevance (simple keyword matching)
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        relevance_score = len(query_words.intersection(response_words)) / max(len(query_words), 1)
        confidence_factors.append(relevance_score)
        
        # Factor 4: Command safety
        safety_score = 1.0 if any(safe_cmd in response.lower() for safe_cmd in self.safe_commands) else 0.7
        confidence_factors.append(safety_score)
        
        # Calculate weighted average
        weights = [0.2, 0.3, 0.3, 0.2]
        confidence = sum(w * f for w, f in zip(weights, confidence_factors))
        
        return min(confidence, 1.0)
    
    def run(self, query: str) -> str:
        try:
            # Get documentation context
            docs = doc_qa_tool(query, None)
            
            # Generate multiple responses for confidence estimation
            responses = []
            confidences = []
            
            for i in range(3):  # Generate 3 responses
                messages = [
                    {
                        "role": "system",
                        "content": f"""You are a Docker expert assistant (iteration {i+1}).
                        Use the documentation context to provide accurate Docker guidance.
                        Be specific and practical in your responses."""
                    },
                    {
                        "role": "user",
                        "content": f"Documentation: {docs}\n\nQuery: {query}"
                    }
                ]
                
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.1 + (i * 0.1)  # Slight variation
                )
                
                answer = response.choices[0].message.content
                confidence = self._calculate_confidence(query, answer)
                
                responses.append(answer)
                confidences.append(confidence)
            
            # Select highest confidence response
            best_idx = np.argmax(confidences)
            best_response = responses[best_idx]
            best_confidence = confidences[best_idx]
            
            # Add confidence information
            result = f"Cleanlab Confidence Score: {best_confidence:.2f}\n\n{best_response}"
            
            # Execute commands only if high confidence
            if best_confidence >= self.confidence_threshold:
                if "docker " in best_response.lower():
                    for safe_cmd in self.safe_commands:
                        if safe_cmd in best_response.lower():
                            execution_result = run_command_tool(safe_cmd)
                            result += f"\n\nHigh Confidence Execution:\n{execution_result}"
                            break
            else:
                result += f"\n\n⚠️ Low confidence ({best_confidence:.2f}). Manual verification recommended."
            
            return result
            
        except Exception as e:
            return f"Cleanlab Error: {str(e)}"
