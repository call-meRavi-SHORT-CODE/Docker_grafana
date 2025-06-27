# Note: Agno is a hypothetical framework - implementing a pattern-based approach
from openai import OpenAI
from agents.base_agent import BaseDockerAgent
from Tool.agent_tool import doc_qa_tool, run_command_tool
from config import OPENAI_API_KEY
import re

class AgnoDockerAgent(BaseDockerAgent):
    def setup(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Define Agno-style patterns
        self.patterns = {
            "list_containers": {
                "triggers": ["list containers", "show containers", "running containers", "ps"],
                "command": "docker ps",
                "safe": True
            },
            "list_images": {
                "triggers": ["list images", "show images", "docker images"],
                "command": "docker images",
                "safe": True
            },
            "docker_version": {
                "triggers": ["version", "docker version"],
                "command": "docker version",
                "safe": True
            },
            "docker_info": {
                "triggers": ["info", "system info", "docker info"],
                "command": "docker info",
                "safe": True
            }
        }
    
    def _match_pattern(self, query: str):
        """Match query against predefined patterns"""
        query_lower = query.lower()
        for pattern_name, pattern_data in self.patterns.items():
            for trigger in pattern_data["triggers"]:
                if trigger in query_lower:
                    return pattern_data
        return None
    
    def run(self, query: str) -> str:
        try:
            # Pattern matching approach
            matched_pattern = self._match_pattern(query)
            
            if matched_pattern:
                # Direct pattern execution
                command = matched_pattern["command"]
                response = f"Agno Pattern Matched: {command}\n\n"
                
                if matched_pattern["safe"]:
                    execution_result = run_command_tool(command)
                    response += f"Execution Result:\n{execution_result}"
                else:
                    response += "Command requires manual confirmation for safety."
                
                return response
            else:
                # Fallback to documentation search + LLM
                docs = doc_qa_tool(query, None)
                
                messages = [
                    {
                        "role": "system",
                        "content": """You are a Docker expert using the Agno framework approach.
                        Provide clear, pattern-based responses for Docker commands.
                        Focus on practical, executable solutions."""
                    },
                    {
                        "role": "user",
                        "content": f"Documentation: {docs}\n\nQuery: {query}"
                    }
                ]
                
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages
                )
                
                answer = response.choices[0].message.content
                
                # Check for safe commands in response
                if "docker " in answer.lower():
                    safe_commands = ["docker ps", "docker images", "docker version", "docker info"]
                    for safe_cmd in safe_commands:
                        if safe_cmd in answer.lower():
                            execution_result = run_command_tool(safe_cmd)
                            answer += f"\n\nExecution Result:\n{execution_result}"
                            break
                
                return answer
                
        except Exception as e:
            return f"Agno Error: {str(e)}"
