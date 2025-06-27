import json
from agents.base_agent import BaseDockerAgent
from Tool.agent_tool import doc_qa_tool, run_command_tool
from config import OPENAI_API_KEY

# Note: This is a simulation of Vercel AI SDK patterns in Python
# In a real implementation, you'd use the JavaScript/TypeScript SDK

class VercelDockerAgent(BaseDockerAgent):
    def setup(self):
        # Simulate Vercel AI SDK setup
        self.api_key = OPENAI_API_KEY
        self.model = "gpt-4o-mini"
        
        # Define tools in Vercel AI SDK style
        self.tools = {
            "docker_docs_search": {
                "description": "Search Docker documentation for relevant information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"]
                }
            },
            "execute_docker_command": {
                "description": "Execute a Docker command safely",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Docker command to execute"}
                    },
                    "required": ["command"]
                }
            }
        }
    
    def run(self, query: str) -> str:
        try:
            # Simulate Vercel AI SDK generateText with tools
            system_message = """You are a Docker expert assistant. You help users with Docker commands.

Available tools:
- docker_docs_search: Search Docker documentation
- execute_docker_command: Execute Docker commands safely

Process:
1. Search documentation for correct syntax
2. Generate appropriate Docker command
3. Execute safe commands automatically
4. Explain what each command does"""

            # Step 1: Search documentation
            docs_result = doc_qa_tool(query, None)
            
            # Step 2: Generate response with command
            response = f"""Based on the Docker documentation, here's how to handle your request: "{query}"

Documentation Context: {docs_result}

"""
            
            # Step 3: Execute safe commands
            if any(safe_cmd in query.lower() for safe_cmd in ['list', 'show', 'ps', 'images', 'version', 'info']):
                if 'container' in query.lower() or 'ps' in query.lower():
                    cmd = "docker ps"
                elif 'image' in query.lower():
                    cmd = "docker images"
                elif 'version' in query.lower():
                    cmd = "docker version"
                elif 'info' in query.lower():
                    cmd = "docker info"
                else:
                    cmd = "docker ps"  # default safe command
                
                response += f"Executing command: `{cmd}`\n\n"
                execution_result = run_command_tool(cmd)
                response += f"Result:\n{execution_result}"
            else:
                response += "For safety, please confirm if you want to execute any potentially destructive commands."
            
            return response
            
        except Exception as e:
            return f"Vercel AI Error: {str(e)}"
