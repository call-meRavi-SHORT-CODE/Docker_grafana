import json
from openai import OpenAI
from agents.base_agent import BaseDockerAgent
from Tool.agent_tool import doc_qa_tool, run_command_tool
from config import OPENAI_API_KEY

class OpenAIDockerAgent(BaseDockerAgent):
    def setup(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "docker_docs_search",
                    "description": "Search Docker documentation for relevant information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query for Docker documentation"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "execute_docker_command",
                    "description": "Execute a Docker command and return the output",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "The Docker command to execute"
                            }
                        },
                        "required": ["command"]
                    }
                }
            }
        ]
    
    def run(self, query: str) -> str:
        try:
            messages = [
                {
                    "role": "system",
                    "content": """You are a Docker expert assistant. You help users with Docker commands and operations.

You have access to two functions:
1. docker_docs_search: Query Docker documentation to find relevant information
2. execute_docker_command: Execute Docker commands safely

When a user asks about Docker:
1. First use docker_docs_search to understand the correct syntax and options
2. Generate the appropriate Docker command
3. If the command is safe (like listing containers, images, etc.), execute it using execute_docker_command
4. If the command could be destructive, ask for confirmation first

Always explain what each command does before executing it."""
                },
                {"role": "user", "content": query}
            ]
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=self.tools,
                tool_choice="auto"
            )
            
            message = response.choices[0].message
            
            # Handle function calls
            if message.tool_calls:
                messages.append(message)
                
                for tool_call in message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    if function_name == "docker_docs_search":
                        function_response = doc_qa_tool(function_args["query"], None)
                    elif function_name == "execute_docker_command":
                        function_response = run_command_tool(function_args["command"])
                    else:
                        function_response = "Unknown function"
                    
                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response
                    })
                
                # Get final response
                second_response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages
                )
                return second_response.choices[0].message.content
            else:
                return message.content
                
        except Exception as e:
            return f"OpenAI Error: {str(e)}"
