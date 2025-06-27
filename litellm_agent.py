import litellm
from agents.base_agent import BaseDockerAgent
from Tool.agent_tool import doc_qa_tool, run_command_tool
from config import OPENAI_API_KEY
import json

class LiteLLMDockerAgent(BaseDockerAgent):
    def setup(self):
        # Configure LiteLLM
        litellm.api_key = OPENAI_API_KEY
        self.model = "gpt-4o-mini"
        
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "docker_docs_search",
                    "description": "Search Docker documentation",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "execute_docker_command",
                    "description": "Execute Docker command",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "Docker command"}
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
                    "content": "You are a Docker expert. Use the available tools to help users with Docker commands."
                },
                {"role": "user", "content": query}
            ]
            
            response = litellm.completion(
                model=self.model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto"
            )
            
            message = response.choices[0].message
            
            if hasattr(message, 'tool_calls') and message.tool_calls:
                # Handle tool calls
                messages.append(message)
                
                for tool_call in message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    if function_name == "docker_docs_search":
                        result = doc_qa_tool(function_args["query"], None)
                    elif function_name == "execute_docker_command":
                        result = run_command_tool(function_args["command"])
                    else:
                        result = "Unknown function"
                    
                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": result
                    })
                
                # Get final response
                final_response = litellm.completion(
                    model=self.model,
                    messages=messages
                )
                return final_response.choices[0].message.content
            else:
                return message.content
                
        except Exception as e:
            return f"LiteLLM Error: {str(e)}"
