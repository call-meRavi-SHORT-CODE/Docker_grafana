from mem0 import Memory
from openai import OpenAI
from agents.base_agent import BaseDockerAgent
from Tool.agent_tool import doc_qa_tool, run_command_tool
from config import OPENAI_API_KEY

class Mem0DockerAgent(BaseDockerAgent):
    def setup(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Initialize Mem0
        config = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": "gpt-4o-mini",
                    "api_key": OPENAI_API_KEY
                }
            }
        }
        self.memory = Memory.from_config(config)
        self.user_id = "docker_user"
    
    def run(self, query: str) -> str:
        try:
            # Get relevant memories
            memories = self.memory.search(query, user_id=self.user_id)
            memory_context = "\n".join([mem["memory"] for mem in memories]) if memories else ""
            
            # Get documentation context
            docs = doc_qa_tool(query, None)
            
            # Prepare context
            context = f"""
Previous interactions: {memory_context}
Docker documentation: {docs}
"""
            
            messages = [
                {
                    "role": "system",
                    "content": """You are a Docker expert assistant with memory of previous interactions. 
                    Use your memory and Docker documentation to provide helpful responses.
                    
                    You can execute safe Docker commands like:
                    - docker ps (list containers)
                    - docker images (list images)
                    - docker version (show version)
                    - docker info (show system info)
                    
                    For destructive commands, explain the risks first."""
                },
                {
                    "role": "user",
                    "content": f"Context: {context}\n\nQuery: {query}"
                }
            ]
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )
            
            answer = response.choices[0].message.content
            
            # Store interaction in memory
            self.memory.add(f"User asked: {query}. Response: {answer}", user_id=self.user_id)
            
            # Execute safe commands if mentioned
            if "docker " in answer.lower():
                lines = answer.split('\n')
                for line in lines:
                    if line.strip().startswith('docker ') and any(safe_cmd in line.lower() for safe_cmd in ['ps', 'images', 'version', 'info']):
                        cmd = line.strip()
                        if '`' in cmd:
                            cmd = cmd.split('`')[1] if '`' in cmd else cmd
                        
                        execution_result = run_command_tool(cmd)
                        answer += f"\n\nExecution Result:\n{execution_result}"
                        
                        # Store execution result in memory
                        self.memory.add(f"Executed command: {cmd}. Result: {execution_result}", user_id=self.user_id)
                        break
            
            return answer
            
        except Exception as e:
            return f"Mem0 Error: {str(e)}"
