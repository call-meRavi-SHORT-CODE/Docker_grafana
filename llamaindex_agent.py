from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool
from agents.base_agent import BaseDockerAgent
from Tool.agent_tool import doc_qa_tool, run_command_tool
from config import OPENAI_API_KEY

class LlamaIndexDockerAgent(BaseDockerAgent):
    def setup(self):
        self.llm = OpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini")
        
        # Create tools
        def docker_docs_search(query: str) -> str:
            """Search Docker documentation for relevant information"""
            return doc_qa_tool(query, self.llm)
        
        def execute_docker_command(command: str) -> str:
            """Execute a Docker command and return the output"""
            return run_command_tool(command)
        
        # Wrap functions as LlamaIndex tools
        self.tools = [
            FunctionTool.from_defaults(fn=docker_docs_search),
            FunctionTool.from_defaults(fn=execute_docker_command),
        ]
        
        # Create ReAct agent
        self.agent = ReActAgent.from_tools(
            self.tools,
            llm=self.llm,
            verbose=True,
            system_prompt="""You are a Docker expert assistant. You help users with Docker commands and operations.

You have access to two tools:
1. docker_docs_search: Query Docker documentation to find relevant information
2. execute_docker_command: Execute Docker commands safely

When a user asks about Docker:
1. First use docker_docs_search to understand the correct syntax and options
2. Generate the appropriate Docker command
3. If the command is safe (like listing containers, images, etc.), execute it using execute_docker_command
4. If the command could be destructive, ask for confirmation first

Always explain what each command does before executing it."""
        )
    
    def run(self, query: str) -> str:
        try:
            response = self.agent.chat(query)
            return str(response)
        except Exception as e:
            return f"LlamaIndex Error: {str(e)}"
