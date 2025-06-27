from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from agents.base_agent import BaseDockerAgent
from Tool.agent_tool import doc_qa_tool, run_command_tool
from config import OPENAI_API_KEY
from Prompt.prompts import system_prompt

class LangGraphDockerAgent(BaseDockerAgent):
    def setup(self):
        self.llm = init_chat_model("gpt-4o-mini", model_provider="openai", api_key=OPENAI_API_KEY)
        
        @tool
        def docker_docs_search(query: str) -> str:
            """Search Docker documentation for relevant information"""
            return doc_qa_tool(query, self.llm)
        
        @tool
        def execute_docker_command(command: str) -> str:
            """Execute a Docker command safely"""
            return run_command_tool(command)
        
        self.tools = [docker_docs_search, execute_docker_command]
        self.agent = create_react_agent(model=self.llm, tools=self.tools, prompt=system_prompt)
    
    def run(self, query: str) -> str:
        try:
            inputs = {"messages": [("user", query)]}
            result = ""
            
            for step in self.agent.stream(inputs, stream_mode="values"):
                msg = step['messages'][-1]
                if hasattr(msg, 'content'):
                    result = msg.content
            
            return result or "No response generated"
            
        except Exception as e:
            return f"LangGraph Error: {str(e)}"
