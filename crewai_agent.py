from crewai import Agent, Task, Crew
from crewai.tools import tool
from langchain_openai import ChatOpenAI
from agents.base_agent import BaseDockerAgent
from Tool.agent_tool import doc_qa_tool, run_command_tool
from config import OPENAI_API_KEY

class CrewAIDockerAgent(BaseDockerAgent):
    def setup(self):
        self.llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini")
        
        @tool
        def docker_docs_search(query: str) -> str:
            """Search Docker documentation for relevant information"""
            return doc_qa_tool(query, self.llm)
        
        @tool
        def execute_docker_command(command: str) -> str:
            """Execute a Docker command safely"""
            return run_command_tool(command)
        
        # Create agents
        self.docker_expert = Agent(
            role="Docker Expert",
            goal="Provide accurate Docker command syntax and explanations",
            backstory="You are an expert in Docker with deep knowledge of CLI commands and best practices.",
            tools=[docker_docs_search],
            llm=self.llm,
            verbose=True
        )
        
        self.command_executor = Agent(
            role="Command Executor",
            goal="Execute Docker commands safely and return results",
            backstory="You specialize in executing Docker commands and interpreting their output.",
            tools=[execute_docker_command],
            llm=self.llm,
            verbose=True
        )
        
        # Create crew
        self.crew = Crew(
            agents=[self.docker_expert, self.command_executor],
            verbose=True
        )
    
    def run(self, query: str) -> str:
        # Create tasks
        research_task = Task(
            description=f"Research the Docker command needed for: {query}. Provide the exact command syntax.",
            agent=self.docker_expert,
            expected_output="A Docker command with explanation"
        )
        
        execution_task = Task(
            description="Execute the Docker command provided by the expert and return the results.",
            agent=self.command_executor,
            expected_output="Command execution results"
        )
        
        # Update crew with current tasks
        self.crew.tasks = [research_task, execution_task]
        
        try:
            result = self.crew.kickoff()
            return str(result)
        except Exception as e:
            return f"CrewAI Error: {str(e)}"
