import autogen
from agents.base_agent import BaseDockerAgent
from Tool.agent_tool import doc_qa_tool, run_command_tool
from config import OPENAI_API_KEY

class AutoGenDockerAgent(BaseDockerAgent):
    def setup(self):
        config_list = [
            {
                "model": "gpt-4o-mini",
                "api_key": OPENAI_API_KEY,
            }
        ]
        
        llm_config = {
            "config_list": config_list,
            "temperature": 0.1,
        }
        
        # Create user proxy agent
        self.user_proxy = autogen.UserProxyAgent(
            name="user_proxy",
            system_message="A human admin who can execute Docker commands.",
            code_execution_config={"last_n_messages": 2, "work_dir": "autogen_workspace"},
            human_input_mode="NEVER",
            function_map={
                "docker_docs_search": lambda query: doc_qa_tool(query, None),
                "execute_docker_command": run_command_tool,
            }
        )
        
        # Create Docker expert agent
        self.docker_expert = autogen.AssistantAgent(
            name="docker_expert",
            system_message="""You are a Docker expert assistant. You help users with Docker commands.
            
            You have access to these functions:
            - docker_docs_search: Search Docker documentation
            - execute_docker_command: Execute Docker commands
            
            When helping users:
            1. First search documentation to understand the correct syntax
            2. Generate the appropriate Docker command
            3. Execute safe commands automatically
            4. For destructive commands, explain the risks first
            """,
            llm_config=llm_config,
        )
        
        # Register functions
        self.user_proxy.register_function(
            function_map={
                "docker_docs_search": lambda query: doc_qa_tool(query, None),
                "execute_docker_command": run_command_tool,
            }
        )
    
    def run(self, query: str) -> str:
        try:
            # Start conversation
            self.user_proxy.initiate_chat(
                self.docker_expert,
                message=f"Help me with this Docker task: {query}",
                max_turns=5
            )
            
            # Get the last message from the conversation
            messages = self.user_proxy.chat_messages[self.docker_expert]
            if messages:
                return messages[-1]["content"]
            else:
                return "No response generated"
                
        except Exception as e:
            return f"AutoGen Error: {str(e)}"
