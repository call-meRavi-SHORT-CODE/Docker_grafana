import dspy
from agents.base_agent import BaseDockerAgent
from Tool.agent_tool import doc_qa_tool, run_command_tool
from config import OPENAI_API_KEY

class DockerAssistant(dspy.Signature):
    """Docker expert that helps with commands and operations"""
    query = dspy.InputField(desc="User's Docker-related question or request")
    documentation = dspy.InputField(desc="Relevant Docker documentation context")
    answer = dspy.OutputField(desc="Helpful response with Docker command and explanation")

class DSPyDockerAgent(BaseDockerAgent):
    def setup(self):
        # Configure DSPy with OpenAI
        lm = dspy.OpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
        dspy.settings.configure(lm=lm)
        
        # Create predictor
        self.predictor = dspy.Predict(DockerAssistant)
    
    def run(self, query: str) -> str:
        try:
            # First, get documentation context
            docs = doc_qa_tool(query, None)
            
            # Generate response using DSPy
            response = self.predictor(query=query, documentation=docs)
            
            # Check if response contains a command to execute
            answer = response.answer
            
            # Simple command detection (look for docker commands)
            if "docker " in answer.lower() and ("ps" in answer.lower() or "images" in answer.lower() or "version" in answer.lower()):
                # Extract and execute safe commands
                lines = answer.split('\n')
                for line in lines:
                    if line.strip().startswith('docker ') and any(safe_cmd in line.lower() for safe_cmd in ['ps', 'images', 'version', 'info']):
                        cmd = line.strip()
                        if cmd.startswith('`') and cmd.endswith('`'):
                            cmd = cmd[1:-1]  # Remove backticks
                        
                        execution_result = run_command_tool(cmd)
                        answer += f"\n\nExecution Result:\n{execution_result}"
                        break
            
            return answer
            
        except Exception as e:
            return f"DSPy Error: {str(e)}"
