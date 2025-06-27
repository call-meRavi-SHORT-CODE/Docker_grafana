# Note: Graphlit is for unstructured data processing - simulating document processing approach
import json
from openai import OpenAI
from agents.base_agent import BaseDockerAgent
from Tool.agent_tool import doc_qa_tool, run_command_tool
from config import OPENAI_API_KEY

class GraphlitDockerAgent(BaseDockerAgent):
    def setup(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Simulate Graphlit document processing
        self.processed_docs = {
            "docker_commands": {
                "ps": "List running containers with details like container ID, image, command, status",
                "images": "Display all Docker images with repository, tag, image ID, created time, size",
                "version": "Show Docker version information including client and server versions",
                "info": "Display system-wide Docker information including containers, images, storage driver"
            },
            "docker_concepts": {
                "container": "A lightweight, standalone package that includes everything needed to run an application",
                "image": "A read-only template used to create containers",
                "dockerfile": "A text file containing instructions to build a Docker image",
                "compose": "A tool for defining and running multi-container Docker applications"
            }
        }
    
    def _process_query_with_graphlit(self, query: str):
        """Simulate Graphlit's document processing and knowledge extraction"""
        query_lower = query.lower()
        relevant_info = []
        
        # Extract relevant command information
        for cmd, description in self.processed_docs["docker_commands"].items():
            if cmd in query_lower or any(word in query_lower for word in description.lower().split()):
                relevant_info.append(f"Command '{cmd}': {description}")
        
        # Extract relevant concept information
        for concept, description in self.processed_docs["docker_concepts"].items():
            if concept in query_lower:
                relevant_info.append(f"Concept '{concept}': {description}")
        
        return relevant_info
    
    def run(self, query: str) -> str:
        try:
            # Process query with simulated Graphlit
            graphlit_results = self._process_query_with_graphlit(query)
            graphlit_context = "\n".join(graphlit_results) if graphlit_results else "No specific matches found"
            
            # Get additional documentation context
            docs = doc_qa_tool(query, None)
            
            # Combine processed information
            context = f"""
Graphlit Processed Information:
{graphlit_context}

Additional Documentation:
{docs}
"""
            
            messages = [
                {
                    "role": "system",
                    "content": """You are a Docker expert using Graphlit's document processing capabilities.
                    You have access to processed Docker documentation and can provide precise, 
                    contextually relevant responses based on structured knowledge extraction."""
                },
                {
                    "role": "user",
                    "content": f"Processed Context: {context}\n\nUser Query: {query}"
                }
            ]
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )
            
            answer = response.choices[0].message.content
            
            # Execute safe commands based on processed knowledge
            safe_commands = ["docker ps", "docker images", "docker version", "docker info"]
            for cmd in safe_commands:
                if cmd.split()[1] in query.lower():  # Check for command name
                    execution_result = run_command_tool(cmd)
                    answer += f"\n\nGraphlit-Processed Execution:\n{execution_result}"
                    break
            
            return f"Graphlit Processing Results:\n{answer}"
            
        except Exception as e:
            return f"Graphlit Error: {str(e)}"
