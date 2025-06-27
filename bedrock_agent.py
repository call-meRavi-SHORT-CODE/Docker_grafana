import boto3
import json
from agents.base_agent import BaseDockerAgent
from Tool.agent_tool import doc_qa_tool, run_command_tool
from config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY

class BedrockDockerAgent(BaseDockerAgent):
    def setup(self):
        # Initialize Bedrock client
        self.bedrock = boto3.client(
            'bedrock-runtime',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name='us-east-1'
        )
        self.model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
    
    def run(self, query: str) -> str:
        try:
            # Get documentation context
            docs = doc_qa_tool(query, None)
            
            # Prepare prompt for Claude
            prompt = f"""You are a Docker expert assistant. Help the user with their Docker question.

Documentation context: {docs}

User question: {query}

Provide a helpful response. If you need to suggest a Docker command, make sure it's safe and explain what it does.
For listing commands (ps, images, version, info), you can suggest execution.
For potentially destructive commands, explain the risks.

Response:"""

            # Call Bedrock
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            })
            
            response = self.bedrock.invoke_model(
                body=body,
                modelId=self.model_id,
                accept='application/json',
                contentType='application/json'
            )
            
            response_body = json.loads(response.get('body').read())
            answer = response_body['content'][0]['text']
            
            # Execute safe commands if suggested
            if "docker " in answer.lower():
                lines = answer.split('\n')
                for line in lines:
                    if line.strip().startswith('docker ') and any(safe_cmd in line.lower() for safe_cmd in ['ps', 'images', 'version', 'info']):
                        cmd = line.strip()
                        if '`' in cmd:
                            cmd = cmd.split('`')[1] if '`' in cmd else cmd
                        
                        execution_result = run_command_tool(cmd)
                        answer += f"\n\nExecution Result:\n{execution_result}"
                        break
            
            return answer
            
        except Exception as e:
            return f"Bedrock Error: {str(e)}"
