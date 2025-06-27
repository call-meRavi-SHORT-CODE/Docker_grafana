import guardrails as gd
from guardrails.validators import ValidLength, ToxicLanguage
from openai import OpenAI
from agents.base_agent import BaseDockerAgent
from Tool.agent_tool import doc_qa_tool, run_command_tool
from config import OPENAI_API_KEY

class GuardrailsDockerAgent(BaseDockerAgent):
    def setup(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Define guardrails spec
        self.rail_spec = """
<rail version="0.1">
<output>
    <string name="response" 
            description="Docker expert response"
            validators="valid-length: 10 2000; toxic-language" />
    <string name="command" 
            description="Docker command if applicable"
            validators="valid-length: 5 100" />
    <boolean name="safe_to_execute" 
             description="Whether the command is safe to execute automatically" />
</output>

<prompt>
You are a Docker expert assistant. Provide helpful responses about Docker commands.

Given this context: {{context}}
User query: {{query}}

Respond with:
1. A helpful explanation
2. The Docker command if applicable
3. Whether it's safe to execute automatically

@xml_prefix_prompt
</prompt>
</rail>
"""
        
        # Create guard
        self.guard = gd.Guard.from_rail_string(self.rail_spec)
    
    def run(self, query: str) -> str:
        try:
            # Get documentation context
            docs = doc_qa_tool(query, None)
            
            # Use guardrails to generate validated response
            raw_llm_output, validated_output = self.guard(
                self.client.chat.completions.create,
                model="gpt-4o-mini",
                messages=[{
                    "role": "user", 
                    "content": f"Context: {docs}\nQuery: {query}"
                }],
                prompt_params={"context": docs, "query": query},
                max_tokens=1000
            )
            
            if validated_output:
                response = validated_output["response"]
                command = validated_output.get("command", "")
                safe_to_execute = validated_output.get("safe_to_execute", False)
                
                # Execute if safe
                if command and safe_to_execute and command.startswith("docker"):
                    execution_result = run_command_tool(command)
                    response += f"\n\nExecution Result:\n{execution_result}"
                
                return response
            else:
                return "Response failed validation checks"
                
        except Exception as e:
            return f"Guardrails Error: {str(e)}"
