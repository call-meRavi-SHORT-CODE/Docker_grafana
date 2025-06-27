import argparse
import sys
import os
from dotenv import load_dotenv

load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Run Docker Agent with different frameworks")
    parser.add_argument("--framework", required=True, 
                       choices=["langgraph", "crewai", "autogen", "llamaindex", "openai", 
                               "dspy", "mem0", "vercel", "litellm", "bedrock", "neo4j", 
                               "guardrails", "agno", "cleanlab", "graphlit"],
                       help="Choose the AI framework to use")
    parser.add_argument("--query", required=True, help="Query to ask the Docker agent")
    
    args = parser.parse_args()
    
    try:
        if args.framework == "langgraph":
            from agents.langgraph_agent import LangGraphDockerAgent
            agent = LangGraphDockerAgent()
        elif args.framework == "crewai":
            from agents.crewai_agent import CrewAIDockerAgent
            agent = CrewAIDockerAgent()
        elif args.framework == "autogen":
            from agents.autogen_agent import AutoGenDockerAgent
            agent = AutoGenDockerAgent()
        elif args.framework == "llamaindex":
            from agents.llamaindex_agent import LlamaIndexDockerAgent
            agent = LlamaIndexDockerAgent()
        elif args.framework == "openai":
            from agents.openai_agent import OpenAIDockerAgent
            agent = OpenAIDockerAgent()
        elif args.framework == "dspy":
            from agents.dspy_agent import DSPyDockerAgent
            agent = DSPyDockerAgent()
        elif args.framework == "mem0":
            from agents.mem0_agent import Mem0DockerAgent
            agent = Mem0DockerAgent()
        elif args.framework == "vercel":
            from agents.vercel_agent import VercelDockerAgent
            agent = VercelDockerAgent()
        elif args.framework == "litellm":
            from agents.litellm_agent import LiteLLMDockerAgent
            agent = LiteLLMDockerAgent()
        elif args.framework == "bedrock":
            from agents.bedrock_agent import BedrockDockerAgent
            agent = BedrockDockerAgent()
        elif args.framework == "neo4j":
            from agents.neo4j_agent import Neo4jDockerAgent
            agent = Neo4jDockerAgent()
        elif args.framework == "guardrails":
            from agents.guardrails_agent import GuardrailsDockerAgent
            agent = GuardrailsDockerAgent()
        elif args.framework == "agno":
            from agents.agno_agent import AgnoDockerAgent
            agent = AgnoDockerAgent()
        elif args.framework == "cleanlab":
            from agents.cleanlab_agent import CleanlabDockerAgent
            agent = CleanlabDockerAgent()
        elif args.framework == "graphlit":
            from agents.graphlit_agent import GraphlitDockerAgent
            agent = GraphlitDockerAgent()
        
        print(f"\n🚀 Running {args.framework.upper()} Docker Agent")
        print(f"Query: {args.query}")
        print("-" * 50)
        
        result = agent.run(args.query)
        print(f"\n✅ Result: {result}")
        
    except Exception as e:
        print(f"❌ Error running {args.framework} agent: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
