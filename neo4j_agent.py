from neo4j import GraphDatabase
from openai import OpenAI
from agents.base_agent import BaseDockerAgent
from Tool.agent_tool import doc_qa_tool, run_command_tool
from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, OPENAI_API_KEY

class Neo4jDockerAgent(BaseDockerAgent):
    def setup(self):
        # Initialize Neo4j driver
        try:
            self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
            self._create_docker_knowledge_graph()
        except Exception as e:
            print(f"Neo4j connection failed: {e}")
            self.driver = None
        
        self.client = OpenAI(api_key=OPENAI_API_KEY)
    
    def _create_docker_knowledge_graph(self):
        """Create a basic Docker knowledge graph"""
        with self.driver.session() as session:
            # Create Docker command nodes and relationships
            session.run("""
                MERGE (docker:Tool {name: 'Docker'})
                MERGE (ps:Command {name: 'ps', description: 'List containers', safe: true})
                MERGE (images:Command {name: 'images', description: 'List images', safe: true})
                MERGE (run:Command {name: 'run', description: 'Create and start container', safe: false})
                MERGE (stop:Command {name: 'stop', description: 'Stop container', safe: false})
                MERGE (rm:Command {name: 'rm', description: 'Remove container', safe: false})
                
                MERGE (docker)-[:HAS_COMMAND]->(ps)
                MERGE (docker)-[:HAS_COMMAND]->(images)
                MERGE (docker)-[:HAS_COMMAND]->(run)
                MERGE (docker)-[:HAS_COMMAND]->(stop)
                MERGE (docker)-[:HAS_COMMAND]->(rm)
                
                MERGE (list:Intent {name: 'list'})
                MERGE (create:Intent {name: 'create'})
                MERGE (manage:Intent {name: 'manage'})
                
                MERGE (list)-[:USES_COMMAND]->(ps)
                MERGE (list)-[:USES_COMMAND]->(images)
                MERGE (create)-[:USES_COMMAND]->(run)
                MERGE (manage)-[:USES_COMMAND]->(stop)
                MERGE (manage)-[:USES_COMMAND]->(rm)
            """)
    
    def _query_graph(self, intent: str):
        """Query the knowledge graph for relevant commands"""
        if not self.driver:
            return []
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH (i:Intent)-[:USES_COMMAND]->(c:Command)
                WHERE i.name CONTAINS $intent OR c.description CONTAINS $intent
                RETURN c.name as command, c.description as description, c.safe as safe
            """, intent=intent.lower())
            
            return [{"command": record["command"], 
                    "description": record["description"], 
                    "safe": record["safe"]} for record in result]
    
    def run(self, query: str) -> str:
        try:
            # Query knowledge graph
            graph_results = self._query_graph(query)
            graph_context = "\n".join([f"- {r['command']}: {r['description']} (safe: {r['safe']})" 
                                     for r in graph_results])
            
            # Get documentation context
            docs = doc_qa_tool(query, None)
            
            # Combine contexts
            context = f"""
Knowledge Graph Results:
{graph_context}

Documentation Context:
{docs}
"""
            
            messages = [
                {
                    "role": "system",
                    "content": """You are a Docker expert with access to a knowledge graph and documentation.
                    Use both sources to provide accurate Docker command suggestions.
                    Execute safe commands automatically, warn about unsafe ones."""
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
            
            # Execute safe commands based on graph knowledge
            safe_commands = [r for r in graph_results if r['safe']]
            if safe_commands and any(cmd['command'] in answer.lower() for cmd in safe_commands):
                for cmd_info in safe_commands:
                    if cmd_info['command'] in answer.lower():
                        cmd = f"docker {cmd_info['command']}"
                        execution_result = run_command_tool(cmd)
                        answer += f"\n\nExecution Result:\n{execution_result}"
                        break
            
            return answer
            
        except Exception as e:
            return f"Neo4j Graph RAG Error: {str(e)}"
    
    def __del__(self):
        if hasattr(self, 'driver') and self.driver:
            self.driver.close()
