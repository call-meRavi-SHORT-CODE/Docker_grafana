"""
Framework Manager - Centralized management of all Docker agent frameworks
"""
import importlib
import logging
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
import time

logger = logging.getLogger(__name__)

class FrameworkManager:
    """Manages all Docker agent frameworks"""
    
    def __init__(self):
        self.frameworks = {}
        self._initialize_frameworks()
    
    def _initialize_frameworks(self):
        """Initialize all available frameworks"""
        framework_configs = {
            'langgraph': {
                'module': 'agents.langgraph_agent',
                'class': 'LangGraphDockerAgent',
                'enabled': True
            },
            'crewai': {
                'module': 'agents.crewai_agent', 
                'class': 'CrewAIDockerAgent',
                'enabled': True
            },
            'autogen': {
                'module': 'agents.autogen_agent',
                'class': 'AutoGenDockerAgent', 
                'enabled': True
            },
            'llamaindex': {
                'module': 'agents.llamaindex_agent',
                'class': 'LlamaIndexDockerAgent',
                'enabled': True
            },
            'openai': {
                'module': 'agents.openai_agent',
                'class': 'OpenAIDockerAgent',
                'enabled': True
            },
            'dspy': {
                'module': 'agents.dspy_agent',
                'class': 'DSPyDockerAgent',
                'enabled': True
            },
            'mem0': {
                'module': 'agents.mem0_agent',
                'class': 'Mem0DockerAgent',
                'enabled': True
            },
            'vercel': {
                'module': 'agents.vercel_agent',
                'class': 'VercelDockerAgent',
                'enabled': True
            },
            'litellm': {
                'module': 'agents.litellm_agent',
                'class': 'LiteLLMDockerAgent',
                'enabled': True
            },
            'bedrock': {
                'module': 'agents.bedrock_agent',
                'class': 'BedrockDockerAgent',
                'enabled': True
            },
            'neo4j': {
                'module': 'agents.neo4j_agent',
                'class': 'Neo4jDockerAgent',
                'enabled': True
            },
            'guardrails': {
                'module': 'agents.guardrails_agent',
                'class': 'GuardrailsDockerAgent',
                'enabled': True
            },
            'agno': {
                'module': 'agents.agno_agent',
                'class': 'AgnoDockerAgent',
                'enabled': True
            },
            'cleanlab': {
                'module': 'agents.cleanlab_agent',
                'class': 'CleanlabDockerAgent',
                'enabled': True
            },
            'graphlit': {
                'module': 'agents.graphlit_agent',
                'class': 'GraphlitDockerAgent',
                'enabled': True
            }
        }
        
        for name, config in framework_configs.items():
            if config['enabled']:
                try:
                    self._load_framework(name, config)
                    logger.info(f"Successfully loaded framework: {name}")
                except Exception as e:
                    logger.error(f"Failed to load framework {name}: {e}")
                    # Continue loading other frameworks
    
    def _load_framework(self, name: str, config: Dict[str, Any]):
        """Load a specific framework"""
        try:
            module = importlib.import_module(config['module'])
            agent_class = getattr(module, config['class'])
            
            # Create instance
            agent_instance = agent_class()
            
            self.frameworks[name] = {
                'instance': agent_instance,
                'class': agent_class,
                'config': config,
                'status': 'loaded'
            }
        except Exception as e:
            logger.error(f"Error loading {name}: {e}")
            self.frameworks[name] = {
                'instance': None,
                'class': None,
                'config': config,
                'status': 'failed',
                'error': str(e)
            }
    
    def get_framework(self, name: str) -> Optional[Any]:
        """Get framework instance by name"""
        framework = self.frameworks.get(name.lower())
        if framework and framework['status'] == 'loaded':
            return framework['instance']
        return None
    
    def execute_query(self, framework_name: str, query: str) -> Dict[str, Any]:
        """Execute query using specified framework"""
        start_time = time.time()
        
        try:
            framework = self.get_framework(framework_name)
            if not framework:
                return {
                    'answer': f"Framework '{framework_name}' not available",
                    'status': 'error',
                    'duration': time.time() - start_time,
                    'framework': framework_name
                }
            
            # Execute the query
            result = framework.run(query)
            duration = time.time() - start_time
            
            return {
                'answer': result,
                'status': 'success',
                'duration': duration,
                'framework': framework_name
            }
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Error executing query with {framework_name}: {e}")
            
            return {
                'answer': f"Error with {framework_name}: {str(e)}",
                'status': 'error', 
                'duration': duration,
                'framework': framework_name,
                'error': str(e)
            }
    
    def get_available_frameworks(self) -> Dict[str, Any]:
        """Get list of available frameworks with their status"""
        return {
            name: {
                'status': info['status'],
                'error': info.get('error')
            }
            for name, info in self.frameworks.items()
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Check health of all frameworks"""
        health_status = {}
        
        for name, framework in self.frameworks.items():
            try:
                if framework['status'] == 'loaded' and framework['instance']:
                    # Try a simple test query
                    test_result = framework['instance'].run("docker version")
                    health_status[name] = {
                        'status': 'healthy',
                        'test_passed': True
                    }
                else:
                    health_status[name] = {
                        'status': 'unhealthy',
                        'test_passed': False,
                        'error': framework.get('error', 'Not loaded')
                    }
            except Exception as e:
                health_status[name] = {
                    'status': 'unhealthy',
                    'test_passed': False,
                    'error': str(e)
                }
        
        return health_status

# Global framework manager instance
framework_manager = FrameworkManager()