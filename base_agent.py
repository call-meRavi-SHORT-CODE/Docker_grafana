from abc import ABC, abstractmethod
from typing import Any

class BaseDockerAgent(ABC):
    """Base class for all Docker agents"""
    
    def __init__(self):
        self.setup()
    
    @abstractmethod
    def setup(self):
        """Initialize the agent"""
        pass
    
    @abstractmethod
    def run(self, query: str) -> str:
        """Run the agent with a query"""
        pass
    
    def get_framework_name(self) -> str:
        """Get the framework name"""
        return self.__class__.__name__.replace("DockerAgent", "")
