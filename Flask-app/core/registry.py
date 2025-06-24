from typing import Dict, Type, Any, List
from abc import ABC, abstractmethod
import importlib
import inspect

class FrameworkAdapter(ABC):
    """Abstract base class for framework adapters"""
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the framework name"""
        pass
    
    @abstractmethod
    def get_supported_models(self) -> List[str]:
        """Return list of supported models for this framework"""
        pass
    
    @abstractmethod
    def create_agent(self, config: Dict[str, Any]) -> Any:
        """Create and return an agent instance"""
        pass
    
    @abstractmethod
    def execute_query(self, agent: Any, query: str) -> Dict[str, Any]:
        """Execute a query using the agent"""
        pass

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the provider name"""
        pass
    
    @abstractmethod
    def get_supported_models(self) -> List[str]:
        """Return list of supported models"""
        pass
    
    @abstractmethod
    def create_llm(self, model_name: str, **kwargs) -> Any:
        """Create and return an LLM instance"""
        pass

class VectorStoreProvider(ABC):
    """Abstract base class for vector store providers"""
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the provider name"""
        pass
    
    @abstractmethod
    def create_store(self, **kwargs) -> Any:
        """Create and return a vector store instance"""
        pass

class ComponentRegistry:
    """Registry for managing framework adapters, LLM providers, and vector stores"""
    
    def __init__(self):
        self._frameworks: Dict[str, Type[FrameworkAdapter]] = {}
        self._llm_providers: Dict[str, Type[LLMProvider]] = {}
        self._vector_stores: Dict[str, Type[VectorStoreProvider]] = {}
    
    def register_framework(self, adapter_class: Type[FrameworkAdapter]):
        """Register a framework adapter"""
        if not issubclass(adapter_class, FrameworkAdapter):
            raise ValueError("Adapter must inherit from FrameworkAdapter")
        
        # Create instance to get name
        instance = adapter_class()
        name = instance.get_name().lower()
        self._frameworks[name] = adapter_class
    
    def register_llm_provider(self, provider_class: Type[LLMProvider]):
        """Register an LLM provider"""
        if not issubclass(provider_class, LLMProvider):
            raise ValueError("Provider must inherit from LLMProvider")
        
        instance = provider_class()
        name = instance.get_name().lower()
        self._llm_providers[name] = provider_class
    
    def register_vector_store(self, store_class: Type[VectorStoreProvider]):
        """Register a vector store provider"""
        if not issubclass(store_class, VectorStoreProvider):
            raise ValueError("Store must inherit from VectorStoreProvider")
        
        instance = store_class()
        name = instance.get_name().lower()
        self._vector_stores[name] = store_class
    
    def get_framework(self, name: str) -> FrameworkAdapter:
        """Get a framework adapter instance"""
        name = name.lower()
        if name not in self._frameworks:
            raise ValueError(f"Framework '{name}' not registered")
        return self._frameworks[name]()
    
    def get_llm_provider(self, name: str) -> LLMProvider:
        """Get an LLM provider instance"""
        name = name.lower()
        if name not in self._llm_providers:
            raise ValueError(f"LLM provider '{name}' not registered")
        return self._llm_providers[name]()
    
    def get_vector_store(self, name: str) -> VectorStoreProvider:
        """Get a vector store provider instance"""
        name = name.lower()
        if name not in self._vector_stores:
            raise ValueError(f"Vector store '{name}' not registered")
        return self._vector_stores[name]()
    
    def get_available_frameworks(self) -> List[str]:
        """Get list of available frameworks"""
        return list(self._frameworks.keys())
    
    def get_available_llm_providers(self) -> List[str]:
        """Get list of available LLM providers"""
        return list(self._llm_providers.keys())
    
    def get_available_vector_stores(self) -> List[str]:
        """Get list of available vector stores"""
        return list(self._vector_stores.keys())
    
    def auto_discover_components(self, package_path: str):
        """Auto-discover and register components from a package"""
        try:
            module = importlib.import_module(package_path)
            
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, FrameworkAdapter) and obj != FrameworkAdapter:
                    self.register_framework(obj)
                elif issubclass(obj, LLMProvider) and obj != LLMProvider:
                    self.register_llm_provider(obj)
                elif issubclass(obj, VectorStoreProvider) and obj != VectorStoreProvider:
                    self.register_vector_store(obj)
        except ImportError as e:
            print(f"Failed to import {package_path}: {e}")

# Global registry instance
registry = ComponentRegistry()