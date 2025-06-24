import os
from pydantic_settings import BaseSettings
from typing import Optional, List

class Settings(BaseSettings):
    # Application Settings
    APP_NAME: str = "Docker Agent"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 5000
    
    # API Settings
    RAG_API_URL: str = "http://localhost:8000"
    
    # Database Settings
    DATABASE_URL: Optional[str] = None
    REDIS_URL: str = "redis://localhost:6379"
    
    # Tracing and Monitoring
    LANGTRACE_API_KEY: Optional[str] = None
    GRAFANA_CLOUD_URL: Optional[str] = None
    GRAFANA_CLOUD_API_KEY: Optional[str] = None
    PROMETHEUS_PORT: int = 8001
    
    # LLM Provider Keys
    OPENAI_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    
    # Supported Configurations
    FRAMEWORKS: List[str] = ["langgraph", "autogen", "llamaindex", "dspy"]
    MODELS: List[str] = [
        'gpt-4o', 'gpt-4o-mini', "gpt-4.1", "gpt-4.1-mini", 
        "gpt-3.5-turbo", 'llama3-8b-8192', 'gemma2-9b-it',
        "llama-3.3-70b-versatile", "gemini-2.0-flash"
    ]
    VECTORSTORES: List[str] = ['Faiss', 'Chroma', 'Annoy']
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()