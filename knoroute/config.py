"""
Centralized configuration management for the Agentic Knowledge Routing System.
Uses Pydantic Settings for type-safe configuration with environment variable support.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # OpenAI Configuration
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    
    # LLM Model Configuration
    llm_model: str = Field(default="gpt-4-turbo-preview", env="LLM_MODEL")
    embedding_model: str = Field(default="text-embedding-3-small", env="EMBEDDING_MODEL")
    
    # Vector Store Configuration
    vector_store_type: Literal["chroma", "faiss"] = Field(default="chroma", env="VECTOR_STORE_TYPE")
    vector_store_path: str = Field(default="./data/vectorstores", env="VECTOR_STORE_PATH")
    
    # Retrieval Configuration
    retrieval_top_k: int = Field(default=5, env="RETRIEVAL_TOP_K")
    max_retry_attempts: int = Field(default=3, env="MAX_RETRY_ATTEMPTS")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_reload: bool = Field(default=True, env="API_RELOAD")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()
