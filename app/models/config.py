"""
App Configuration using Pydantic Settings.
Reads from environment variables and .env file.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- LLM ----
    llm_provider: Literal["openai", "groq", "google", "openrouter"] = "groq"
    openai_api_key: str = ""
    groq_api_key: str = ""
    google_api_key: str = ""
    openrouter_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    groq_model: str = "llama-3.3-70b-versatile"
    google_model: str = "gemini-1.5-flash"
    openrouter_model: str = "meta-llama/llama-3.1-70b-instruct"

    # ---- Vector Store ----
    vector_store_type: Literal["faiss", "chroma"] = "faiss"
    vector_store_path: str = "./vectorstore"
    embedding_model: str = "all-MiniLM-L6-v2"

    # ---- Data ----
    data_dir: str = "./data"
    raw_data_file: str = "./data/shl_raw.json"
    clean_data_file: str = "./data/shl_catalog.json"

    # ---- App ----
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_env: str = "development"
    log_level: str = "INFO"

    # ---- Retrieval ----
    top_k_results: int = 10
    similarity_threshold: float = 0.3

    # ---- Conversation ----
    max_turns: int = 8
    max_tokens: int = 2048
    temperature: float = 0.3

    # ---- Security ----
    enable_rate_limiting: bool = False
    max_requests_per_minute: int = 60

    # ---- Cache ----
    enable_cache: bool = True
    cache_dir: str = "./data/cache"
    cache_ttl: int = 3600

    # ---- SHL ----
    shl_catalog_url: str = "https://www.shl.com/solutions/products/product-catalog/"

    @property
    def active_api_key(self) -> str:
        """Return the API key for the configured provider."""
        mapping = {
            "openai": self.openai_api_key,
            "groq": self.groq_api_key,
            "google": self.google_api_key,
            "openrouter": self.openrouter_api_key,
        }
        return mapping.get(self.llm_provider, "")

    @property
    def active_model(self) -> str:
        """Return the model name for the configured provider."""
        mapping = {
            "openai": self.openai_model,
            "groq": self.groq_model,
            "google": self.google_model,
            "openrouter": self.openrouter_model,
        }
        return mapping.get(self.llm_provider, self.groq_model)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton settings instance."""
    return Settings()
