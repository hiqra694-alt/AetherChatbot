import os
from functools import lru_cache
from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Centralized configuration management for the application.
    Loads secrets from .env file and validates them on startup.
    """
    gemini_api_key: str = Field("", validation_alias=AliasChoices("gemini_api_key", "GEMINI_API_KEY"))
    openai_api_key: str = Field("", validation_alias=AliasChoices("openai_api_key", "OPENAI_API_KEY"))
    anthropic_api_key: str = Field("", validation_alias=AliasChoices("anthropic_api_key", "ANTHROPIC_API_KEY"))
    groq_api_key: str = Field("", validation_alias=AliasChoices("groq_api_key", "GROQ_API_KEY"))
    
    supabase_url: str = Field("", validation_alias=AliasChoices("supabase_url", "SUPABASE_URL", "NEXT_PUBLIC_SUPABASE_URL"))
    supabase_anon_key: str = Field("", validation_alias=AliasChoices("supabase_anon_key", "SUPABASE_ANON_KEY", "NEXT_PUBLIC_SUPABASE_ANON_KEY"))

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

@lru_cache
def get_settings() -> Settings:
    """
    Returns a cached instance of the settings.
    Ensures configuration is only read and validated once at application startup.
    """
    return Settings()
