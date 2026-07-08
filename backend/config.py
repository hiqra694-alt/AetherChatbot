import os
from pydantic import Field, AliasChoices # pyright: ignore [reportMissingImports]
from pydantic_settings import BaseSettings # pyright: ignore [reportMissingImports]
from dotenv import load_dotenv # pyright: ignore [reportMissingImports]

# Resolve absolute path to .env file relative to this config.py file
config_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(config_dir, ".env")

# Load dotenv into environment variables
load_dotenv(dotenv_path=env_path)

class Settings(BaseSettings):
    gemini_api_key: str = Field("", validation_alias=AliasChoices("gemini_api_key", "GEMINI_API_KEY"))
    openai_api_key: str = Field("", validation_alias=AliasChoices("openai_api_key", "OPENAI_API_KEY"))
    anthropic_api_key: str = Field("", validation_alias=AliasChoices("anthropic_api_key", "ANTHROPIC_API_KEY"))
    grok_api_key: str = Field("", validation_alias=AliasChoices("grok_api_key", "GROK_API_KEY"))
    supabase_url: str = Field("", validation_alias=AliasChoices("supabase_url", "SUPABASE_URL", "NEXT_PUBLIC_SUPABASE_URL"))
    supabase_anon_key: str = Field("", validation_alias=AliasChoices("supabase_anon_key", "SUPABASE_ANON_KEY", "NEXT_PUBLIC_SUPABASE_ANON_KEY"))

    class Config:
        env_file = env_path
        extra = "ignore"

settings = Settings()
