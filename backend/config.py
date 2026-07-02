import os
from pydantic_settings import BaseSettings # pyright: ignore [reportMissingImports]
from dotenv import load_dotenv # pyright: ignore [reportMissingImports]

# Resolve absolute path to .env file relative to this config.py file
config_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(config_dir, ".env")

# Load dotenv into environment variables
load_dotenv(dotenv_path=env_path)

class Settings(BaseSettings):
    gemini_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    supabase_url: str = ""
    supabase_anon_key: str = ""

    class Config:
        env_file = env_path
        extra = "ignore"

settings = Settings()
