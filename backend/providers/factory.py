from providers.base import BaseProvider
from providers.mock import MockProvider
from providers.gemini import GeminiProvider
from providers.openai import OpenAIProvider
from providers.anthropic import AnthropicProvider
from providers.groq import GroqProvider

class ProviderFactory:
    @staticmethod
    def get_provider(provider_name: str) -> BaseProvider:
        name_lower = provider_name.lower().strip()
        if name_lower == "gemini":
            return GeminiProvider()
        elif name_lower == "openai":
            return OpenAIProvider()
        elif name_lower == "claude" or name_lower == "anthropic":
            return AnthropicProvider()
        elif name_lower == "groq":
            return GroqProvider()
        elif name_lower == "mock":
            return MockProvider()
        else:
            raise ValueError(f"Unknown provider type: {provider_name}")
