from openai import AsyncOpenAI # pyright: ignore [reportMissingImports]
from typing import AsyncGenerator, List
from providers.base import BaseProvider
from core.config import get_settings
from api.chat.schemas import Message

class GroqProvider(BaseProvider):
    def __init__(self):
        settings = get_settings()
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY is not configured.")
        self.client = AsyncOpenAI(api_key=settings.groq_api_key, base_url="https://api.groq.com/openai/v1")

    async def stream_response(self, messages: List[Message]) -> AsyncGenerator[str, None]:
        formatted_messages = []
        for m in messages:
            formatted_messages.append({
                "role": m.role,
                "content": m.content
            })
            
        stream = await self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=formatted_messages,
            stream=True
        )
        
        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content
