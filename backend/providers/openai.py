import os
from openai import AsyncOpenAI # pyright: ignore [reportMissingImports]
from typing import AsyncGenerator
from providers.base import BaseProvider

class OpenAIProvider(BaseProvider):
    def __init__(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not configured.")
        self.client = AsyncOpenAI(api_key=api_key)

    async def stream_response(self, messages: list) -> AsyncGenerator[str, None]:
        formatted_messages = []
        for m in messages:
            formatted_messages.append({
                "role": m["role"],
                "content": m["content"]
            })
            
        stream = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=formatted_messages,
            stream=True
        )
        
        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content
