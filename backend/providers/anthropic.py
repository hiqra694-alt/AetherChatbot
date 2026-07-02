import os
from anthropic import AsyncAnthropic # pyright: ignore [reportMissingImports]
from typing import AsyncGenerator
from providers.base import BaseProvider

class AnthropicProvider(BaseProvider):
    def __init__(self):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not configured.")
        self.client = AsyncAnthropic(api_key=api_key)

    async def stream_response(self, messages: list) -> AsyncGenerator[str, None]:
        formatted_messages = []
        for m in messages:
            role = m["role"]
            if role not in ["user", "assistant"]:
                role = "user"
            formatted_messages.append({
                "role": role,
                "content": m["content"]
            })
            
        async with self.client.messages.stream(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1024,
            messages=formatted_messages
        ) as stream:
            async for text in stream.text_stream:
                yield text
