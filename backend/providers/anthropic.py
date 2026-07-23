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

    async def stream_response(self, messages: list, tools: list = None) -> AsyncGenerator[str, None]:
        formatted_messages = []
        for m in messages:
            msg_role = getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else "user")
            msg_content = getattr(m, "content", None) or (m.get("content") if isinstance(m, dict) else "")
            role = msg_role if msg_role in ["user", "assistant"] else "user"
            formatted_messages.append({
                "role": role,
                "content": msg_content or ""
            })
            
        async with self.client.messages.stream(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1024,
            messages=formatted_messages
        ) as stream:
            async for text in stream.text_stream:
                yield text
