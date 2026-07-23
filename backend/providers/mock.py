import asyncio
from typing import AsyncGenerator
from providers.base import BaseProvider

class MockProvider(BaseProvider):
    async def stream_response(self, messages: list, tools: list = None) -> AsyncGenerator[str, None]:
        response_text = (
            "This is a simulated streaming response from the Mock AI Provider. "
            "It runs entirely locally on the FastAPI backend without requiring any API keys. "
            "This allows you to verify that the frontend's Server-Sent Events (SSE) listener, "
            "Supabase database connection, and auth middleware are working seamlessly."
        )
        
        words = response_text.split(" ")
        for i, word in enumerate(words):
            chunk = (" " if i > 0 else "") + word
            yield chunk
            await asyncio.sleep(0.05)
