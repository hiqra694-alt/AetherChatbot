import os
import warnings
import google.generativeai as genai # pyright: ignore [reportMissingImports]
from typing import AsyncGenerator
from providers.base import BaseProvider

# Suppress the FutureWarning from the deprecated google.generativeai package
warnings.filterwarnings("ignore", category=FutureWarning, module="google.generativeai")

class GeminiProvider(BaseProvider):
    def __init__(self):
        # Read from environment (loaded by config.py via dotenv)
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not configured.")
        genai.configure(api_key=api_key)

    async def stream_response(self, messages: list, tools: list = None) -> AsyncGenerator[str, None]:
        contents = []
        for m in messages:
            msg_role = getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else "user")
            msg_content = getattr(m, "content", None) or (m.get("content") if isinstance(m, dict) else "")
            role = "user" if msg_role == "user" else "model"
            contents.append({
                "role": role,
                "parts": [msg_content or ""]
            })
            
        model = genai.GenerativeModel("gemini-2.0-flash-lite")
        response = await model.generate_content_async(contents, stream=True)
        async for chunk in response:
            if chunk.text:
                yield chunk.text
