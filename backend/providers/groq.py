from openai import AsyncOpenAI # pyright: ignore [reportMissingImports]
from typing import AsyncGenerator, List
import re
from providers.base import BaseProvider
from core.config import get_settings
from api.chat.schemas import Message

class GroqProvider(BaseProvider):
    def __init__(self):
        settings = get_settings()
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY is not configured.")
        self.client = AsyncOpenAI(api_key=settings.groq_api_key, base_url="https://api.groq.com/openai/v1")

    async def stream_response(self, messages: List[Message], tools: list = None) -> AsyncGenerator[str, None]:
        formatted_messages = []
        for m in messages:
            msg = {
                "role": m.role,
                "content": m.content if m.content is not None else ""
            }
            if m.tool_calls:
                msg["tool_calls"] = m.tool_calls
            if m.tool_call_id:
                msg["tool_call_id"] = m.tool_call_id
            if m.name:
                msg["name"] = m.name
            formatted_messages.append(msg)
            
        kwargs = {
            "model": "llama-3.1-8b-instant",
            "messages": formatted_messages,
            "stream": True
        }
        if tools:
            kwargs["tools"] = tools

        stream = await self.client.chat.completions.create(**kwargs)
        
        tool_calls = {}
        buffer = ""

        async for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'tool_calls') and delta.tool_calls:
                    for tc in delta.tool_calls:
                        index = tc.index
                        if index not in tool_calls:
                            tool_calls[index] = {
                                "id": tc.id or "",
                                "type": "function",
                                "function": {
                                    "name": tc.function.name if tc.function and tc.function.name else "",
                                    "arguments": tc.function.arguments if tc.function and tc.function.arguments else ""
                                }
                            }
                        else:
                            if tc.id:
                                tool_calls[index]["id"] = tc.id
                            if tc.function:
                                if tc.function.name:
                                    tool_calls[index]["function"]["name"] += tc.function.name
                                if tc.function.arguments:
                                    tool_calls[index]["function"]["arguments"] += tc.function.arguments
                
                content = delta.content
                if content:
                    buffer += content
                    
                    # 1. Strip complete function/tool tags (with or without leading '<')
                    buffer = re.sub(r'<?function=.*?</function>', '', buffer, flags=re.DOTALL | re.IGNORECASE)
                    buffer = re.sub(r'<?tool_call.*?</tool_call>', '', buffer, flags=re.DOTALL | re.IGNORECASE)
                    
                    # 2. Strip newline-terminated function/tool lines
                    buffer = re.sub(r'(?:^|\n)<?function=[^\n]*\n', '\n', buffer, flags=re.DOTALL | re.IGNORECASE)
                    buffer = re.sub(r'(?:^|\n)<?tool_call[^\n]*\n', '\n', buffer, flags=re.DOTALL | re.IGNORECASE)
                    
                    # 3. If buffer contains an active/unclosed function or tool tag, hold until closed or ended
                    if re.search(r'<?function=', buffer, re.IGNORECASE) or re.search(r'<?tool_call', buffer, re.IGNORECASE):
                        continue
                        
                    # 4. Check if buffer ends with a potential tag prefix
                    potential_prefixes = [
                        "<", "<f", "<fu", "<fun", "<func", "<funct", "<functi", "<functio", "<function", "<function=",
                        "fun", "func", "funct", "functi", "functio", "function", "function=",
                        "<t", "<to", "<too", "<tool", "<tool_", "<tool_c", "<tool_ca", "<tool_cal", "<tool_call",
                        "too", "tool", "tool_", "tool_c", "tool_ca", "tool_cal", "tool_call"
                    ]
                    is_prefix = False
                    for prefix in potential_prefixes:
                        if buffer.endswith(prefix):
                            is_prefix = True
                            break
                            
                    if is_prefix:
                        continue
                        
                    yield buffer
                    buffer = ""

        if buffer:
            buffer = re.sub(r'<?function=.*$', '', buffer, flags=re.DOTALL | re.IGNORECASE)
            buffer = re.sub(r'<?tool_call.*$', '', buffer, flags=re.DOTALL | re.IGNORECASE)
            if buffer:
                yield buffer

        if tool_calls:
            yield {
                "type": "tool_calls",
                "tool_calls": list(tool_calls.values())
            }
