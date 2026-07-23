import pytest
import sys
import os
from unittest.mock import MagicMock, AsyncMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.chat.schemas import Message
from api.chat.services import ChatService, SYSTEM_PROMPT
from providers.mock import MockProvider

@pytest.mark.asyncio
async def test_system_prompt_insertion_empty_history():
    history = []
    provider = MockProvider()
    mock_supabase = MagicMock()
    
    async def request_is_disconnected():
        return False

    chunks = []
    async for chunk in ChatService.stream_chat(
        provider_instance=provider,
        history=history,
        supabase=mock_supabase,
        session_id="test_sess",
        provider_name="mock",
        request_is_disconnected=request_is_disconnected,
        use_web_search=False
    ):
        chunks.append(chunk)

    assert len(history) > 0
    assert history[0].role == "system"
    assert history[0].content == SYSTEM_PROMPT.content
    assert "TOOL RESTRICTION" in history[0].content

@pytest.mark.asyncio
async def test_system_prompt_insertion_existing_history():
    history = [Message(role="user", content="Hello")]
    provider = MockProvider()
    mock_supabase = MagicMock()
    
    async def request_is_disconnected():
        return False

    chunks = []
    async for chunk in ChatService.stream_chat(
        provider_instance=provider,
        history=history,
        supabase=mock_supabase,
        session_id="test_sess",
        provider_name="mock",
        request_is_disconnected=request_is_disconnected,
        use_web_search=False
    ):
        chunks.append(chunk)

    assert history[0].role == "system"
    assert history[0].content == SYSTEM_PROMPT.content
    assert history[1].role == "user"
