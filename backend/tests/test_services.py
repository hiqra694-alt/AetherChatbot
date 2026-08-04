import json
import pytest
import sys
import os
from unittest.mock import MagicMock, AsyncMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.chat.schemas import Message
from api.chat.services import ChatService, SYSTEM_PROMPT
from providers.mock import MockProvider


class ToolCallingFakeProvider:
    """
    Captures the `tools` kwarg it's called with on every turn, and — when
    scripted with `tool_call_script` — yields a tool_calls chunk on the
    first turn before falling back to plain text on the next turn. Used to
    exercise the agentic tool-call loop in ChatService.stream_chat without
    hitting a real LLM provider.
    """
    def __init__(self, tool_call_script=None, final_text="Done."):
        self.tool_call_script = tool_call_script
        self.final_text = final_text
        self.calls_tools = []

    async def stream_response(self, messages, tools=None):
        self.calls_tools.append(tools)
        if self.tool_call_script and len(self.calls_tools) == 1:
            yield {"type": "tool_calls", "tool_calls": self.tool_call_script}
        else:
            for word in self.final_text.split(" "):
                yield word + " "

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
    ):
        chunks.append(chunk)

    assert len(history) > 0
    assert history[0].role == "system"
    assert history[0].content == SYSTEM_PROMPT.content
    assert "I don't know based on the provided documents" in history[0].content

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
    ):
        chunks.append(chunk)

    assert history[0].role == "system"
    assert history[0].content == SYSTEM_PROMPT.content
    assert history[1].role == "user"

@pytest.mark.asyncio
async def test_stream_chat_scopes_rag_to_uploaded_document(monkeypatch):
    """
    When a file is attached in the same request as the message, retrieval
    must be scoped to that document_name so a generic prompt like "summarize
    the pdf" is grounded in the file just uploaded rather than an older
    document that happens to rank higher on raw embedding similarity.
    """
    history = [Message(role="user", content="summarize the pdf")]
    provider = MockProvider()
    mock_supabase = MagicMock()

    captured_kwargs = {}

    async def fake_get_relevant_context(supabase, query, user_id, top_k=3, document_name=None):
        captured_kwargs["document_name"] = document_name
        return []

    monkeypatch.setattr("api.chat.services.get_relevant_context", fake_get_relevant_context)

    async def request_is_disconnected():
        return False

    async for _ in ChatService.stream_chat(
        provider_instance=provider,
        history=history,
        supabase=mock_supabase,
        session_id="test_sess",
        provider_name="mock",
        request_is_disconnected=request_is_disconnected,
        user_id="user-123",
        scoped_document_name="IQRA HAMEED_CV.pdf"
    ):
        pass

    assert captured_kwargs["document_name"] == "IQRA HAMEED_CV.pdf"

@pytest.mark.asyncio
async def test_stream_chat_no_file_offers_knowledge_base_tool_without_auto_retrieval(monkeypatch):
    """
    Without a file attached in the request (agentic mode), retrieval must
    NOT happen automatically — search_knowledge_base is merely offered as a
    tool, and get_relevant_context should only run if the model itself
    decides to call it. A provider that never emits a tool call (like
    MockProvider) must produce zero Voyage/Supabase round trips.
    """
    history = [Message(role="user", content="what did we discuss earlier?")]
    provider = MockProvider()
    mock_supabase = MagicMock()

    fake_get_relevant_context = AsyncMock(return_value=[])
    monkeypatch.setattr("api.chat.services.get_relevant_context", fake_get_relevant_context)

    async def request_is_disconnected():
        return False

    async for _ in ChatService.stream_chat(
        provider_instance=provider,
        history=history,
        supabase=mock_supabase,
        session_id="test_sess",
        provider_name="mock",
        request_is_disconnected=request_is_disconnected,
        user_id="user-123"
    ):
        pass

    fake_get_relevant_context.assert_not_called()

@pytest.mark.asyncio
async def test_stream_chat_with_file_passes_no_tools():
    """
    BRANCH A: when a file is attached (scoped_document_name set), retrieval
    already happened deterministically in the system prompt, so the model
    must not be offered any tools this turn.
    """
    history = [Message(role="user", content="summarize the pdf")]
    provider = ToolCallingFakeProvider(final_text="Here is a summary.")
    mock_supabase = MagicMock()

    async def request_is_disconnected():
        return False

    async for _ in ChatService.stream_chat(
        provider_instance=provider,
        history=history,
        supabase=mock_supabase,
        session_id="test_sess",
        provider_name="mock",
        request_is_disconnected=request_is_disconnected,
        user_id="user-123",
        scoped_document_name="IQRA HAMEED_CV.pdf"
    ):
        pass

    assert provider.calls_tools == [[]]

@pytest.mark.asyncio
async def test_stream_chat_agentic_tool_call_triggers_knowledge_base_search(monkeypatch):
    """
    BRANCH B agentic flow: the model calls search_knowledge_base itself.
    The backend must intercept that tool call, run get_relevant_context
    scoped to the query it provided, feed the result back as a tool
    message, and stream the model's final answer.
    """
    history = [Message(role="user", content="what does my CV say about SwimAI?")]

    tool_call_script = [{
        "id": "call_1",
        "type": "function",
        "function": {"name": "search_knowledge_base", "arguments": json.dumps({"query": "SwimAI project"})}
    }]
    provider = ToolCallingFakeProvider(tool_call_script=tool_call_script, final_text="SwimAI optimizes club operations.")
    mock_supabase = MagicMock()

    captured_kwargs = {}

    async def fake_get_relevant_context(supabase, query, user_id, top_k=3, document_name=None):
        captured_kwargs["query"] = query
        captured_kwargs["user_id"] = user_id
        captured_kwargs["document_name"] = document_name
        return []

    # search_knowledge_base (api/chat/tools.py) imports get_relevant_context
    # directly from api.documents.services, so it must be patched at that
    # call site rather than api.chat.services (which only uses it for the
    # Branch A file-attached path).
    monkeypatch.setattr("api.chat.tools.get_relevant_context", fake_get_relevant_context)

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
        user_id="user-123"
    ):
        chunks.append(chunk)

    assert captured_kwargs["query"] == "SwimAI project"
    assert captured_kwargs["user_id"] == "user-123"
    assert captured_kwargs["document_name"] is None

    tool_messages = [m for m in history if m.role == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0].name == "search_knowledge_base"

    streamed_text = "".join(
        json.loads(c[len("data: "):])["content"]
        for c in chunks
        if c.startswith("data: ") and "content" in c
    )
    assert "SwimAI optimizes club operations" in streamed_text
