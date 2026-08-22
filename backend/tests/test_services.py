import contextlib
import json
import pytest
import sys
import os
from unittest.mock import MagicMock, AsyncMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp import types
from api.chat.schemas import Message
from api.chat.services import ChatService, SYSTEM_PROMPT
from api.chat.tools import ALL_TOOL_NAMES
from api.memory.schemas import UserMemoryProfile
from connector_integrations.connector_manager import MCPClientManager, MCPServerConnection, mcp_manager
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


class PreambleThenToolCallFakeProvider:
    """
    Simulates the exact production regression: llama-3.3-70b-versatile
    streams narrated preamble content ("Let me check that for you.") BEFORE
    emitting a native tool_calls delta in the very same turn -- despite
    SYSTEM_PROMPT forbidding it -- then answers for real on the synthesis
    turn. Used to verify ChatService.stream_chat retracts that preamble
    instead of merging it with the real, post-tool answer.
    """
    def __init__(self, preamble, tool_call_script, final_text="Done."):
        self.preamble = preamble
        self.tool_call_script = tool_call_script
        self.final_text = final_text
        self.calls_tools = []

    async def stream_response(self, messages, tools=None):
        self.calls_tools.append(tools)
        if len(self.calls_tools) == 1:
            for word in self.preamble.split(" "):
                yield word + " "
            yield {"type": "tool_calls", "tool_calls": self.tool_call_script}
        else:
            for word in self.final_text.split(" "):
                yield word + " "


class TextTagFakeProvider:
    """
    Simulates a Groq/llama-3.3-70b-versatile turn where the tool call
    arrived as a text_tool_call event (the provider adapter's dual-mode
    fallback for a tool call emitted as a raw `<tool_name>...</tool_name>`
    tag) rather than a native tool_calls object. Used to exercise
    ChatService.stream_chat's fallback handling independent of the Groq
    adapter itself.
    """
    def __init__(self, tool_name, tool_arguments, final_text="Done."):
        self.tool_name = tool_name
        self.tool_arguments = tool_arguments
        self.final_text = final_text
        self.calls_tools = []

    async def stream_response(self, messages, tools=None):
        self.calls_tools.append(tools)
        if len(self.calls_tools) == 1:
            yield {"type": "text_tool_call", "name": self.tool_name, "arguments": self.tool_arguments}
        else:
            for word in self.final_text.split(" "):
                yield word + " "


class SynthesisValidationErrorFakeProvider:
    """
    Simulates the real production failure mode behind "attempted to call
    tool X which was not in request.tools": a native tool call succeeds on
    turn 1, but the follow-up synthesis turn (turn_tools=[]) raises the
    provider API's tool-call validation error instead of streaming text --
    e.g. the model still attempts a call even though no tools were bound
    for that request. Always raises on every turn after the first, so both
    the single silent retry and the eventual graceful-degrade path are
    exercised.
    """
    def __init__(self, tool_call_script):
        self.tool_call_script = tool_call_script
        self.calls_tools = []

    async def stream_response(self, messages, tools=None):
        self.calls_tools.append(tools)
        if len(self.calls_tools) == 1:
            yield {"type": "tool_calls", "tool_calls": self.tool_call_script}
            return
        raise Exception(
            "Error code: 400 - {'error': {'message': \"Failed to call a function. "
            "tool call validation failed: attempted to call tool 'search_knowledge_base' "
            "which was not in request.tools\", 'type': 'invalid_request_error'}}"
        )
        yield  # pragma: no cover -- unreachable, keeps this an async generator

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
    assert history[0].content.startswith(SYSTEM_PROMPT.content)
    assert "<current_time>" in history[0].content
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
    assert history[0].content.startswith(SYSTEM_PROMPT.content)
    assert "<current_time>" in history[0].content
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

    async def fake_get_relevant_context(supabase, query, user_id, session_id, top_k=3, document_name=None):
        captured_kwargs["session_id"] = session_id
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

    assert captured_kwargs["session_id"] == "test_sess"
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


def _fake_mcp_connection(connector_id, tool_name):
    connection = MCPServerConnection(connector_id, f"http://fake/{connector_id}/sse")
    connection.tools = [types.Tool(
        name=tool_name,
        description=f"Fake tool for connector {connector_id}.",
        inputSchema={"type": "object", "properties": {}},
    )]
    connection.session = AsyncMock()
    return connection


@pytest.mark.asyncio
async def test_stream_chat_enabled_connectors_filters_mcp_tools_offered_to_provider(monkeypatch):
    """
    Phase 3: when the request carries enabled_connectors, only the enabled
    connector's MCP tools should end up in the `tools` payload handed to the
    provider (and therefore to Groq) -- other connected connectors' tools
    must be excluded, while every native tool stays available regardless.
    """
    fake_manager = MCPClientManager(servers=[])
    fake_manager._connections["some_tool"] = _fake_mcp_connection("some_tool", "some_tool_action")
    fake_manager._connections["other_connector"] = _fake_mcp_connection("other_connector", "other_connector_action")
    monkeypatch.setattr("api.chat.services.mcp_manager", fake_manager)

    history = [Message(role="user", content="do something")]
    provider = ToolCallingFakeProvider(final_text="Done.")
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
        enabled_connectors=["some_tool"],
    ):
        pass

    offered_names = {t["function"]["name"] for t in provider.calls_tools[0]}
    assert set(ALL_TOOL_NAMES).issubset(offered_names)
    assert "some_tool_action" in offered_names
    assert "other_connector_action" not in offered_names


@pytest.mark.asyncio
async def test_stream_chat_no_enabled_connectors_offers_every_connected_connector(monkeypatch):
    """
    Backward compatibility: omitting enabled_connectors entirely (None,
    the default when the frontend sends nothing) must still offer every
    connected connector's tools, same as before this option existed.
    """
    fake_manager = MCPClientManager(servers=[])
    fake_manager._connections["some_tool"] = _fake_mcp_connection("some_tool", "some_tool_action")
    fake_manager._connections["other_connector"] = _fake_mcp_connection("other_connector", "other_connector_action")
    monkeypatch.setattr("api.chat.services.mcp_manager", fake_manager)

    history = [Message(role="user", content="do something")]
    provider = ToolCallingFakeProvider(final_text="Done.")
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
    ):
        pass

    offered_names = {t["function"]["name"] for t in provider.calls_tools[0]}
    assert "some_tool_action" in offered_names
    assert "other_connector_action" in offered_names


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

    async def fake_get_relevant_context(supabase, query, user_id, session_id, top_k=3, document_name=None):
        captured_kwargs["query"] = query
        captured_kwargs["user_id"] = user_id
        captured_kwargs["session_id"] = session_id
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
    assert captured_kwargs["session_id"] == "test_sess"
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


@pytest.mark.asyncio
async def test_stream_chat_preamble_before_tool_call_is_retracted_not_merged(monkeypatch):
    """
    Regression coverage for the merged-double-answer bug: a turn that
    streams narrated preamble text before also deciding to call a tool must
    not leave that preamble concatenated with the real, post-tool answer.
    The preamble is allowed to have already streamed live (SSE can't be
    un-sent), but ChatService must emit exactly one retract event right
    after it, and the persisted/final message content must contain only
    the real answer.
    """
    history = [Message(role="user", content="what's the weather in lahore")]

    tool_call_script = [{
        "id": "call_1",
        "type": "function",
        "function": {"name": "get_weather", "arguments": json.dumps({"city": "Lahore"})}
    }]
    provider = PreambleThenToolCallFakeProvider(
        preamble="Let me check that for you.",
        tool_call_script=tool_call_script,
        final_text="It is sunny and 28 degrees in Lahore right now.",
    )

    inserted_rows = []

    def fake_insert(row):
        inserted_rows.append(row)
        result = MagicMock()
        result.execute.return_value = MagicMock(data=[row])
        return result

    mock_messages_table = MagicMock()
    mock_messages_table.insert.side_effect = fake_insert
    mock_supabase = MagicMock()
    mock_supabase.table.return_value = mock_messages_table

    async def fake_get_weather(city, unit="celsius"):
        return json.dumps({"city": city, "temperature": 28, "unit": unit})

    monkeypatch.setattr("api.chat.tools.get_weather", fake_get_weather)

    async def request_is_disconnected():
        return False

    chunks = []
    async for chunk in ChatService.stream_chat(
        provider_instance=provider,
        history=history,
        supabase=mock_supabase,
        session_id="test_sess",
        provider_name="groq",
        request_is_disconnected=request_is_disconnected,
        user_id="user-123",
    ):
        chunks.append(chunk)

    events = []
    for c in chunks:
        if not c.startswith("data: ") or c.strip() == "data: [DONE]":
            continue
        parsed = json.loads(c[len("data: "):])
        if "content" in parsed:
            events.append(("content", parsed["content"]))
        elif "retract" in parsed:
            events.append(("retract", True))

    retract_positions = [i for i, (kind, _) in enumerate(events) if kind == "retract"]
    assert len(retract_positions) == 1
    retract_pos = retract_positions[0]

    before_retract = "".join(v for kind, v in events[:retract_pos] if kind == "content")
    after_retract = "".join(v for kind, v in events[retract_pos + 1:] if kind == "content")

    # The preamble did stream live (proving the fix preserves real-time
    # streaming) but strictly before the retract event...
    assert "Let me check that for you." in before_retract
    # ...and the real answer streams strictly after it.
    assert "It is sunny and 28 degrees in Lahore right now." in after_retract
    assert "Let me check" not in after_retract

    # The persisted assistant message must contain only the real answer --
    # never the retracted preamble.
    assistant_rows = [r for r in inserted_rows if r.get("role") == "assistant"]
    assert len(assistant_rows) == 1
    assert "Let me check" not in assistant_rows[0]["content"]
    assert "It is sunny and 28 degrees in Lahore right now." in assistant_rows[0]["content"]


@pytest.mark.asyncio
async def test_stream_chat_native_tool_call_with_truncated_arguments_still_executes(monkeypatch):
    """
    A native tool_calls delta's accumulated `arguments` string can itself
    be dirty/truncated (e.g. a flaky generation cut off mid-value) rather
    than a clean JSON object. Previously this fell back to `tool_args = {}`
    on the first JSONDecodeError, silently dropping a required argument
    like `city` and guaranteeing a "could not find" style tool failure.
    ChatService must instead recover the real argument via
    repair_tool_arguments and execute the tool with it intact.
    """
    history = [Message(role="user", content="what's the weather in san francisco")]

    tool_call_script = [{
        "id": "call_1",
        "type": "function",
        # Deliberately truncated mid-value, as if cut off by a token limit.
        "function": {"name": "get_weather", "arguments": '{"city": "San Francisco'}
    }]
    provider = ToolCallingFakeProvider(tool_call_script=tool_call_script, final_text="It is sunny in San Francisco.")
    mock_supabase = MagicMock()

    captured_args = {}

    async def fake_get_weather(city, unit="celsius"):
        captured_args["city"] = city
        return json.dumps({"city": city, "temperature": 22, "unit": unit})

    monkeypatch.setattr("api.chat.tools.get_weather", fake_get_weather)

    async def request_is_disconnected():
        return False

    chunks = []
    async for chunk in ChatService.stream_chat(
        provider_instance=provider,
        history=history,
        supabase=mock_supabase,
        session_id="test_sess",
        provider_name="groq",
        request_is_disconnected=request_is_disconnected,
        user_id="user-123",
    ):
        chunks.append(chunk)

    assert captured_args["city"] == "San Francisco"

    tool_messages = [m for m in history if m.role == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0].name == "get_weather"

    streamed_text = "".join(
        json.loads(c[len("data: "):])["content"]
        for c in chunks
        if c.startswith("data: ") and '"content"' in c
    )
    assert "It is sunny in San Francisco." in streamed_text


@pytest.mark.asyncio
async def test_stream_chat_text_tool_call_fallback_executes_and_synthesizes(monkeypatch):
    """
    Regression coverage for the raw-tag leak: when a provider (Groq on
    llama-3.3-70b-versatile) emits a tool call as a text_tool_call event
    instead of a native tool_calls object, ChatService must still execute
    the real tool, feed its result back as a proper tool message, and
    stream a clean synthesis answer -- never the raw tag, and never the
    generic "error formatting its response" failure.
    """
    history = [Message(role="user", content="how is weather in lahore today")]

    provider = TextTagFakeProvider(
        tool_name="get_weather",
        tool_arguments=json.dumps({"city": "Lahore"}),
        final_text="It is sunny in Lahore.",
    )
    mock_supabase = MagicMock()

    captured_args = {}

    async def fake_get_weather(city, unit="celsius"):
        captured_args["city"] = city
        return json.dumps({"city": city, "temperature": 30, "unit": unit})

    monkeypatch.setattr("api.chat.tools.get_weather", fake_get_weather)

    async def request_is_disconnected():
        return False

    chunks = []
    async for chunk in ChatService.stream_chat(
        provider_instance=provider,
        history=history,
        supabase=mock_supabase,
        session_id="test_sess",
        provider_name="groq",
        request_is_disconnected=request_is_disconnected,
        user_id="user-123",
    ):
        chunks.append(chunk)

    assert captured_args["city"] == "Lahore"

    tool_messages = [m for m in history if m.role == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0].name == "get_weather"

    error_events = [c for c in chunks if c.startswith("data: ") and '"error"' in c]
    assert error_events == []

    streamed_text = "".join(
        json.loads(c[len("data: "):])["content"]
        for c in chunks
        if c.startswith("data: ") and '"content"' in c
    )
    assert "It is sunny in Lahore." in streamed_text
    assert "<get_weather>" not in streamed_text


@pytest.mark.asyncio
async def test_stream_chat_text_tool_call_not_offered_this_turn_is_discarded(monkeypatch):
    """
    Defense-in-depth for "never invoke a tool outside what was explicitly
    offered this turn": if a provider emits a text_tool_call event naming a
    real tool that simply wasn't offered this turn, ChatService must
    re-validate it against turn_tool_names before executing -- independent
    of whatever gating the provider adapter itself did. BRANCH A (a file
    attached) offers zero tools, so any text_tool_call here names something
    outside the granted set and must be discarded rather than executed.
    """
    history = [Message(role="user", content="summarize the pdf")]

    provider = TextTagFakeProvider(
        tool_name="get_weather",
        tool_arguments=json.dumps({"city": "Lahore"}),
        final_text="Here is a summary.",
    )
    mock_supabase = MagicMock()

    weather_called = AsyncMock(return_value=json.dumps({"city": "Lahore", "temperature": 30}))
    monkeypatch.setattr("api.chat.tools.get_weather", weather_called)

    async def request_is_disconnected():
        return False

    chunks = []
    async for chunk in ChatService.stream_chat(
        provider_instance=provider,
        history=history,
        supabase=mock_supabase,
        session_id="test_sess",
        provider_name="groq",
        request_is_disconnected=request_is_disconnected,
        user_id="user-123",
        scoped_document_name="IQRA HAMEED_CV.pdf",
    ):
        chunks.append(chunk)

    weather_called.assert_not_called()
    tool_messages = [m for m in history if m.role == "tool"]
    assert tool_messages == []

    error_events = [c for c in chunks if c.startswith("data: ") and '"error"' in c]
    assert error_events == []


@pytest.mark.asyncio
async def test_stream_chat_synthesis_turn_validation_error_degrades_silently(monkeypatch):
    """
    Regression coverage for "attempted to call tool X which was not in
    request.tools": when the synthesis turn (after a real tool call already
    ran) hits this validation error, ChatService must not surface a
    user-facing error event -- it should log a warning, retry once with
    tool-calling forced off, and if that still fails, degrade gracefully
    (finish the stream with whatever text exists, no error) rather than
    letting the exception propagate to the frontend.
    """
    history = [Message(role="user", content="what does my CV say about SwimAI?")]

    tool_call_script = [{
        "id": "call_1",
        "type": "function",
        "function": {"name": "search_knowledge_base", "arguments": json.dumps({"query": "SwimAI"})}
    }]
    provider = SynthesisValidationErrorFakeProvider(tool_call_script)
    mock_supabase = MagicMock()

    fake_get_relevant_context = AsyncMock(return_value=[])
    monkeypatch.setattr("api.chat.tools.get_relevant_context", fake_get_relevant_context)

    async def request_is_disconnected():
        return False

    chunks = []
    async for chunk in ChatService.stream_chat(
        provider_instance=provider,
        history=history,
        supabase=mock_supabase,
        session_id="test_sess",
        provider_name="groq",
        request_is_disconnected=request_is_disconnected,
        user_id="user-123",
    ):
        chunks.append(chunk)

    error_events = [c for c in chunks if c.startswith("data: ") and '"error"' in c]
    assert error_events == []
    assert any(c.strip() == "data: [DONE]" for c in chunks)
    # 3 calls total: the original tool-calling turn (tools offered), then
    # the synthesis turn's first attempt and its forced-tools-off retry
    # (both empty, since this was already a post-tool synthesis turn) --
    # after which the loop gives up gracefully instead of raising a third
    # time.
    assert len(provider.calls_tools) == 3
    assert provider.calls_tools[0]
    assert provider.calls_tools[1] == []
    assert provider.calls_tools[2] == []


@pytest.mark.asyncio
async def test_generate_and_store_title_updates_session(monkeypatch):
    """
    generate_and_store_title reads the session's first assistant reply back
    out of the DB (stripping the embedded <aether-sources> tag), asks a
    provider for a short title, cleans up quoting/whitespace, and writes the
    result onto chat_sessions.title.
    """
    class FakeTitleProvider:
        async def stream_response(self, messages, tools=None):
            for word in ['"Fibonacci', 'Python', 'Function"']:
                yield word + " "

    monkeypatch.setattr(
        "api.chat.services.ProviderFactory.get_provider",
        lambda name: FakeTitleProvider(),
    )

    messages_table = MagicMock()
    messages_table.select.return_value.eq.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = MagicMock(
        data=[{"content": "Here is a Fibonacci function.<aether-sources>[{\"title\": \"x\"}]</aether-sources>"}]
    )
    sessions_table = MagicMock()

    mock_supabase = MagicMock()
    mock_supabase.table.side_effect = lambda name: messages_table if name == "messages" else sessions_table

    await ChatService.generate_and_store_title(mock_supabase, "sess-1", "Write a python fibonacci function")

    sessions_table.update.assert_called_once_with({"title": "Fibonacci Python Function"})
    sessions_table.update.return_value.eq.assert_called_once_with("id", "sess-1")


@pytest.mark.asyncio
async def test_generate_and_store_title_skips_update_on_empty_title(monkeypatch):
    """
    If the provider yields nothing usable, no chat_sessions row should be
    touched -- the session keeps its original placeholder title rather than
    being overwritten with an empty string.
    """
    class EmptyTitleProvider:
        async def stream_response(self, messages, tools=None):
            return
            yield  # pragma: no cover -- unreachable, keeps this an async generator

    monkeypatch.setattr(
        "api.chat.services.ProviderFactory.get_provider",
        lambda name: EmptyTitleProvider(),
    )

    messages_table = MagicMock()
    messages_table.select.return_value.eq.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = MagicMock(
        data=[{"content": "Some answer."}]
    )
    sessions_table = MagicMock()

    mock_supabase = MagicMock()
    mock_supabase.table.side_effect = lambda name: messages_table if name == "messages" else sessions_table

    await ChatService.generate_and_store_title(mock_supabase, "sess-1", "hi")

    sessions_table.update.assert_not_called()


@pytest.mark.asyncio
async def test_generate_and_store_title_never_raises_on_provider_failure(monkeypatch):
    """
    Title generation is a best-effort side task scheduled after the response
    has already been streamed to the user -- a provider error (e.g. missing
    API key, rate limit) must be swallowed, never propagated, since there's
    no request left to fail.
    """
    def raise_missing_key(name):
        raise ValueError("GROQ_API_KEY is not configured.")

    monkeypatch.setattr("api.chat.services.ProviderFactory.get_provider", raise_missing_key)

    mock_supabase = MagicMock()
    mock_supabase.table.return_value.select.return_value.eq.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = MagicMock(
        data=[{"content": "Some answer."}]
    )

    await ChatService.generate_and_store_title(mock_supabase, "sess-1", "hi")


@pytest.mark.asyncio
async def test_build_system_prompt_injects_user_memory(monkeypatch):
    """
    Continuous memory profile: the user's single narrative paragraph must be
    injected into the system prompt on every turn, independent of whether a
    document is attached, and without mutating the shared SYSTEM_PROMPT
    singleton.
    """
    fake_profile = UserMemoryProfile(narrative="Works at Zylo as a backend engineer.", updated_at="2026-07-23T10:00:00Z")

    async def fake_get_user_memory(supabase, user_id):
        assert user_id == "user-123"
        return fake_profile

    monkeypatch.setattr("api.chat.services.get_user_memory", fake_get_user_memory)

    mock_supabase = MagicMock()
    system_message = await ChatService._build_system_prompt(mock_supabase, "user-123", "test_sess", "hello")

    assert system_message is not SYSTEM_PROMPT
    assert system_message.content.startswith(SYSTEM_PROMPT.content)
    assert "<user_memory>" in system_message.content
    assert "Works at Zylo as a backend engineer." in system_message.content
    # The shared singleton must never be mutated by this injection.
    assert "Works at Zylo" not in SYSTEM_PROMPT.content


@pytest.mark.asyncio
async def test_build_system_prompt_injects_temporal_block_when_no_memory_or_document(monkeypatch):
    """
    Even with no user memory and no document context, every request must
    still get a fresh system prompt carrying the live current-time block --
    unlike the optional memory/document blocks, this one is never skipped,
    so the shared SYSTEM_PROMPT singleton is never returned as-is.
    """
    async def fake_get_user_memory(supabase, user_id):
        return None

    monkeypatch.setattr("api.chat.services.get_user_memory", fake_get_user_memory)

    mock_supabase = MagicMock()
    system_message = await ChatService._build_system_prompt(mock_supabase, "user-123", "test_sess", "hello")

    assert system_message is not SYSTEM_PROMPT
    assert system_message.content.startswith(SYSTEM_PROMPT.content)
    assert "<current_time>" in system_message.content
    assert "<user_memory>" not in system_message.content
    assert "<knowledge_base_context>" not in system_message.content
    # The shared singleton must never be mutated by this injection.
    assert "<current_time>" not in SYSTEM_PROMPT.content


@pytest.mark.asyncio
async def test_build_system_prompt_still_injects_temporal_block_on_memory_failure(monkeypatch):
    async def raise_error(supabase, user_id):
        raise RuntimeError("db unavailable")

    monkeypatch.setattr("api.chat.services.get_user_memory", raise_error)

    mock_supabase = MagicMock()
    system_message = await ChatService._build_system_prompt(mock_supabase, "user-123", "test_sess", "hello")

    assert system_message is not SYSTEM_PROMPT
    assert "<current_time>" in system_message.content
    assert "<user_memory>" not in system_message.content


@pytest.mark.asyncio
async def test_build_system_prompt_document_retrieval_failure_falls_back_to_empty_context(monkeypatch):
    """
    Regression coverage for the StreamReset/500 bug: a document vector
    search failure (embedding call or Supabase RPC raising, e.g. an HTTP/2
    StreamReset) must degrade to an empty context rather than propagating
    out of _build_system_prompt and crashing the chat stream.
    """
    async def fake_get_user_memory(supabase, user_id):
        return None

    async def raise_stream_reset(supabase, query, user_id, session_id, top_k=3, document_name=None):
        raise RuntimeError("StreamReset: stream reset by peer")

    monkeypatch.setattr("api.chat.services.get_user_memory", fake_get_user_memory)
    monkeypatch.setattr("api.chat.services.get_relevant_context", raise_stream_reset)

    mock_supabase = MagicMock()
    system_message = await ChatService._build_system_prompt(
        mock_supabase, "user-123", "test_sess", "summarize the pdf", document_name="cv.pdf"
    )

    # No exception propagated, and the document-context failure degrades to
    # no knowledge_base_context block -- the temporal block still comes
    # through, since it's unrelated to (and unaffected by) that retrieval.
    assert "<current_time>" in system_message.content
    assert "<knowledge_base_context>" not in system_message.content


@pytest.mark.asyncio
async def test_build_system_prompt_combines_memory_and_document_context(monkeypatch):
    """
    Both blocks are independently optional and strictly additive -- when a
    file is attached AND the user has a stored memory profile, both must
    appear in the same generated system prompt.
    """
    from api.documents.schemas import RetrievedChunk

    fake_profile = UserMemoryProfile(narrative="Prefers concise answers.", updated_at="2026-07-23T10:00:00Z")

    async def fake_get_user_memory(supabase, user_id):
        return fake_profile

    async def fake_get_relevant_context(supabase, query, user_id, session_id, top_k=3, document_name=None):
        return [RetrievedChunk(document_name=document_name, chunk_text="CV chunk text", similarity=0.9)]

    monkeypatch.setattr("api.chat.services.get_user_memory", fake_get_user_memory)
    monkeypatch.setattr("api.chat.services.get_relevant_context", fake_get_relevant_context)

    mock_supabase = MagicMock()
    system_message = await ChatService._build_system_prompt(
        mock_supabase, "user-123", "test_sess", "summarize the pdf", document_name="cv.pdf"
    )

    assert "<user_memory>" in system_message.content
    assert "Prefers concise answers." in system_message.content
    assert "<knowledge_base_context>" in system_message.content
    assert "CV chunk text" in system_message.content
