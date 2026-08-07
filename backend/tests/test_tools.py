import pytest
import json
import sys
import os
from unittest.mock import MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.chat.tools import (
    search_chat_history,
    SEARCH_CHAT_HISTORY_TOOL,
    SEARCH_KNOWLEDGE_BASE_TOOL,
    execute_tool,
    repair_tool_arguments,
)


def test_repair_tool_arguments_valid_json_passthrough():
    assert repair_tool_arguments("get_weather", '{"city": "Lahore", "unit": "celsius"}') == {
        "city": "Lahore", "unit": "celsius"
    }


def test_repair_tool_arguments_truncated_mid_string_value():
    # Cut off mid-value, as if a token limit hit right after the city name.
    assert repair_tool_arguments("get_weather", '{"city": "San Francisco') == {"city": "San Francisco"}


def test_repair_tool_arguments_truncated_after_trailing_comma():
    assert repair_tool_arguments(
        "duckduckgo_search", '{"search_query": "coffee shops near me",'
    ) == {"search_query": "coffee shops near me"}


def test_repair_tool_arguments_truncated_mid_second_key():
    assert repair_tool_arguments(
        "get_weather", '{"city": "Lahore", "uni'
    ) == {"city": "Lahore"}


def test_repair_tool_arguments_unrecoverable_json_falls_back_to_regex_scan():
    # Not JSON-repairable at all (garbled structure), but the required
    # field is still sitting there in plain text for the regex fallback.
    assert repair_tool_arguments(
        "get_weather", 'garbled nonsense "city": "Lahore" more garbage'
    ) == {"city": "Lahore"}


def test_repair_tool_arguments_empty_string_returns_empty_dict():
    assert repair_tool_arguments("get_weather", "") == {}
    assert repair_tool_arguments("get_weather", "   ") == {}


def test_repair_tool_arguments_totally_unrecoverable_returns_empty_dict():
    assert repair_tool_arguments("get_weather", "not json and no fields at all") == {}

@pytest.mark.asyncio
async def test_search_chat_history_tool_schema():
    assert SEARCH_CHAT_HISTORY_TOOL["function"]["name"] == "search_chat_history"
    assert "required" in SEARCH_CHAT_HISTORY_TOOL["function"]["parameters"]
    assert len(SEARCH_CHAT_HISTORY_TOOL["function"]["parameters"]["required"]) == 0
    assert "across user sessions" in SEARCH_CHAT_HISTORY_TOOL["function"]["description"]

@pytest.mark.asyncio
async def test_search_chat_history_with_specific_query():
    mock_supabase = MagicMock()
    mock_table = MagicMock()
    mock_supabase.table.return_value = mock_table
    
    mock_query = MagicMock()
    mock_table.select.return_value = mock_query
    mock_query.ilike.return_value = mock_query
    mock_query.order.return_value = mock_query
    mock_query.limit.return_value = mock_query
    
    mock_res = MagicMock()
    mock_res.data = [
        {"role": "user", "content": "Tell me about Python", "created_at": "2026-07-23T10:00:00Z"},
        {"role": "assistant", "content": "Python is a great programming language.", "created_at": "2026-07-23T10:00:05Z"}
    ]
    mock_query.execute.return_value = mock_res

    res_str = await search_chat_history(mock_supabase, session_id="sess_123", query="Python", limit=5)
    parsed = json.loads(res_str)
    
    assert "results" in parsed
    assert len(parsed["results"]) == 2
    assert parsed["results"][0]["content"] == "Python is a great programming language."

@pytest.mark.asyncio
async def test_search_chat_history_meta_query_fallback():
    mock_supabase = MagicMock()
    mock_table = MagicMock()
    mock_supabase.table.return_value = mock_table
    
    mock_query = MagicMock()
    mock_table.select.return_value = mock_query
    mock_query.order.return_value = mock_query
    mock_query.limit.return_value = mock_query
    
    mock_res = MagicMock()
    mock_res.data = [
        {"role": "user", "content": "What is the weather in Tokyo?", "created_at": "2026-07-23T09:00:00Z"}
    ]
    mock_query.execute.return_value = mock_res

    res_str = await search_chat_history(mock_supabase, session_id="sess_123", query="previous sessions", limit=5)
    parsed = json.loads(res_str)
    
    assert "results" in parsed
    assert len(parsed["results"]) == 1
    assert parsed["results"][0]["content"] == "What is the weather in Tokyo?"

@pytest.mark.asyncio
async def test_search_chat_history_no_history():
    mock_supabase = MagicMock()
    mock_table = MagicMock()
    mock_supabase.table.return_value = mock_table
    
    mock_query = MagicMock()
    mock_table.select.return_value = mock_query
    mock_query.order.return_value = mock_query
    mock_query.limit.return_value = mock_query
    
    mock_res = MagicMock()
    mock_res.data = []
    mock_query.execute.return_value = mock_res

    res_str = await search_chat_history(mock_supabase, session_id="sess_123", query="", limit=5)
    parsed = json.loads(res_str)
    
    assert "result" in parsed
    assert "No past chat history found" in parsed["result"]

@pytest.mark.asyncio
async def test_execute_tool_dispatcher():
    mock_supabase = MagicMock()
    mock_table = MagicMock()
    mock_supabase.table.return_value = mock_table
    mock_query = MagicMock()
    mock_table.select.return_value = mock_query
    mock_query.order.return_value = mock_query
    mock_query.limit.return_value = mock_query
    mock_res = MagicMock()
    mock_res.data = []
    mock_query.execute.return_value = mock_res

    res = await execute_tool("search_chat_history", {}, mock_supabase, "sess_1")
    parsed = json.loads(res)
    assert "result" in parsed

@pytest.mark.asyncio
async def test_search_knowledge_base_tool_schema():
    assert SEARCH_KNOWLEDGE_BASE_TOOL["function"]["name"] == "search_knowledge_base"
    assert SEARCH_KNOWLEDGE_BASE_TOOL["function"]["parameters"]["required"] == ["query"]
    assert "Knowledge Base" in SEARCH_KNOWLEDGE_BASE_TOOL["function"]["description"]

@pytest.mark.asyncio
async def test_execute_tool_dispatcher_search_knowledge_base(monkeypatch):
    async def fake_get_relevant_context(supabase, query, user_id, session_id, top_k=3, document_name=None):
        assert query == "SwimAI project"
        assert user_id == "user-123"
        assert session_id == "sess_1"
        assert document_name is None
        return []

    monkeypatch.setattr("api.chat.tools.get_relevant_context", fake_get_relevant_context)

    mock_supabase = MagicMock()
    res = await execute_tool("search_knowledge_base", {"query": "SwimAI project"}, mock_supabase, "sess_1", "user-123")
    parsed = json.loads(res)
    assert "No relevant information found" in parsed["result"]

@pytest.mark.asyncio
async def test_execute_tool_dispatcher_search_knowledge_base_no_user_id():
    mock_supabase = MagicMock()
    res = await execute_tool("search_knowledge_base", {"query": "SwimAI project"}, mock_supabase, "sess_1")
    parsed = json.loads(res)
    assert "error" in parsed
