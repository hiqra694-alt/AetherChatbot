import pytest
import json
import sys
import os
from unittest.mock import MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.chat.tools import search_chat_history, SEARCH_CHAT_HISTORY_TOOL, execute_tool

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
