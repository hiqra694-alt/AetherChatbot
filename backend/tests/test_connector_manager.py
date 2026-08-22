import contextlib
import json
import sys
import os
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp import types
from connector_integrations.connector_manager import (
    MCPClientManager,
    MCPServerConnection,
    get_merged_tool_schemas,
)
from api.chat.tools import ALL_TOOLS, ALL_TOOL_NAMES


def _fake_connection(connector_id: str, tool_name: str, session: AsyncMock = None) -> MCPServerConnection:
    """A connected-looking MCPServerConnection with one cached tool, built
    directly (no real network I/O) so schema-merging/filtering and
    call_tool can be exercised without a live MCP server."""
    connection = MCPServerConnection(connector_id, f"http://fake/{connector_id}/sse")
    connection.tools = [types.Tool(
        name=tool_name,
        description=f"Fake tool for connector {connector_id}.",
        inputSchema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    )]
    connection.session = session or AsyncMock()
    return connection


def _manager_with_two_connectors():
    manager = MCPClientManager(servers=[])
    manager._connections["some_tool"] = _fake_connection("some_tool", "some_tool_action")
    manager._connections["other_connector"] = _fake_connection("other_connector", "other_connector_action")
    return manager


def test_get_merged_tool_schemas_includes_all_connectors_by_default():
    """Backward compatibility: enabled_connectors omitted (None) offers
    every connected connector's tools alongside every native tool."""
    manager = _manager_with_two_connectors()

    merged = get_merged_tool_schemas(ALL_TOOLS, manager)
    names = {t["function"]["name"] for t in merged}

    assert set(ALL_TOOL_NAMES).issubset(names)
    assert "some_tool_action" in names
    assert "other_connector_action" in names


def test_get_merged_tool_schemas_filters_by_enabled_connectors():
    """enabled_connectors=["some_tool"] must only offer that connector's
    tools -- other connected connectors' tools are excluded, while every
    native tool remains present regardless."""
    manager = _manager_with_two_connectors()

    merged = get_merged_tool_schemas(ALL_TOOLS, manager, enabled_connectors=["some_tool"])
    names = {t["function"]["name"] for t in merged}

    assert set(ALL_TOOL_NAMES).issubset(names)
    assert "some_tool_action" in names
    assert "other_connector_action" not in names


def test_get_merged_tool_schemas_empty_enabled_connectors_keeps_native_tools_only():
    """An explicit empty list (every connector toggled off) must still
    leave every native tool available -- only MCP tools are ever gated by
    enabled_connectors."""
    manager = _manager_with_two_connectors()

    merged = get_merged_tool_schemas(ALL_TOOLS, manager, enabled_connectors=[])
    names = {t["function"]["name"] for t in merged}

    assert names == set(ALL_TOOL_NAMES)


@pytest.mark.asyncio
async def test_call_tool_extracts_text_result():
    manager = MCPClientManager(servers=[])
    session = AsyncMock()
    session.call_tool.return_value = types.CallToolResult(
        content=[types.TextContent(type="text", text="42 results found.")],
        isError=False,
    )
    manager._connections["some_tool"] = _fake_connection("some_tool", "some_tool_action", session=session)

    result = await manager.call_tool("some_tool_action", {"query": "hello"})

    assert result == "42 results found."
    session.call_tool.assert_awaited_once_with("some_tool_action", {"query": "hello"})


@pytest.mark.asyncio
async def test_call_tool_unknown_name_degrades_gracefully():
    manager = MCPClientManager(servers=[])

    result = await manager.call_tool("does_not_exist", {})

    parsed = json.loads(result)
    assert "error" in parsed


@pytest.mark.asyncio
async def test_call_tool_error_result_surfaces_as_error_json():
    manager = MCPClientManager(servers=[])
    session = AsyncMock()
    session.call_tool.return_value = types.CallToolResult(
        content=[types.TextContent(type="text", text="rate limited")],
        isError=True,
    )
    manager._connections["some_tool"] = _fake_connection("some_tool", "some_tool_action", session=session)

    result = await manager.call_tool("some_tool_action", {"query": "hello"})

    parsed = json.loads(result)
    assert parsed["error"] == "rate limited"


@pytest.mark.asyncio
async def test_connect_uses_sse_client(monkeypatch):
    """MCPServerConnection.connect() dials every static/generic connector
    over SSE."""
    calls = []

    @contextlib.asynccontextmanager
    async def fake_sse_client(url, headers=None):
        calls.append(("sse", url, headers))
        yield (AsyncMock(), AsyncMock())

    monkeypatch.setattr("connector_integrations.connector_manager.sse_client", fake_sse_client)

    fake_session = AsyncMock()
    fake_session.list_tools = AsyncMock(return_value=MagicMock(tools=[]))

    class _FakeSessionCtx:
        async def __aenter__(self):
            return fake_session
        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr("connector_integrations.connector_manager.ClientSession", lambda r, w: _FakeSessionCtx())

    connection = MCPServerConnection("some_id", "http://fake/sse", headers={"Authorization": "Bearer t"})
    await connection.connect()

    assert calls == [("sse", "http://fake/sse", {"Authorization": "Bearer t"})]
    fake_session.initialize.assert_awaited_once()
