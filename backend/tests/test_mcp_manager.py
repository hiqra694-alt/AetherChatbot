import json
import sys
import os
from unittest.mock import AsyncMock

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp import types
from mcp_integration.mcp_manager import MCPClientManager, MCPServerConnection, get_merged_tool_schemas, filter_workspace_tools
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
    every connected connector's tools alongside every native tool -- the
    exact Phase 2 behavior, unaffected by Phase 3's filtering option."""
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


class _FakeSettings:
    def __init__(self, google_workspace_mcp_url="", github_mcp_server_url=""):
        self.google_workspace_mcp_url = google_workspace_mcp_url
        self.github_mcp_server_url = github_mcp_server_url


@pytest.mark.asyncio
async def test_create_workspace_session_no_token_yields_none_and_connects_nothing(monkeypatch):
    """No token (the default -- e.g. BRANCH A, or a request that never
    enabled the connector) must not attempt any connection at all, even
    when a Workspace URL is configured."""
    monkeypatch.setattr(
        "mcp_integration.mcp_manager.get_settings",
        lambda: _FakeSettings(google_workspace_mcp_url="http://fake-workspace/sse"),
    )
    connect_called = False

    async def fake_connect(self):
        nonlocal connect_called
        connect_called = True

    monkeypatch.setattr(MCPServerConnection, "connect", fake_connect)

    manager = MCPClientManager(servers=[])
    async with manager.create_workspace_session(None) as session:
        assert session is None

    assert connect_called is False
    assert manager._connections == {}


@pytest.mark.asyncio
async def test_create_workspace_session_no_url_configured_yields_none(monkeypatch):
    monkeypatch.setattr(
        "mcp_integration.mcp_manager.get_settings",
        lambda: _FakeSettings(google_workspace_mcp_url=""),
    )

    manager = MCPClientManager(servers=[])
    async with manager.create_workspace_session("user-oauth-token") as session:
        assert session is None


@pytest.mark.asyncio
async def test_create_workspace_session_success_authenticates_and_closes_on_exit(monkeypatch):
    """A successful session must be opened with the caller's own token as
    a Bearer header, yielded to the caller, and closed again the moment
    the `async with` block exits -- and must never be registered on the
    manager's shared `_connections`, since it's scoped to one user/request."""
    monkeypatch.setattr(
        "mcp_integration.mcp_manager.get_settings",
        lambda: _FakeSettings(google_workspace_mcp_url="http://fake-workspace/sse"),
    )

    connect_calls = []
    close_calls = []

    async def fake_connect(self):
        connect_calls.append((self.url, self.headers))
        self.tools = [types.Tool(name="gmail_search", description="d", inputSchema={"type": "object", "properties": {}})]
        self.session = AsyncMock()

    async def fake_close(self):
        close_calls.append(self.connector_id)
        self.session = None

    monkeypatch.setattr(MCPServerConnection, "connect", fake_connect)
    monkeypatch.setattr(MCPServerConnection, "close", fake_close)

    manager = MCPClientManager(servers=[])
    async with manager.create_workspace_session("user-oauth-token") as session:
        assert session is not None
        assert session.connector_id == "google_workspace"
        assert [t.name for t in session.tools] == ["gmail_search"]
        # Never leaked onto the shared singleton dict, even while open.
        assert manager._connections == {}

    assert connect_calls == [("http://fake-workspace/sse", {"Authorization": "Bearer user-oauth-token"})]
    assert close_calls == ["google_workspace"]
    assert manager._connections == {}


@pytest.mark.asyncio
async def test_create_workspace_session_connect_failure_yields_none_and_still_closes(monkeypatch):
    """A broken/unreachable Workspace server must degrade the turn to 'no
    Workspace tools available' (yield None) rather than crash the chat
    stream, and must still attempt to close whatever partial connection
    state exists."""
    monkeypatch.setattr(
        "mcp_integration.mcp_manager.get_settings",
        lambda: _FakeSettings(google_workspace_mcp_url="http://fake-workspace/sse"),
    )
    close_calls = []

    async def fake_connect(self):
        raise RuntimeError("connection refused")

    async def fake_close(self):
        close_calls.append(self.connector_id)

    monkeypatch.setattr(MCPServerConnection, "connect", fake_connect)
    monkeypatch.setattr(MCPServerConnection, "close", fake_close)

    manager = MCPClientManager(servers=[])
    async with manager.create_workspace_session("bad-token") as session:
        assert session is None

    assert close_calls == ["google_workspace"]


@pytest.mark.asyncio
async def test_create_workspace_session_closes_even_if_caller_raises(monkeypatch):
    """Cleanup must happen even when the code inside the `async with`
    block raises -- e.g. an error partway through the chat turn -- not
    only on the clean-exit path."""
    monkeypatch.setattr(
        "mcp_integration.mcp_manager.get_settings",
        lambda: _FakeSettings(google_workspace_mcp_url="http://fake-workspace/sse"),
    )
    close_calls = []

    async def fake_connect(self):
        self.tools = []
        self.session = AsyncMock()

    async def fake_close(self):
        close_calls.append(self.connector_id)

    monkeypatch.setattr(MCPServerConnection, "connect", fake_connect)
    monkeypatch.setattr(MCPServerConnection, "close", fake_close)

    manager = MCPClientManager(servers=[])
    with pytest.raises(ValueError):
        async with manager.create_workspace_session("user-oauth-token") as session:
            assert session is not None
            raise ValueError("boom mid-turn")

    assert close_calls == ["google_workspace"]


# ==================================================
# create_github_session (Phase 6) -- same contract as create_workspace_session,
# mirrored test-for-test against GITHUB_MCP_SERVER_URL instead.
# ==================================================

@pytest.mark.asyncio
async def test_create_github_session_no_token_yields_none_and_connects_nothing(monkeypatch):
    monkeypatch.setattr(
        "mcp_integration.mcp_manager.get_settings",
        lambda: _FakeSettings(github_mcp_server_url="http://fake-github/sse"),
    )
    connect_called = False

    async def fake_connect(self):
        nonlocal connect_called
        connect_called = True

    monkeypatch.setattr(MCPServerConnection, "connect", fake_connect)

    manager = MCPClientManager(servers=[])
    async with manager.create_github_session(None) as session:
        assert session is None

    assert connect_called is False
    assert manager._connections == {}


@pytest.mark.asyncio
async def test_create_github_session_no_url_configured_yields_none(monkeypatch):
    monkeypatch.setattr(
        "mcp_integration.mcp_manager.get_settings",
        lambda: _FakeSettings(github_mcp_server_url=""),
    )

    manager = MCPClientManager(servers=[])
    async with manager.create_github_session("user-oauth-token") as session:
        assert session is None


@pytest.mark.asyncio
async def test_create_github_session_success_authenticates_and_closes_on_exit(monkeypatch):
    """A successful per-user GitHub session must be opened with the
    caller's own token as a Bearer header, yielded to the caller, and
    closed again the moment the `async with` block exits -- and must never
    be registered on the manager's shared `_connections`, since it's
    scoped to one user/request (independent of any static, shared-PAT
    GitHub connector that might also be connected)."""
    monkeypatch.setattr(
        "mcp_integration.mcp_manager.get_settings",
        lambda: _FakeSettings(github_mcp_server_url="http://fake-github/sse"),
    )

    connect_calls = []
    close_calls = []

    async def fake_connect(self):
        connect_calls.append((self.url, self.headers))
        self.tools = [types.Tool(name="create_issue", description="d", inputSchema={"type": "object", "properties": {}})]
        self.session = AsyncMock()

    async def fake_close(self):
        close_calls.append(self.connector_id)
        self.session = None

    monkeypatch.setattr(MCPServerConnection, "connect", fake_connect)
    monkeypatch.setattr(MCPServerConnection, "close", fake_close)

    manager = MCPClientManager(servers=[])
    async with manager.create_github_session("user-oauth-token") as session:
        assert session is not None
        assert session.connector_id == "github"
        assert [t.name for t in session.tools] == ["create_issue"]
        # Never leaked onto the shared singleton dict, even while open.
        assert manager._connections == {}

    assert connect_calls == [("http://fake-github/sse", {"Authorization": "Bearer user-oauth-token"})]
    assert close_calls == ["github"]
    assert manager._connections == {}


@pytest.mark.asyncio
async def test_create_github_session_connect_failure_yields_none_and_still_closes(monkeypatch):
    monkeypatch.setattr(
        "mcp_integration.mcp_manager.get_settings",
        lambda: _FakeSettings(github_mcp_server_url="http://fake-github/sse"),
    )
    close_calls = []

    async def fake_connect(self):
        raise RuntimeError("connection refused")

    async def fake_close(self):
        close_calls.append(self.connector_id)

    monkeypatch.setattr(MCPServerConnection, "connect", fake_connect)
    monkeypatch.setattr(MCPServerConnection, "close", fake_close)

    manager = MCPClientManager(servers=[])
    async with manager.create_github_session("bad-token") as session:
        assert session is None

    assert close_calls == ["github"]


@pytest.mark.asyncio
async def test_create_github_session_closes_even_if_caller_raises(monkeypatch):
    monkeypatch.setattr(
        "mcp_integration.mcp_manager.get_settings",
        lambda: _FakeSettings(github_mcp_server_url="http://fake-github/sse"),
    )
    close_calls = []

    async def fake_connect(self):
        self.tools = []
        self.session = AsyncMock()

    async def fake_close(self):
        close_calls.append(self.connector_id)

    monkeypatch.setattr(MCPServerConnection, "connect", fake_connect)
    monkeypatch.setattr(MCPServerConnection, "close", fake_close)

    manager = MCPClientManager(servers=[])
    with pytest.raises(ValueError):
        async with manager.create_github_session("user-oauth-token") as session:
            assert session is not None
            raise ValueError("boom mid-turn")

    assert close_calls == ["github"]


# ==================================================
# filter_workspace_tools (Phase 7) -- narrows a Workspace session's tools to
# whichever granular Gmail/Docs/Calendar/Drive sub-connector(s) are enabled.
# ==================================================

def _workspace_tool(name: str) -> types.Tool:
    return types.Tool(name=name, description="d", inputSchema={"type": "object", "properties": {}})


_ALL_WORKSPACE_TOOLS = [
    _workspace_tool("gmail_search"),
    _workspace_tool("docs_read"),
    _workspace_tool("calendar_list_events"),
    _workspace_tool("drive_list_files"),
    _workspace_tool("gdrive_upload"),
]


def test_filter_workspace_tools_none_offers_everything():
    """No enabled_connectors sent at all (an old caller, or a request that
    never sends the field) must offer every discovered tool, unfiltered --
    matches get_merged_tool_schemas' own None contract for static connectors."""
    result = filter_workspace_tools(_ALL_WORKSPACE_TOOLS, None)
    assert [t.name for t in result] == [t.name for t in _ALL_WORKSPACE_TOOLS]


def test_filter_workspace_tools_legacy_id_offers_everything():
    """The pre-Phase-7 single "google_workspace" toggle must still offer
    every tool, for backward compatibility with any caller still sending it."""
    result = filter_workspace_tools(_ALL_WORKSPACE_TOOLS, ["google_workspace"])
    assert [t.name for t in result] == [t.name for t in _ALL_WORKSPACE_TOOLS]


def test_filter_workspace_tools_single_sub_connector():
    result = filter_workspace_tools(_ALL_WORKSPACE_TOOLS, ["google_gmail"])
    assert [t.name for t in result] == ["gmail_search"]


def test_filter_workspace_tools_multiple_sub_connectors():
    result = filter_workspace_tools(_ALL_WORKSPACE_TOOLS, ["google_docs", "google_calendar"])
    assert {t.name for t in result} == {"docs_read", "calendar_list_events"}


def test_filter_workspace_tools_drive_matches_both_prefixes():
    """Drive tools may be prefixed either drive_ or gdrive_ depending on the
    MCP server's own naming -- both must match."""
    result = filter_workspace_tools(_ALL_WORKSPACE_TOOLS, ["google_drive"])
    assert {t.name for t in result} == {"drive_list_files", "gdrive_upload"}


def test_filter_workspace_tools_empty_list_offers_nothing():
    assert filter_workspace_tools(_ALL_WORKSPACE_TOOLS, []) == []


def test_filter_workspace_tools_unrelated_ids_only_offers_nothing():
    """enabled_connectors present but containing only non-Google ids (e.g.
    just "github" toggled on) must offer no Workspace tools at all."""
    assert filter_workspace_tools(_ALL_WORKSPACE_TOOLS, ["github"]) == []
