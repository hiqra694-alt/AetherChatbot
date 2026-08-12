import contextlib
import json
import sys
import os
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp import types
from mcp_integration.mcp_manager import (
    MCPClientManager,
    MCPServerConnection,
    get_merged_tool_schemas,
    filter_workspace_tools,
    _refresh_google_access_token,
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
    def __init__(
        self,
        google_workspace_mcp_url="",
        github_mcp_server_url="",
        google_client_id="",
        google_client_secret="",
        gmail_mcp_client_id="",
        gmail_mcp_client_secret="",
        gmail_mcp_redirect_uri="",
    ):
        self.google_workspace_mcp_url = google_workspace_mcp_url
        self.github_mcp_server_url = github_mcp_server_url
        self.google_client_id = google_client_id
        self.google_client_secret = google_client_secret
        self.gmail_mcp_client_id = gmail_mcp_client_id
        self.gmail_mcp_client_secret = gmail_mcp_client_secret
        self.gmail_mcp_redirect_uri = gmail_mcp_redirect_uri


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
# _refresh_google_access_token / create_workspace_session's refresh-token
# fallback -- server-side substitute for a live access token when the
# request didn't carry one (see mcp_manager.py's docstrings for why).
# ==================================================

def _mock_oauth_supabase(rows=None, raise_exc=None):
    table = MagicMock()
    for method in ("select", "eq", "limit"):
        getattr(table, method).return_value = table
    if raise_exc is not None:
        table.execute.side_effect = raise_exc
    else:
        table.execute.return_value = MagicMock(data=rows if rows is not None else [])
    supabase = MagicMock()
    supabase.table.return_value = table
    return supabase, table


class _FakeHttpResponse:
    def __init__(self, json_data, raise_error=False):
        self._json_data = json_data
        self._raise_error = raise_error

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self._raise_error:
            raise Exception("400 Bad Request: invalid_grant")


def _fake_async_client(response_or_factory):
    """A minimal stand-in for httpx.AsyncClient's `async with` usage,
    returning `response_or_factory` (or calling it, if callable) from
    `.post`, with no real network I/O."""
    class _FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, data=None, timeout=None):
            self.last_call = (url, data)
            return response_or_factory() if callable(response_or_factory) else response_or_factory

    return _FakeAsyncClient


@pytest.mark.asyncio
async def test_refresh_google_access_token_no_client_credentials_skips_lookup(monkeypatch):
    """No GOOGLE_CLIENT_ID/SECRET configured must skip the DB lookup
    entirely and return None -- the fallback is simply unavailable."""
    monkeypatch.setattr("mcp_integration.mcp_manager.get_settings", lambda: _FakeSettings())
    supabase = MagicMock()

    result = await _refresh_google_access_token(supabase, "user-123")

    assert result is None
    supabase.table.assert_not_called()


@pytest.mark.asyncio
async def test_refresh_google_access_token_no_stored_token_returns_none(monkeypatch):
    monkeypatch.setattr(
        "mcp_integration.mcp_manager.get_settings",
        lambda: _FakeSettings(google_client_id="id", google_client_secret="secret"),
    )
    supabase, table = _mock_oauth_supabase(rows=[])

    result = await _refresh_google_access_token(supabase, "user-123")

    assert result is None
    table.eq.assert_any_call("user_id", "user-123")
    table.eq.assert_any_call("provider", "google")


@pytest.mark.asyncio
async def test_refresh_google_access_token_db_error_returns_none(monkeypatch):
    monkeypatch.setattr(
        "mcp_integration.mcp_manager.get_settings",
        lambda: _FakeSettings(google_client_id="id", google_client_secret="secret"),
    )
    supabase, _ = _mock_oauth_supabase(raise_exc=RuntimeError("db down"))

    result = await _refresh_google_access_token(supabase, "user-123")

    assert result is None


@pytest.mark.asyncio
async def test_refresh_google_access_token_success_posts_to_google_and_returns_access_token(monkeypatch):
    monkeypatch.setattr(
        "mcp_integration.mcp_manager.get_settings",
        lambda: _FakeSettings(google_client_id="test-client-id", google_client_secret="test-client-secret"),
    )
    supabase, _ = _mock_oauth_supabase(rows=[{"refresh_token": "stored-refresh-token"}])

    fake_client_cls = _fake_async_client(_FakeHttpResponse({"access_token": "fresh-access-token"}))
    monkeypatch.setattr("mcp_integration.mcp_manager.httpx.AsyncClient", fake_client_cls)

    result = await _refresh_google_access_token(supabase, "user-123")

    assert result == "fresh-access-token"


@pytest.mark.asyncio
async def test_refresh_google_access_token_sends_correct_grant_params(monkeypatch):
    monkeypatch.setattr(
        "mcp_integration.mcp_manager.get_settings",
        lambda: _FakeSettings(google_client_id="test-client-id", google_client_secret="test-client-secret"),
    )
    supabase, _ = _mock_oauth_supabase(rows=[{"refresh_token": "stored-refresh-token"}])

    captured = {}

    class _CapturingAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, data=None, timeout=None):
            captured["url"] = url
            captured["data"] = data
            return _FakeHttpResponse({"access_token": "fresh-access-token"})

    monkeypatch.setattr("mcp_integration.mcp_manager.httpx.AsyncClient", _CapturingAsyncClient)

    await _refresh_google_access_token(supabase, "user-123")

    assert captured["url"] == "https://oauth2.googleapis.com/token"
    assert captured["data"] == {
        "client_id": "test-client-id",
        "client_secret": "test-client-secret",
        "refresh_token": "stored-refresh-token",
        "grant_type": "refresh_token",
    }


@pytest.mark.asyncio
async def test_refresh_google_access_token_google_error_returns_none(monkeypatch):
    """A revoked/expired refresh token (Google returns an error status) must
    degrade to None, logged, never raised."""
    monkeypatch.setattr(
        "mcp_integration.mcp_manager.get_settings",
        lambda: _FakeSettings(google_client_id="id", google_client_secret="secret"),
    )
    supabase, _ = _mock_oauth_supabase(rows=[{"refresh_token": "revoked-token"}])

    fake_client_cls = _fake_async_client(_FakeHttpResponse({}, raise_error=True))
    monkeypatch.setattr("mcp_integration.mcp_manager.httpx.AsyncClient", fake_client_cls)

    result = await _refresh_google_access_token(supabase, "user-123")

    assert result is None


@pytest.mark.asyncio
async def test_create_workspace_session_falls_back_to_refreshed_token_when_no_live_token(monkeypatch):
    """No live access token, but supabase+user_id given: the session must
    open using a token minted via _refresh_google_access_token instead of
    silently yielding None."""
    monkeypatch.setattr(
        "mcp_integration.mcp_manager.get_settings",
        lambda: _FakeSettings(google_workspace_mcp_url="http://fake-workspace/sse"),
    )

    async def fake_refresh(supabase, user_id):
        assert user_id == "user-123"
        return "refreshed-access-token"

    monkeypatch.setattr("mcp_integration.mcp_manager._refresh_google_access_token", fake_refresh)

    connect_calls = []

    async def fake_connect(self):
        connect_calls.append((self.url, self.headers))
        self.session = AsyncMock()

    async def fake_close(self):
        self.session = None

    monkeypatch.setattr(MCPServerConnection, "connect", fake_connect)
    monkeypatch.setattr(MCPServerConnection, "close", fake_close)

    manager = MCPClientManager(servers=[])
    supabase = MagicMock()
    async with manager.create_workspace_session(None, supabase=supabase, user_id="user-123") as session:
        assert session is not None

    assert connect_calls == [("http://fake-workspace/sse", {"Authorization": "Bearer refreshed-access-token"})]


@pytest.mark.asyncio
async def test_create_workspace_session_live_token_skips_refresh_fallback(monkeypatch):
    """A live token takes precedence -- the refresh-token fallback must
    never even be consulted when one is already present."""
    monkeypatch.setattr(
        "mcp_integration.mcp_manager.get_settings",
        lambda: _FakeSettings(google_workspace_mcp_url="http://fake-workspace/sse"),
    )
    refresh_called = False

    async def fake_refresh(supabase, user_id):
        nonlocal refresh_called
        refresh_called = True
        return "should-not-be-used"

    monkeypatch.setattr("mcp_integration.mcp_manager._refresh_google_access_token", fake_refresh)

    async def fake_connect(self):
        self.session = AsyncMock()

    async def fake_close(self):
        self.session = None

    monkeypatch.setattr(MCPServerConnection, "connect", fake_connect)
    monkeypatch.setattr(MCPServerConnection, "close", fake_close)

    manager = MCPClientManager(servers=[])
    supabase = MagicMock()
    async with manager.create_workspace_session("live-token", supabase=supabase, user_id="user-123") as session:
        assert session is not None

    assert refresh_called is False


@pytest.mark.asyncio
async def test_create_workspace_session_no_supabase_or_user_id_skips_refresh_fallback(monkeypatch):
    """Omitting supabase/user_id (the default -- e.g. BRANCH A, or a turn
    where Google wasn't requested this turn, see ChatService.stream_chat)
    must keep the exact prior no-token behavior: no fallback attempted, no
    session opened."""
    monkeypatch.setattr(
        "mcp_integration.mcp_manager.get_settings",
        lambda: _FakeSettings(google_workspace_mcp_url="http://fake-workspace/sse"),
    )
    refresh_called = False

    async def fake_refresh(supabase, user_id):
        nonlocal refresh_called
        refresh_called = True
        return "token"

    monkeypatch.setattr("mcp_integration.mcp_manager._refresh_google_access_token", fake_refresh)

    manager = MCPClientManager(servers=[])
    async with manager.create_workspace_session(None) as session:
        assert session is None

    assert refresh_called is False


@pytest.mark.asyncio
async def test_create_workspace_session_refresh_fallback_failure_yields_none(monkeypatch):
    """The refresh helper itself returning None (no stored token, or Google
    rejected it) must degrade to no Workspace session, same as never having
    sent a token at all."""
    monkeypatch.setattr(
        "mcp_integration.mcp_manager.get_settings",
        lambda: _FakeSettings(google_workspace_mcp_url="http://fake-workspace/sse"),
    )

    async def fake_refresh(supabase, user_id):
        return None

    monkeypatch.setattr("mcp_integration.mcp_manager._refresh_google_access_token", fake_refresh)

    manager = MCPClientManager(servers=[])
    supabase = MagicMock()
    async with manager.create_workspace_session(None, supabase=supabase, user_id="user-123") as session:
        assert session is None


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


# ==================================================
# MCPServerConnection transport branching -- "sse" (default, every
# pre-existing connector) vs "streamable_http" (Gmail MCP only).
# ==================================================

@pytest.mark.asyncio
async def test_connect_default_transport_uses_sse_client(monkeypatch):
    """Omitting `transport` must preserve the exact prior sse_client-only
    behavior -- zero regression for every connector that predates the
    Gmail MCP connector."""
    calls = []

    @contextlib.asynccontextmanager
    async def fake_sse_client(url, headers=None):
        calls.append(("sse", url, headers))
        yield (AsyncMock(), AsyncMock())

    monkeypatch.setattr("mcp_integration.mcp_manager.sse_client", fake_sse_client)

    fake_session = AsyncMock()
    fake_session.list_tools = AsyncMock(return_value=MagicMock(tools=[]))

    class _FakeSessionCtx:
        async def __aenter__(self):
            return fake_session
        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr("mcp_integration.mcp_manager.ClientSession", lambda r, w: _FakeSessionCtx())

    connection = MCPServerConnection("some_id", "http://fake/sse", headers={"Authorization": "Bearer t"})
    await connection.connect()

    assert calls == [("sse", "http://fake/sse", {"Authorization": "Bearer t"})]
    fake_session.initialize.assert_awaited_once()


@pytest.mark.asyncio
async def test_connect_streamable_http_transport_uses_streamable_http_client(monkeypatch):
    """transport="streamable_http" (only ever set by
    create_gmail_mcp_session) must dial via mcp.client.streamable_http
    instead of sse_client -- Google's managed Gmail MCP server speaks
    Streamable HTTP, not SSE."""
    calls = []

    class _FakeHttpClient:
        def __init__(self, headers=None, timeout=None):
            self.headers = headers
            self.timeout = timeout
        async def __aenter__(self):
            return self
        async def __aexit__(self, *exc):
            return False

    @contextlib.asynccontextmanager
    async def fake_streamable_http_client(url, http_client=None):
        calls.append(("streamable_http", url, http_client.headers))
        yield (AsyncMock(), AsyncMock())

    monkeypatch.setattr("mcp_integration.mcp_manager.httpx.AsyncClient", _FakeHttpClient)
    monkeypatch.setattr("mcp_integration.mcp_manager.streamable_http_client", fake_streamable_http_client)

    fake_session = AsyncMock()
    fake_session.list_tools = AsyncMock(return_value=MagicMock(tools=[]))

    class _FakeSessionCtx:
        async def __aenter__(self):
            return fake_session
        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr("mcp_integration.mcp_manager.ClientSession", lambda r, w: _FakeSessionCtx())

    connection = MCPServerConnection(
        "google_gmail_mcp", "https://gmailmcp.googleapis.com/mcp/v1",
        headers={"Authorization": "Bearer gmail-token"}, transport="streamable_http",
    )
    await connection.connect()

    assert calls == [
        ("streamable_http", "https://gmailmcp.googleapis.com/mcp/v1", {"Authorization": "Bearer gmail-token"})
    ]
    fake_session.initialize.assert_awaited_once()


# ==================================================
# create_gmail_mcp_session -- isolated Gmail MCP connector (Phase 1). Same
# yield/never-raise contract as create_workspace_session/create_github_session,
# but authenticated via its own GMAIL_MCP_* settings and its own
# 'google_gmail_mcp' provider row (mcp_integration.gmail_mcp), never the
# plain 'google' row create_workspace_session's refresh fallback reads.
# ==================================================

@pytest.mark.asyncio
async def test_create_gmail_mcp_session_no_supabase_or_user_id_yields_none():
    manager = MCPClientManager(servers=[])
    async with manager.create_gmail_mcp_session() as session:
        assert session is None
    assert manager._connections == {}


@pytest.mark.asyncio
async def test_create_gmail_mcp_session_not_configured_yields_none(monkeypatch):
    monkeypatch.setattr("mcp_integration.mcp_manager.get_settings", lambda: _FakeSettings())

    manager = MCPClientManager(servers=[])
    supabase = MagicMock()
    async with manager.create_gmail_mcp_session(supabase=supabase, user_id="user-123") as session:
        assert session is None


@pytest.mark.asyncio
async def test_create_gmail_mcp_session_no_stored_refresh_token_yields_none(monkeypatch):
    monkeypatch.setattr(
        "mcp_integration.mcp_manager.get_settings",
        lambda: _FakeSettings(
            gmail_mcp_client_id="id", gmail_mcp_client_secret="secret", gmail_mcp_redirect_uri="uri",
        ),
    )

    async def fake_refresh(supabase, user_id):
        return None

    monkeypatch.setattr("mcp_integration.mcp_manager.refresh_gmail_mcp_access_token", fake_refresh)

    manager = MCPClientManager(servers=[])
    supabase = MagicMock()
    async with manager.create_gmail_mcp_session(supabase=supabase, user_id="user-123") as session:
        assert session is None


@pytest.mark.asyncio
async def test_create_gmail_mcp_session_success_uses_streamable_http_and_closes_on_exit(monkeypatch):
    monkeypatch.setattr(
        "mcp_integration.mcp_manager.get_settings",
        lambda: _FakeSettings(
            gmail_mcp_client_id="id", gmail_mcp_client_secret="secret", gmail_mcp_redirect_uri="uri",
        ),
    )

    async def fake_refresh(supabase, user_id):
        assert user_id == "user-123"
        return "fresh-gmail-access-token"

    monkeypatch.setattr("mcp_integration.mcp_manager.refresh_gmail_mcp_access_token", fake_refresh)

    connect_calls = []
    close_calls = []

    async def fake_connect(self):
        connect_calls.append((self.url, self.headers, self.transport))
        self.tools = [types.Tool(name="gmail_send", description="d", inputSchema={"type": "object", "properties": {}})]
        self.session = AsyncMock()

    async def fake_close(self):
        close_calls.append(self.connector_id)
        self.session = None

    monkeypatch.setattr(MCPServerConnection, "connect", fake_connect)
    monkeypatch.setattr(MCPServerConnection, "close", fake_close)

    manager = MCPClientManager(servers=[])
    supabase = MagicMock()
    async with manager.create_gmail_mcp_session(supabase=supabase, user_id="user-123") as session:
        assert session is not None
        assert session.connector_id == "google_gmail_mcp"
        assert [t.name for t in session.tools] == ["gmail_send"]
        # Never leaked onto the shared singleton dict, same isolation
        # guarantee as create_workspace_session/create_github_session.
        assert manager._connections == {}

    assert connect_calls == [
        ("https://gmailmcp.googleapis.com/mcp/v1", {"Authorization": "Bearer fresh-gmail-access-token"}, "streamable_http")
    ]
    assert close_calls == ["google_gmail_mcp"]
    assert manager._connections == {}


@pytest.mark.asyncio
async def test_create_gmail_mcp_session_connect_failure_yields_none_and_still_closes(monkeypatch):
    monkeypatch.setattr(
        "mcp_integration.mcp_manager.get_settings",
        lambda: _FakeSettings(
            gmail_mcp_client_id="id", gmail_mcp_client_secret="secret", gmail_mcp_redirect_uri="uri",
        ),
    )

    async def fake_refresh(supabase, user_id):
        return "token"

    monkeypatch.setattr("mcp_integration.mcp_manager.refresh_gmail_mcp_access_token", fake_refresh)

    close_calls = []

    async def fake_connect(self):
        raise RuntimeError("connection refused")

    async def fake_close(self):
        close_calls.append(self.connector_id)

    monkeypatch.setattr(MCPServerConnection, "connect", fake_connect)
    monkeypatch.setattr(MCPServerConnection, "close", fake_close)

    manager = MCPClientManager(servers=[])
    supabase = MagicMock()
    async with manager.create_gmail_mcp_session(supabase=supabase, user_id="user-123") as session:
        assert session is None

    assert close_calls == ["google_gmail_mcp"]


@pytest.mark.asyncio
async def test_create_gmail_mcp_session_uses_isolated_refresh_helper_not_google_workspace_one(monkeypatch):
    """Confirms create_gmail_mcp_session calls the Gmail-MCP-specific
    refresh helper (reading the 'google_gmail_mcp' provider row) rather
    than _refresh_google_access_token (the 'google' provider row
    create_workspace_session uses) -- the two connectors' credentials must
    never cross over."""
    monkeypatch.setattr(
        "mcp_integration.mcp_manager.get_settings",
        lambda: _FakeSettings(
            gmail_mcp_client_id="id", gmail_mcp_client_secret="secret", gmail_mcp_redirect_uri="uri",
        ),
    )

    workspace_refresh_called = False

    async def fake_workspace_refresh(supabase, user_id):
        nonlocal workspace_refresh_called
        workspace_refresh_called = True
        return "should-not-be-used"

    async def fake_gmail_refresh(supabase, user_id):
        return "gmail-specific-token"

    monkeypatch.setattr("mcp_integration.mcp_manager._refresh_google_access_token", fake_workspace_refresh)
    monkeypatch.setattr("mcp_integration.mcp_manager.refresh_gmail_mcp_access_token", fake_gmail_refresh)

    connect_calls = []

    async def fake_connect(self):
        connect_calls.append(self.headers)
        self.session = AsyncMock()

    async def fake_close(self):
        self.session = None

    monkeypatch.setattr(MCPServerConnection, "connect", fake_connect)
    monkeypatch.setattr(MCPServerConnection, "close", fake_close)

    manager = MCPClientManager(servers=[])
    supabase = MagicMock()
    async with manager.create_gmail_mcp_session(supabase=supabase, user_id="user-123") as session:
        assert session is not None

    assert connect_calls == [{"Authorization": "Bearer gmail-specific-token"}]
    assert workspace_refresh_called is False
