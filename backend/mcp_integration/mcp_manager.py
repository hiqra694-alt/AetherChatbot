"""
MCP Client Manager.

Owns connections to zero or more *external* MCP servers over the HTTP/SSE
transport, discovers their tool schemas, and merges those schemas alongside
AetherChat's own native tool registry (api.chat.tools.ALL_TOOLS) into one
list shaped for the LLM -- optionally filtered down to a per-request set of
enabled connectors (Phase 3).

Two connector lifetimes coexist here:

- Static connectors (Brave Search, a shared-PAT GitHub, ...): connected once
  at app startup, held in the MCPClientManager singleton's `_connections`
  for the process's lifetime, and shared read-only across every request.
- Per-request, per-user connectors -- Google Workspace (Phase 4) via
  `create_workspace_session`, and GitHub (Phase 6) via
  `create_github_session` -- opened fresh per request, authenticated with
  that one user's own OAuth token, and closed again before the request
  finishes. Never touch `_connections` and are never shared across
  requests -- see create_workspace_session's docstring for why that
  isolation matters; create_github_session follows the identical contract.
  GitHub can therefore run in either or both modes at once: a shared PAT
  (GITHUB_PERSONAL_ACCESS_TOKEN, static, process-wide) and/or per-user
  OAuth tokens (per-request) -- they're independent connector instances
  that happen to point at the same connector_id/tool namespace.

Strictly additive: nothing in this module mutates the existing Groq
provider, RAG pipeline, native tools, or Supabase wiring. Each external
server is identified by a stable `connector_id` (e.g. "github",
"brave_search") rather than its raw URL, so the frontend can toggle
connectors on/off by a human-readable id and a server's tools can be traced
back to the connector that offers them.
"""

import asyncio
import contextlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterable, Optional

import httpx
from mcp import types
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamable_http_client
from supabase import Client

from core.config import get_settings
from mcp_integration.gmail_mcp import GMAIL_MCP_SERVER_URL, refresh_gmail_mcp_access_token

logger = logging.getLogger(__name__)

# Ceiling on how long any single MCP server's connect()+initialize()+
# list_tools() handshake may take, applied via asyncio.wait_for everywhere
# a connection is opened. Without this, a server that accepts the TCP/SSE
# connection but never completes (or never responds to) the MCP handshake
# -- e.g. a misconfigured URL pointing at a port nothing is actually
# serving MCP on -- hangs that await forever: at startup that means the
# whole FastAPI app never finishes starting, and mid-request it means the
# user's HTTP request never completes either. A plain connection refused/
# DNS failure already fails fast on its own; this constant exists for the
# slower, hung-not-failed case those don't cover.
CONNECT_TIMEOUT_SECONDS = 10.0

# Google's OAuth token endpoint, used only by _refresh_google_access_token
# below to exchange a stored refresh_token for a fresh access token.
GOOGLE_TOKEN_REFRESH_URL = "https://oauth2.googleapis.com/token"


async def _refresh_google_access_token(supabase: Client, user_id: str) -> Optional[str]:
    """
    Server-side fallback for when a chat request doesn't carry a live Google
    OAuth access token (see ChatService.stream_chat) -- e.g. the token the
    frontend cached in memory right after linkIdentity has aged out of the
    browser session on reload, a known limitation of Supabase's own
    provider_token handling (never persisted past the initial OAuth
    exchange, see src/app/page.tsx). Looks up this user's long-lived
    refresh_token -- stored via POST /api/connectors/store-token at link
    time, see api/connectors/routers.py -- and exchanges it for a fresh,
    short-lived access token directly against Google's own token endpoint,
    using this deployment's GOOGLE_CLIENT_ID/GOOGLE_CLIENT_SECRET.

    Returns None -- logged, never raised -- on every failure mode: no
    client id/secret configured, no stored refresh token for this user, a
    DB error, or Google rejecting/revoking the refresh token (e.g. the user
    revoked access from their Google Account settings). The caller
    (create_workspace_session) treats a None return exactly like "no token
    was ever sent" and simply proceeds without a Workspace session this
    turn, rather than ever raising out of a chat request.
    """
    settings = get_settings()
    if not settings.google_client_id or not settings.google_client_secret:
        # Previously a bare `return None` -- this is a config gap, not an
        # expected "user just hasn't linked anything" case, and was
        # indistinguishable from every other silent-None branch below. Now
        # logged at ERROR so a misconfigured deployment shows up in logs
        # instead of just quietly never offering Workspace tools to anyone.
        logger.error(
            "MCP: Google refresh-token fallback skipped for user %s -- "
            "GOOGLE_CLIENT_ID/GOOGLE_CLIENT_SECRET not configured.",
            user_id,
        )
        return None

    try:
        res = (
            supabase.table("user_oauth_tokens")
            .select("refresh_token")
            .eq("user_id", user_id)
            .eq("provider", "google")
            .limit(1)
            .execute()
        )
    except Exception:
        logger.error(
            "MCP: failed to look up stored Google refresh token for user %s.", user_id, exc_info=True
        )
        return None

    rows = res.data or []
    refresh_token = rows[0].get("refresh_token") if rows else None
    if not refresh_token:
        logger.error(
            "MCP: no stored Google refresh_token found for user %s -- "
            "user_oauth_tokens has no row for (user_id=%s, provider=google). "
            "POST /api/connectors/store-token returning 200 only proves the "
            "row was written for *some* user/provider at *some* point -- it "
            "does not prove this row, for this user, still exists now.",
            user_id, user_id,
        )
        return None

    try:
        async with httpx.AsyncClient() as http_client:
            response = await http_client.post(
                GOOGLE_TOKEN_REFRESH_URL,
                data={
                    "client_id": settings.google_client_id,
                    "client_secret": settings.google_client_secret,
                    "refresh_token": refresh_token,
                    "grant_type": "refresh_token",
                },
                timeout=CONNECT_TIMEOUT_SECONDS,
            )
        response.raise_for_status()
    except httpx.HTTPStatusError as http_err:
        # The exact reason Google rejected this refresh_token -- most
        # commonly a 400 invalid_grant (token expired/revoked, e.g. the user
        # revoked access from their Google Account settings, or this
        # client_id/secret doesn't match the one the token was minted
        # under). Logged explicitly (status + body) instead of via a
        # generic `except Exception`, which discarded both entirely.
        logger.error(
            "MCP: Google rejected the refresh_token for user %s -- status=%s body=%s",
            user_id, http_err.response.status_code, http_err.response.text,
        )
        return None
    except Exception:
        # Network-level failure (DNS, timeout, connection refused) -- there
        # is no HTTP response to inspect at all here, unlike the
        # HTTPStatusError branch above.
        logger.error(
            "MCP: network error calling Google's token endpoint for user %s.", user_id, exc_info=True
        )
        return None

    access_token = response.json().get("access_token")
    if not access_token:
        logger.error(
            "MCP: Google's token endpoint returned 200 with no access_token for user %s -- body=%s",
            user_id, response.text,
        )
        return None

    return access_token


@dataclass
class MCPServerSpec:
    """One external MCP server to connect to: a stable id the frontend/
    config can refer to it by, its SSE endpoint, and any headers the
    connection needs (e.g. a GitHub PAT as a Bearer token)."""
    connector_id: str
    url: str
    headers: Optional[dict[str, str]] = field(default=None)


async def _call_tool_on_session(session: ClientSession, tool_name: str, arguments: dict) -> str:
    """
    Shared execution + CallToolResult-parsing logic used by every kind of
    MCP session -- a singleton-registered connector's session as well as an
    ad-hoc per-request session (e.g. Google Workspace) that isn't tracked
    in MCPClientManager._connections at all. Returns the result as a
    string ready to drop straight into a `role: "tool"` message, the same
    way api.chat.tools.execute_tool's return value already is.

    Never raises: a JSON-RPC error or transport failure degrades to a JSON
    error string instead of blowing up the calling turn, matching how
    every native tool in api.chat.tools already degrades on failure.
    """
    try:
        result = await session.call_tool(tool_name, arguments)
    except Exception:
        logger.exception(f"MCP: call_tool('{tool_name}') failed.")
        return json.dumps({"error": f"MCP tool '{tool_name}' execution failed."})

    text_parts = [block.text for block in result.content if isinstance(block, types.TextContent)]
    result_text = "\n".join(text_parts) if text_parts else ""

    # NOTE: despite the wire-level field being named `isError`, the SDK's
    # Python-side attribute is the snake_case `is_error` -- same aliasing
    # quirk as Tool.inputSchema/.input_schema below. `getattr(..., "isError", ...)`
    # would silently always be False here.
    if getattr(result, "is_error", False):
        logger.warning(f"MCP: tool '{tool_name}' returned an error result: {result_text}")
        return json.dumps({"error": result_text or f"MCP tool '{tool_name}' returned an error."})

    if not result_text:
        return json.dumps({"result": f"MCP tool '{tool_name}' executed but returned no text content."})

    return result_text


class MCPServerConnection:
    """A single live session to one external MCP server, plus its cached
    tool schemas. `connect`/`close` must be awaited from the same asyncio
    task (true of a FastAPI lifespan context manager, which runs its
    startup and shutdown halves in one task -- and equally true of a
    single chat request's async generator) -- the underlying transport
    uses anyio task groups that aren't safe to enter in one task and exit
    in another."""

    def __init__(
        self,
        connector_id: str,
        url: str,
        headers: Optional[dict[str, str]] = None,
        transport: str = "sse",
    ):
        self.connector_id = connector_id
        self.url = url
        self.headers = headers
        # "sse" (default, every pre-existing connector) or "streamable_http"
        # -- only the Gmail MCP connector (create_gmail_mcp_session) uses
        # the latter, since that's the transport Google's managed Gmail MCP
        # server speaks. Purely additive: omitting this param preserves the
        # exact prior sse_client-only behavior for every other caller.
        self.transport = transport
        self.session: Optional[ClientSession] = None
        self.tools: list[types.Tool] = []
        self._exit_stack = contextlib.AsyncExitStack()

    async def connect(self) -> None:
        if self.transport == "streamable_http":
            http_client = await self._exit_stack.enter_async_context(
                httpx.AsyncClient(headers=self.headers, timeout=CONNECT_TIMEOUT_SECONDS)
            )
            read_stream, write_stream = await self._exit_stack.enter_async_context(
                streamable_http_client(self.url, http_client=http_client)
            )
        else:
            read_stream, write_stream = await self._exit_stack.enter_async_context(
                sse_client(self.url, headers=self.headers)
            )
        self.session = await self._exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
        await self.session.initialize()
        result = await self.session.list_tools()
        self.tools = result.tools

    async def close(self) -> None:
        await self._exit_stack.aclose()
        self.session = None

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        if self.session is None:
            logger.warning(f"MCP: call_tool('{tool_name}') on connector '{self.connector_id}' with no active session.")
            return json.dumps({"error": f"MCP connector '{self.connector_id}' has no active session."})
        return await _call_tool_on_session(self.session, tool_name, arguments)


class MCPClientManager:
    """
    Accepts a list of MCPServerSpec (external MCP servers over HTTP/SSE,
    e.g. Brave Search or GitHub) and manages connecting to all of them,
    caching their advertised tools per connector_id, and shutting every
    session down cleanly. A single server being unreachable is logged and
    skipped rather than raised -- one bad MCP endpoint must never prevent
    the FastAPI app itself from starting.
    """

    def __init__(self, servers: Optional[list[MCPServerSpec]] = None):
        self.server_specs: list[MCPServerSpec] = list(servers or [])
        self._connections: dict[str, MCPServerConnection] = {}  # keyed by connector_id

    async def initialize(self) -> None:
        for spec in self.server_specs:
            connection = MCPServerConnection(spec.connector_id, spec.url, spec.headers)
            try:
                await asyncio.wait_for(connection.connect(), timeout=CONNECT_TIMEOUT_SECONDS)
            except Exception:
                logger.exception(f"MCP: failed to connect connector '{spec.connector_id}' at '{spec.url}'; skipping.")
                await connection.close()
                continue
            self._connections[spec.connector_id] = connection
            logger.info(
                f"MCP: connected connector '{spec.connector_id}' ({spec.url}) -- "
                f"discovered {len(connection.tools)} tool(s)."
            )

    async def shutdown(self) -> None:
        for connection in self._connections.values():
            try:
                await connection.close()
            except Exception:
                logger.exception(f"MCP: error while closing connector '{connection.connector_id}'.")
        self._connections.clear()

    def list_connector_ids(self) -> list[str]:
        """Every connector_id currently connected."""
        return list(self._connections.keys())

    def list_cached_tools(self, connector_ids: Optional[Iterable[str]] = None) -> list[types.Tool]:
        """Every MCP tool discovered across connected servers, flattened
        into one list. When `connector_ids` is given, only tools from
        connectors in that set are included -- None (the default) means
        every connected connector, preserving the pre-filtering behavior.
        Read-only against the cache populated by initialize() -- never
        opens a connection itself."""
        allowed = set(connector_ids) if connector_ids is not None else None
        tools: list[types.Tool] = []
        for connector_id, connection in self._connections.items():
            if allowed is not None and connector_id not in allowed:
                continue
            tools.extend(connection.tools)
        return tools

    def get_connection_for_tool(self, tool_name: str) -> Optional[MCPServerConnection]:
        """The MCPServerConnection that owns `tool_name`, for the
        execution layer to route a call to the right server. None if no
        connected server currently advertises that tool. Deliberately
        ignores connector filtering -- a tool that was actually offered to
        the model this turn (i.e. passed the enabled_connectors filter
        when the schema list was built) must always be executable, so this
        looks across every connected connector, not just an enabled
        subset."""
        for connection in self._connections.values():
            if any(t.name == tool_name for t in connection.tools):
                return connection
        return None

    def get_session_for_tool(self, tool_name: str) -> Optional[ClientSession]:
        """The ClientSession that owns `tool_name` -- see
        get_connection_for_tool. Kept as a thin convenience wrapper for
        anything that only needs the raw session."""
        connection = self.get_connection_for_tool(tool_name)
        return connection.session if connection else None

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """
        Executes `tool_name` on whichever connected, singleton-registered
        MCP server advertised it. For a Google Workspace tool from a
        per-request session, call `.call_tool` directly on the
        MCPServerConnection yielded by create_workspace_session instead --
        that session is never registered here, by design (see
        create_workspace_session).
        """
        connection = self.get_connection_for_tool(tool_name)
        if connection is None:
            logger.warning(f"MCP: call_tool('{tool_name}') requested but no connected server advertises it.")
            return json.dumps({"error": f"No connected MCP server advertises tool '{tool_name}'."})
        return await connection.call_tool(tool_name, arguments)

    @contextlib.asynccontextmanager
    async def create_workspace_session(
        self,
        token: Optional[str],
        supabase: Optional[Client] = None,
        user_id: Optional[str] = None,
    ) -> AsyncIterator[Optional[MCPServerConnection]]:
        """
        Opens a short-lived MCP session to the Google Workspace connector,
        authenticated with `token` -- this specific user's own OAuth access
        token, sent as `Authorization: Bearer <token>` on the SSE
        connection -- and guarantees it is closed again before this
        context manager exits, whether the caller's `async with` block
        finishes normally, raises, or is cancelled mid-stream (e.g. the
        client disconnects).

        Deliberately NEVER stored in `self._connections`: unlike the
        static singleton connectors (GitHub, Brave Search, ...), which are
        connected once at startup and shared read-only across every
        request, a Workspace session is scoped to exactly one request/user
        and authenticated with that user's own credential. Registering it
        on the shared manager -- even temporarily -- would risk another
        concurrent request's tool call being routed through (and thus
        executing with) a different user's Google token. Each call to this
        method gets its own private connection instead.

        `supabase`/`user_id`, when both given, back a server-side fallback:
        if `token` itself is falsy, this looks up and refreshes this user's
        stored Google refresh_token (see _refresh_google_access_token)
        before falling back to "no session" -- letting a previously-linked
        user's Workspace tools keep working across reloads even once their
        browser-cached access token has aged out of the session, without
        ever re-prompting for OAuth consent. Omit both (the default) to
        keep the exact prior behavior: no token in, no session, full stop --
        see ChatService.stream_chat for when the caller chooses to pass
        them (only when this turn could actually use the result).

        Yields None -- and opens no connection at all -- when `token` is
        still falsy after that fallback, or no `GOOGLE_WORKSPACE_MCP_URL` is
        configured. That's the default for every request that isn't using
        this connector, and for BRANCH A (a file attached, no tools offered
        this turn) regardless of whether a token was sent -- see
        ChatService.stream_chat, which passes None for `token` (and omits
        `supabase`/`user_id`) in that case specifically to skip the SSE
        round trip entirely.

        A connection failure (bad token, unreachable server) is logged and
        also yields None rather than raising -- one broken Workspace
        session must degrade the turn to "no Workspace tools available"
        exactly like an unreachable static connector already does, never
        crash the chat stream.
        """
        settings = get_settings()

        had_live_token = bool(token)
        if not token and supabase is not None and user_id:
            token = await _refresh_google_access_token(supabase, user_id)

        if not token:
            # Previously a bare `yield None` -- gives no signal at all
            # whether this was "no token sent and no refresh fallback
            # attempted" (supabase/user_id omitted -- see ChatService
            # .stream_chat's BRANCH A / google_workspace_requested gating)
            # vs. "refresh fallback WAS attempted and failed" (see
            # _refresh_google_access_token's own ERROR logs just above this
            # one in the log stream for the real reason in that case).
            logger.error(
                "MCP: no Google Workspace session opened for user %s -- no live access token was sent"
                " and %s.",
                user_id,
                "the refresh-token fallback was attempted (see the error logged just above, if any)"
                if (not had_live_token and supabase is not None and user_id)
                else "no refresh-token fallback was attempted (supabase/user_id not passed this turn)",
            )
            yield None
            return

        if not settings.google_workspace_mcp_url:
            logger.error(
                "MCP: no Google Workspace session opened for user %s -- a usable access token was "
                "obtained, but GOOGLE_WORKSPACE_MCP_URL is not configured.",
                user_id,
            )
            yield None
            return

        connection = MCPServerConnection(
            "google_workspace",
            settings.google_workspace_mcp_url,
            headers={"Authorization": f"Bearer {token}"},
        )
        try:
            await asyncio.wait_for(connection.connect(), timeout=CONNECT_TIMEOUT_SECONDS)
        except Exception:
            logger.error(
                "MCP: failed to establish Google Workspace session for user %s at %s -- "
                "continuing without it.",
                user_id, settings.google_workspace_mcp_url, exc_info=True,
            )
            await connection.close()
            yield None
            return

        logger.info(
            "MCP: Google Workspace session established for user %s -- discovered %d tool(s): %s",
            user_id, len(connection.tools), [t.name for t in connection.tools],
        )

        try:
            yield connection
        finally:
            await connection.close()

    @contextlib.asynccontextmanager
    async def create_github_session(self, token: Optional[str]) -> AsyncIterator[Optional[MCPServerConnection]]:
        """
        Per-user counterpart to create_workspace_session, for a caller's own
        GitHub OAuth access token (Phase 6) rather than this process's
        single shared GITHUB_PERSONAL_ACCESS_TOKEN. Opens a short-lived MCP
        session against the same `GITHUB_MCP_SERVER_URL` the static
        connector uses (see _build_default_server_specs), authenticated
        with `token` as `Authorization: Bearer <token>` instead of the
        static PAT, and guarantees it's closed again before this context
        manager exits -- normal completion, an exception, or cancellation
        mid-stream.

        Deliberately NEVER stored in `self._connections`, for the exact
        same tenant-isolation reason documented on create_workspace_session:
        this session is authenticated with one specific user's own
        credential, so registering it on the shared manager -- even
        temporarily -- would risk a concurrent request being routed through
        another user's GitHub token.

        Yields None -- and opens no connection at all -- when `token` is
        falsy or no `GITHUB_MCP_SERVER_URL` is configured, and also on a
        connection failure (bad token, unreachable server), logged but
        never raised. Same contract as create_workspace_session in every
        other respect -- see its docstring for the full rationale.
        """
        settings = get_settings()
        if not token or not settings.github_mcp_server_url:
            yield None
            return

        connection = MCPServerConnection(
            "github",
            settings.github_mcp_server_url,
            headers={"Authorization": f"Bearer {token}"},
        )
        try:
            await asyncio.wait_for(connection.connect(), timeout=CONNECT_TIMEOUT_SECONDS)
        except Exception:
            logger.exception("MCP: failed to establish per-user GitHub session; continuing without it.")
            await connection.close()
            yield None
            return

        try:
            yield connection
        finally:
            await connection.close()

    @contextlib.asynccontextmanager
    async def create_gmail_mcp_session(
        self,
        supabase: Optional[Client] = None,
        user_id: Optional[str] = None,
    ) -> AsyncIterator[Optional[MCPServerConnection]]:
        """
        Per-user session for Google's managed Gmail MCP server
        (mcp_integration.gmail_mcp.GMAIL_MCP_SERVER_URL) -- fully isolated
        from create_workspace_session/create_github_session above: it
        authenticates with its own OAuth client (GMAIL_MCP_CLIENT_ID/
        SECRET) via gmail_mcp.refresh_gmail_mcp_access_token, which reads
        this user's stored token from the distinct 'google_gmail_mcp'
        user_oauth_tokens row rather than the plain 'google' row
        create_workspace_session's own refresh fallback uses -- see
        mcp_integration/gmail_mcp.py's module docstring for why those two
        code paths are kept fully separate rather than shared. Also unlike
        every other connector here, it dials over Streamable HTTP
        (MCPServerConnection(transport="streamable_http")), not SSE, since
        that's the transport Google's managed MCP server speaks.

        Same yield contract as create_workspace_session/create_github_session:
        None -- and no connection opened at all -- when `supabase`/`user_id`
        aren't both given, GMAIL_MCP_CLIENT_ID/SECRET/REDIRECT_URI aren't
        fully configured, this user has no stored google_gmail_mcp refresh
        token (refresh_gmail_mcp_access_token returns None), or the connect
        itself fails -- every case logged, never raised, so one broken/
        unlinked Gmail MCP session degrades a turn to "no Gmail tools
        available" exactly like the other connectors already do. Never
        registered on self._connections, for the identical per-user
        tenant-isolation reason documented on create_workspace_session.

        The returned connection's `.tools` are ordinary `mcp.types.Tool`
        objects, convertible with mcp_tool_to_native_schema exactly like
        workspace_connection.tools/github_connection.tools already are in
        ChatService.stream_chat -- merging them into that turn's
        active_tools is the same one-line pattern used there for
        github_connection, left for whichever phase wires this connector
        into the live chat loop.
        """
        if supabase is None or not user_id:
            yield None
            return

        settings = get_settings()
        if not (settings.gmail_mcp_client_id and settings.gmail_mcp_client_secret and settings.gmail_mcp_redirect_uri):
            yield None
            return

        access_token = await refresh_gmail_mcp_access_token(supabase, user_id)
        if not access_token:
            yield None
            return

        connection = MCPServerConnection(
            "google_gmail_mcp",
            GMAIL_MCP_SERVER_URL,
            headers={"Authorization": f"Bearer {access_token}"},
            transport="streamable_http",
        )
        try:
            await asyncio.wait_for(connection.connect(), timeout=CONNECT_TIMEOUT_SECONDS)
        except Exception:
            logger.error(
                "MCP: failed to establish Gmail MCP session for user %s at %s -- continuing without it.",
                user_id, GMAIL_MCP_SERVER_URL, exc_info=True,
            )
            await connection.close()
            yield None
            return

        logger.info(
            "MCP: Gmail MCP session established for user %s -- discovered %d tool(s): %s",
            user_id, len(connection.tools), [t.name for t in connection.tools],
        )

        try:
            yield connection
        finally:
            await connection.close()


def _build_default_server_specs() -> list[MCPServerSpec]:
    """
    Reads Settings to build the list of STATIC MCP servers to connect to
    at startup (the Google Workspace connector is intentionally excluded --
    it's opened dynamically per-request by create_workspace_session, never
    at startup, since it has no single shared credential to connect with):

    - `mcp_server_urls` (generic, unnamed URLs from Phase 1) each get an
      auto-assigned connector_id ("mcp_server_1", "mcp_server_2", ...).
    - The GitHub MCP connector is added ONLY when both
      `github_mcp_server_url` and `github_personal_access_token` are
      configured -- the token is sent as a Bearer header on the SSE
      connection. Neither being set (the default) means GitHub is simply
      never dialed; that's the expected zero-risk state until both are
      provided. A token with no URL (or vice versa) is a likely
      misconfiguration, so it's logged rather than silently ignored.
    """
    settings = get_settings()
    specs: list[MCPServerSpec] = []

    for i, url in enumerate(settings.mcp_server_url_list, start=1):
        specs.append(MCPServerSpec(connector_id=f"mcp_server_{i}", url=url))

    if settings.github_mcp_server_url and settings.github_personal_access_token:
        specs.append(MCPServerSpec(
            connector_id="github",
            url=settings.github_mcp_server_url,
            headers={"Authorization": f"Bearer {settings.github_personal_access_token}"},
        ))
    elif settings.github_personal_access_token or settings.github_mcp_server_url:
        logger.warning(
            "MCP: GitHub connector needs both GITHUB_MCP_SERVER_URL and "
            "GITHUB_PERSONAL_ACCESS_TOKEN set; only one is configured, so it will not be connected."
        )

    return specs


# Single shared instance for the whole app -- main.py's lifespan calls
# initialize()/shutdown() on exactly this object, and api.chat.services
# imports this same instance to merge schemas and route tool calls.
# Defined here (rather than in main.py) so importing it never risks a
# circular import through main -> api.chat.routers -> api.chat.services.
mcp_manager = MCPClientManager(servers=_build_default_server_specs())


def mcp_tool_to_native_schema(tool: types.Tool) -> dict[str, Any]:
    """Converts an MCP `Tool` (name/description/input_schema -- the SDK's
    Python-side attribute name for the wire-level `inputSchema` field) into
    this app's native tool schema shape -- {"type": "function", "function":
    {...}} -- matching api.chat.tools.ALL_TOOLS, so both can sit in the
    same list handed to the LLM. Public: also used directly by
    ChatService.stream_chat to merge a per-request Google Workspace
    session's tools, which aren't part of any MCPClientManager-cached list."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.input_schema,
        },
    }


# Phase 7: granular Google Workspace sub-connector ids, as sent by the
# frontend's per-service toggles (Gmail/Docs/Calendar/Drive), each mapped to
# the tool-name prefix(es) that service's MCP tools are expected to use.
# Drive tools may be prefixed either way depending on the MCP server's own
# naming, so both are matched.
GOOGLE_WORKSPACE_SUB_CONNECTOR_PREFIXES: dict[str, tuple[str, ...]] = {
    "google_gmail": ("gmail_",),
    "google_docs": ("docs_",),
    "google_calendar": ("calendar_",),
    "google_drive": ("drive_", "gdrive_"),
}

# The pre-Phase-7 single toggle -- still honored for backward compatibility
# with any caller (or cached frontend build) still sending it instead of the
# four granular ids.
LEGACY_GOOGLE_WORKSPACE_CONNECTOR_ID = "google_workspace"


def filter_workspace_tools(
    tools: list[types.Tool],
    enabled_connectors: Optional[list[str]],
) -> list[types.Tool]:
    """
    Narrows a per-request Google Workspace session's discovered tools
    (create_workspace_session) down to whichever granular sub-connectors the
    frontend actually toggled on this turn, by matching each tool's name
    against GOOGLE_WORKSPACE_SUB_CONNECTOR_PREFIXES -- e.g. only Gmail's
    tools are offered when `enabled_connectors` contains "google_gmail" but
    none of the other three. Exists because a Workspace session's tools
    aren't part of MCPClientManager's cached, singleton-registered
    connectors, so get_merged_tool_schemas' own enabled_connectors filtering
    (which only ever looks at `manager.list_cached_tools`) never sees them at
    all -- this is the equivalent filter for that separate, per-request tool
    list. Called directly from ChatService.stream_chat's own Workspace-tool
    merge loop, not from get_merged_tool_schemas.

    Mirrors get_merged_tool_schemas' own enabled_connectors contract:
    - None (nothing sent -- an old caller, or a request that never sends the
      field) offers every discovered tool, unfiltered.
    - The legacy single "google_workspace" id, if present, also offers every
      discovered tool, unfiltered -- full backward compatibility with the
      pre-Phase-7 single-toggle behavior.
    - Otherwise, only tools matching an *enabled* sub-connector's prefix are
      returned. None of the four enabled (e.g. an empty list, or a list with
      only unrelated ids like "github") returns no Workspace tools at all --
      matches list_cached_tools' own "explicit empty selection means
      nothing" behavior for static connectors.
    """
    if enabled_connectors is None or LEGACY_GOOGLE_WORKSPACE_CONNECTOR_ID in enabled_connectors:
        return list(tools)

    active_prefixes = tuple(
        prefix
        for connector_id, prefixes in GOOGLE_WORKSPACE_SUB_CONNECTOR_PREFIXES.items()
        if connector_id in enabled_connectors
        for prefix in prefixes
    )
    if not active_prefixes:
        return []

    return [tool for tool in tools if tool.name.startswith(active_prefixes)]


def get_merged_tool_schemas(
    native_tools: list[dict[str, Any]],
    manager: MCPClientManager,
    enabled_connectors: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """
    Merges AetherChat's hardcoded native tool schemas with the cached
    schemas of tools discovered from connected MCP servers into one unified
    list, in the same shape the LLM provider expects.

    Native tools are always included -- `enabled_connectors` only ever
    filters which *MCP* connectors' tools are offered this turn, never the
    native registry. When `enabled_connectors` is None (the default), every
    connected connector's tools are included, exactly matching pre-Phase-3
    behavior -- so an old caller, or a request that never sends the field,
    is unaffected.

    A native tool name always wins on collision -- native tools are this
    app's own audited implementations, so an external server advertising
    e.g. "calculator" never shadows them.
    """
    native_names = {t["function"]["name"] for t in native_tools}
    merged = list(native_tools)
    for tool in manager.list_cached_tools(enabled_connectors):
        if tool.name in native_names:
            logger.warning(
                f"MCP: external tool '{tool.name}' collides with a native tool name; keeping the native one."
            )
            continue
        merged.append(mcp_tool_to_native_schema(tool))
    return merged
