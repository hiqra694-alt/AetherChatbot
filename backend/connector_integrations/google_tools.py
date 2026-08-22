"""
Native Google Workspace tools (Phase 3) -- Gmail/Calendar/Drive functions
that call the official Google API client libraries directly, replacing the
retired Google Managed MCP Workspace connector (see
connector_integrations/connector_manager.py's module docstring).

Each tool is a plain `async def(supabase, user_id, params: <Tool>Params)`
function: `params` is a Pydantic model (its own "proper input schema") whose
fields are exactly the tool's arguments, with `Field(..., description=...)`
on each one -- `<Tool>Params.model_json_schema()` is what
google_tool_to_native_schema (below) hands the LLM as the tool's
`parameters` schema, the same shape
`connector_integrations.connector_manager.mcp_tool_to_native_schema` already
produces for MCP-sourced tools. Every function returns a plain,
JSON-serializable dict -- either the result, or `{"error": "..."}` on
failure, mirroring every other tool-execution path in this codebase (e.g.
mcp_manager._call_tool_on_session): never raise out of a tool call.

Wired into the live chat loop (Phase 4) via GOOGLE_WORKSPACE_TOOLS +
execute_google_workspace_tool below -- see api/chat/services.py's
stream_chat (schema merge + "Executing Google Workspace Tool" branch) and
connector_integrations/intent_router.py's select_google_workspace_tools
(toggle filtering + keyword routing, kept dependency-free of this module).

Credentials: get_credentials() loads this user's stored 'google_workspace'
refresh token (written by api/connectors/google_oauth.py's callback) and
refreshes it into a live access token via google-auth. Each tool then builds
the specific service resource it needs (gmail v1 / calendar v3 / drive v3)
with googleapiclient.discovery.build(). Both the refresh and every
service .execute() call are blocking (httplib2/requests under the hood, not
asyncio) -- run via asyncio.to_thread so the event loop is never blocked by
them.
"""

import asyncio
import base64
import datetime
import json
import logging
from email.mime.text import MIMEText
from typing import Awaitable, Callable, List, Optional, Tuple, Type

from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaInMemoryUpload
from pydantic import BaseModel, Field, ValidationError
from supabase import Client

from api.connectors.google_oauth import (
    GOOGLE_OAUTH_TOKEN_URI,
    GOOGLE_WORKSPACE_PROVIDER,
    GOOGLE_WORKSPACE_SCOPES,
)
from core.config import get_settings

logger = logging.getLogger(__name__)


class GoogleWorkspaceNotConnectedError(Exception):
    """
    Raised internally by get_credentials() when this user has no usable
    Google Workspace credential -- never lets this escape a tool function;
    each of the 8 tools below catches it and returns a plain
    {"error": ...} dict instead, the same "degrade, never raise out of a
    tool call" contract every other connector in this codebase follows.
    """


async def _get_stored_refresh_token(supabase: Client, user_id: str) -> Optional[str]:
    """The stored Google Workspace refresh token for `user_id`, if any --
    see api/connectors/google_oauth.py's callback for where this is
    written."""
    res = (
        supabase.table("user_oauth_tokens")
        .select("refresh_token")
        .eq("user_id", user_id)
        .eq("provider", GOOGLE_WORKSPACE_PROVIDER)
        .limit(1)
        .execute()
    )
    rows = res.data or []
    return rows[0].get("refresh_token") if rows else None


async def get_credentials(supabase: Client, user_id: str) -> Credentials:
    """
    Builds a live, refreshed google.oauth2.credentials.Credentials for
    `user_id` from their stored Google Workspace refresh token. Raises
    GoogleWorkspaceNotConnectedError -- never any other exception type --
    when the OAuth client isn't configured, no refresh token is stored for
    this user, or Google rejects the refresh (revoked/expired grant); every
    caller is expected to catch exactly that one type.
    """
    settings = get_settings()
    if not settings.google_workspace_client_id or not settings.google_workspace_client_secret:
        raise GoogleWorkspaceNotConnectedError(
            "Google Workspace connector is not configured on this deployment."
        )

    try:
        refresh_token = await _get_stored_refresh_token(supabase, user_id)
    except Exception as exc:
        logger.error("Google Workspace: failed to look up stored refresh token for user %s.", user_id, exc_info=True)
        raise GoogleWorkspaceNotConnectedError("Failed to look up the stored Google Workspace connection.") from exc

    if not refresh_token:
        raise GoogleWorkspaceNotConnectedError(
            "Google Workspace is not connected for this user. Connect it via "
            "/api/connectors/google/authorize first."
        )

    credentials = Credentials(
        token=None,
        refresh_token=refresh_token,
        token_uri=GOOGLE_OAUTH_TOKEN_URI,
        client_id=settings.google_workspace_client_id,
        client_secret=settings.google_workspace_client_secret,
        scopes=GOOGLE_WORKSPACE_SCOPES,
    )

    try:
        await asyncio.to_thread(credentials.refresh, GoogleAuthRequest())
    except Exception as exc:
        # Most commonly a revoked/expired refresh token (the user removed
        # this app's access from their Google Account settings) -- logged
        # with the real reason, degraded to the one exception type callers
        # already handle.
        logger.error("Google Workspace: failed to refresh access token for user %s.", user_id, exc_info=True)
        raise GoogleWorkspaceNotConnectedError(
            "Failed to refresh Google Workspace credentials -- the connection may have been revoked. "
            "Disconnect and reconnect via /api/connectors/google/authorize."
        ) from exc

    return credentials


async def _build_service(service_name: str, version: str, credentials: Credentials):
    """googleapiclient.discovery.build() ships static discovery docs for
    Gmail/Calendar/Drive so this rarely makes a network call, but it's still
    synchronous -- run off the event loop like every other Google API call
    here. static_discovery=False keeps behavior identical to the client
    library's own default resolution order (static first, network
    fallback) rather than forcing one or the other."""
    return await asyncio.to_thread(
        build, service_name, version, credentials=credentials, cache_discovery=False
    )


def _b64url_encode_message(mime_message: MIMEText) -> str:
    return base64.urlsafe_b64encode(mime_message.as_bytes()).decode()


# ============================================================
# 1. gmail_search_recent
# ============================================================

class GmailSearchRecentParams(BaseModel):
    query: str = Field(
        ...,
        description=(
            "Gmail search query using Gmail's own search operators, e.g. "
            "'from:boss@company.com is:unread', 'subject:invoice after:2024/01/01', "
            "or a plain keyword search."
        ),
    )
    max_results: int = Field(10, ge=1, le=50, description="Maximum number of messages to return.")


async def gmail_search_recent(supabase: Client, user_id: str, params: GmailSearchRecentParams) -> dict:
    """Search the user's Gmail for recent messages matching a query, returning
    each match's id, thread id, subject, sender, date, and snippet. Call this
    first to find a message before reading its full content or replying to
    it -- it does not return full message bodies."""
    try:
        credentials = await get_credentials(supabase, user_id)
        service = await _build_service("gmail", "v1", credentials)
    except GoogleWorkspaceNotConnectedError as exc:
        return {"error": str(exc)}

    def _run() -> list:
        result = service.users().messages().list(
            userId="me", q=params.query, maxResults=params.max_results
        ).execute()
        messages = []
        for msg in result.get("messages", []):
            detail = service.users().messages().get(
                userId="me", id=msg["id"], format="metadata",
                metadataHeaders=["Subject", "From", "Date"],
            ).execute()
            headers = {h["name"]: h["value"] for h in detail.get("payload", {}).get("headers", [])}
            messages.append({
                "id": msg["id"],
                "thread_id": detail.get("threadId"),
                "subject": headers.get("Subject", ""),
                "from": headers.get("From", ""),
                "date": headers.get("Date", ""),
                "snippet": detail.get("snippet", ""),
            })
        return messages

    try:
        messages = await asyncio.to_thread(_run)
    except HttpError as exc:
        logger.error("Gmail search failed for user %s: %s", user_id, exc)
        return {"error": f"Gmail search failed: {exc}"}

    return {"messages": messages}


# ============================================================
# 2. gmail_read_thread
# ============================================================

class GmailReadThreadParams(BaseModel):
    thread_id: str = Field(..., description="The Gmail thread id to read, as returned by gmail_search_recent.")


def _decode_gmail_body(payload: dict) -> str:
    """Best-effort plain-text extraction from a Gmail message payload --
    walks multipart MIME parts looking for text/plain, falling back to
    text/html (still base64url-decoded, tags included) if no plain-text
    part exists."""
    def _walk(part: dict) -> Optional[str]:
        mime_type = part.get("mimeType", "")
        body_data = part.get("body", {}).get("data")
        if mime_type == "text/plain" and body_data:
            return base64.urlsafe_b64decode(body_data + "==").decode("utf-8", errors="replace")
        for sub_part in part.get("parts", []) or []:
            found = _walk(sub_part)
            if found:
                return found
        if mime_type == "text/html" and body_data:
            return base64.urlsafe_b64decode(body_data + "==").decode("utf-8", errors="replace")
        return None

    return _walk(payload) or ""


async def gmail_read_thread(supabase: Client, user_id: str, params: GmailReadThreadParams) -> dict:
    """Read the full content of every message in a Gmail thread (subject,
    sender, date, and body text for each message), in order. Use this after
    gmail_search_recent has identified the thread the user is asking about."""
    try:
        credentials = await get_credentials(supabase, user_id)
        service = await _build_service("gmail", "v1", credentials)
    except GoogleWorkspaceNotConnectedError as exc:
        return {"error": str(exc)}

    def _run() -> list:
        thread = service.users().threads().get(userId="me", id=params.thread_id, format="full").execute()
        messages = []
        for msg in thread.get("messages", []):
            payload = msg.get("payload", {})
            headers = {h["name"]: h["value"] for h in payload.get("headers", [])}
            messages.append({
                "id": msg.get("id"),
                "subject": headers.get("Subject", ""),
                "from": headers.get("From", ""),
                "to": headers.get("To", ""),
                "date": headers.get("Date", ""),
                "body": _decode_gmail_body(payload),
            })
        return messages

    try:
        messages = await asyncio.to_thread(_run)
    except HttpError as exc:
        logger.error("Gmail read thread failed for user %s: %s", user_id, exc)
        return {"error": f"Failed to read Gmail thread: {exc}"}

    return {"thread_id": params.thread_id, "messages": messages}


# ============================================================
# 3. gmail_create_draft
# ============================================================

class GmailCreateDraftParams(BaseModel):
    to: str = Field(..., description="Recipient email address.")
    subject: str = Field(..., description="Email subject line.")
    body: str = Field(..., description="Plain-text email body.")


async def gmail_create_draft(supabase: Client, user_id: str, params: GmailCreateDraftParams) -> dict:
    """Create a draft email in the user's Gmail account without sending it.
    Use this when the user asks you to draft, prepare, or write an email for
    their review rather than send it immediately."""
    try:
        credentials = await get_credentials(supabase, user_id)
        service = await _build_service("gmail", "v1", credentials)
    except GoogleWorkspaceNotConnectedError as exc:
        return {"error": str(exc)}

    mime_message = MIMEText(params.body)
    mime_message["to"] = params.to
    mime_message["subject"] = params.subject
    raw = _b64url_encode_message(mime_message)

    def _run() -> dict:
        return service.users().drafts().create(userId="me", body={"message": {"raw": raw}}).execute()

    try:
        draft = await asyncio.to_thread(_run)
    except HttpError as exc:
        logger.error("Gmail create draft failed for user %s: %s", user_id, exc)
        return {"error": f"Failed to create Gmail draft: {exc}"}

    return {"draft_id": draft.get("id"), "message_id": draft.get("message", {}).get("id")}


# ============================================================
# 4. gmail_send_email
# ============================================================

class GmailSendEmailParams(BaseModel):
    to: str = Field(..., description="Recipient email address.")
    subject: str = Field(..., description="Email subject line.")
    body: str = Field(..., description="Plain-text email body.")


async def gmail_send_email(supabase: Client, user_id: str, params: GmailSendEmailParams) -> dict:
    """Send an email immediately from the user's Gmail account. Only use
    this when the user has clearly asked you to send an email now -- prefer
    gmail_create_draft when they want to review it first."""
    try:
        credentials = await get_credentials(supabase, user_id)
        service = await _build_service("gmail", "v1", credentials)
    except GoogleWorkspaceNotConnectedError as exc:
        return {"error": str(exc)}

    mime_message = MIMEText(params.body)
    mime_message["to"] = params.to
    mime_message["subject"] = params.subject
    raw = _b64url_encode_message(mime_message)

    def _run() -> dict:
        return service.users().messages().send(userId="me", body={"raw": raw}).execute()

    try:
        sent = await asyncio.to_thread(_run)
    except HttpError as exc:
        logger.error("Gmail send failed for user %s: %s", user_id, exc)
        return {"error": f"Failed to send email: {exc}"}

    return {"message_id": sent.get("id"), "thread_id": sent.get("threadId")}


# ============================================================
# 5. calendar_list_events
# ============================================================

class CalendarListEventsParams(BaseModel):
    time_min: Optional[str] = Field(
        None, description="RFC3339 timestamp (e.g. '2024-06-01T00:00:00Z'). Defaults to now if omitted."
    )
    time_max: Optional[str] = Field(
        None, description="RFC3339 timestamp. Omit for no upper bound."
    )
    max_results: int = Field(10, ge=1, le=50, description="Maximum number of events to return.")


async def calendar_list_events(supabase: Client, user_id: str, params: CalendarListEventsParams) -> dict:
    """List upcoming events on the user's primary Google Calendar, optionally
    bounded by a time range. Use this to check the user's schedule or find an
    event before modifying it."""
    try:
        credentials = await get_credentials(supabase, user_id)
        service = await _build_service("calendar", "v3", credentials)
    except GoogleWorkspaceNotConnectedError as exc:
        return {"error": str(exc)}

    time_min = params.time_min or datetime.datetime.now(datetime.timezone.utc).isoformat()

    def _run() -> list:
        kwargs = dict(
            calendarId="primary",
            timeMin=time_min,
            maxResults=params.max_results,
            singleEvents=True,
            orderBy="startTime",
        )
        if params.time_max:
            kwargs["timeMax"] = params.time_max
        result = service.events().list(**kwargs).execute()
        events = []
        for event in result.get("items", []):
            events.append({
                "id": event.get("id"),
                "summary": event.get("summary", ""),
                "start": event.get("start", {}).get("dateTime") or event.get("start", {}).get("date"),
                "end": event.get("end", {}).get("dateTime") or event.get("end", {}).get("date"),
                "location": event.get("location", ""),
                "html_link": event.get("htmlLink", ""),
            })
        return events

    try:
        events = await asyncio.to_thread(_run)
    except HttpError as exc:
        logger.error("Calendar list events failed for user %s: %s", user_id, exc)
        return {"error": f"Failed to list calendar events: {exc}"}

    return {"events": events}


# ============================================================
# 6. calendar_create_event
# ============================================================

class CalendarCreateEventParams(BaseModel):
    summary: str = Field(..., description="Event title.")
    start_time: str = Field(..., description="RFC3339 start timestamp, e.g. '2024-06-01T15:00:00-07:00'.")
    end_time: str = Field(..., description="RFC3339 end timestamp, e.g. '2024-06-01T16:00:00-07:00'.")
    description: Optional[str] = Field(None, description="Optional event description/notes.")
    attendees: Optional[List[str]] = Field(None, description="Optional list of attendee email addresses to invite.")
    timezone: str = Field("UTC", description="IANA timezone name for start/end, e.g. 'America/Los_Angeles'.")


async def calendar_create_event(supabase: Client, user_id: str, params: CalendarCreateEventParams) -> dict:
    """Create a new event on the user's primary Google Calendar, optionally
    inviting attendees. Use this when the user asks you to schedule a
    meeting or add something to their calendar."""
    try:
        credentials = await get_credentials(supabase, user_id)
        service = await _build_service("calendar", "v3", credentials)
    except GoogleWorkspaceNotConnectedError as exc:
        return {"error": str(exc)}

    event_body = {
        "summary": params.summary,
        "start": {"dateTime": params.start_time, "timeZone": params.timezone},
        "end": {"dateTime": params.end_time, "timeZone": params.timezone},
    }
    if params.description:
        event_body["description"] = params.description
    if params.attendees:
        event_body["attendees"] = [{"email": email} for email in params.attendees]

    def _run() -> dict:
        return service.events().insert(
            calendarId="primary", body=event_body, sendUpdates="all" if params.attendees else "none"
        ).execute()

    try:
        event = await asyncio.to_thread(_run)
    except HttpError as exc:
        logger.error("Calendar create event failed for user %s: %s", user_id, exc)
        return {"error": f"Failed to create calendar event: {exc}"}

    return {"event_id": event.get("id"), "html_link": event.get("htmlLink", "")}


# ============================================================
# 7. drive_search_docs
# ============================================================

class DriveSearchDocsParams(BaseModel):
    query: str = Field(..., description="Text to search for in file names, e.g. 'Q3 budget'.")
    max_results: int = Field(10, ge=1, le=50, description="Maximum number of files to return.")


async def drive_search_docs(supabase: Client, user_id: str, params: DriveSearchDocsParams) -> dict:
    """Search the user's Google Drive for files by name. Read-only -- use
    this to find a file's id before referencing it elsewhere; it does not
    return file content."""
    try:
        credentials = await get_credentials(supabase, user_id)
        service = await _build_service("drive", "v3", credentials)
    except GoogleWorkspaceNotConnectedError as exc:
        return {"error": str(exc)}

    # Drive's query language treats ' as a string delimiter -- escape any
    # literal single quote in the user's search text so it can't break out
    # of the `name contains '...'` clause.
    escaped_query = params.query.replace("'", "\\'")

    def _run() -> list:
        result = service.files().list(
            q=f"name contains '{escaped_query}' and trashed = false",
            pageSize=params.max_results,
            fields="files(id, name, mimeType, webViewLink, modifiedTime)",
        ).execute()
        return result.get("files", [])

    try:
        files = await asyncio.to_thread(_run)
    except HttpError as exc:
        logger.error("Drive search failed for user %s: %s", user_id, exc)
        return {"error": f"Drive search failed: {exc}"}

    return {"files": files}


# ============================================================
# 8. drive_create_doc
# ============================================================

class DriveCreateDocParams(BaseModel):
    title: str = Field(..., description="Title of the new Google Doc.")
    content: Optional[str] = Field(None, description="Optional plain-text content to populate the document with.")


async def drive_create_doc(supabase: Client, user_id: str, params: DriveCreateDocParams) -> dict:
    """Create a new Google Doc in the user's Drive, optionally pre-filled
    with plain-text content. Use this when the user asks you to draft a
    document, notes, or a report as a Google Doc."""
    try:
        credentials = await get_credentials(supabase, user_id)
        service = await _build_service("drive", "v3", credentials)
    except GoogleWorkspaceNotConnectedError as exc:
        return {"error": str(exc)}

    file_metadata = {"name": params.title, "mimeType": "application/vnd.google-apps.document"}

    def _run() -> dict:
        if params.content:
            media = MediaInMemoryUpload(params.content.encode("utf-8"), mimetype="text/plain")
            return service.files().create(
                body=file_metadata, media_body=media, fields="id, webViewLink"
            ).execute()
        return service.files().create(body=file_metadata, fields="id, webViewLink").execute()

    try:
        created = await asyncio.to_thread(_run)
    except HttpError as exc:
        logger.error("Drive create doc failed for user %s: %s", user_id, exc)
        return {"error": f"Failed to create Google Doc: {exc}"}

    return {
        "document_id": created.get("id"),
        "web_view_link": created.get("webViewLink", f"https://docs.google.com/document/d/{created.get('id')}/edit"),
    }


# ============================================================
# Registry + dispatch -- ties each tool's name to its function and Params
# model, so callers (intent_router.select_google_workspace_tools,
# ChatService.stream_chat) never need their own hardcoded list of the 8
# tools.
# ============================================================

GoogleToolFunc = Callable[[Client, str, BaseModel], Awaitable[dict]]

GOOGLE_WORKSPACE_TOOLS: dict[str, Tuple[GoogleToolFunc, Type[BaseModel]]] = {
    "gmail_search_recent": (gmail_search_recent, GmailSearchRecentParams),
    "gmail_read_thread": (gmail_read_thread, GmailReadThreadParams),
    "gmail_create_draft": (gmail_create_draft, GmailCreateDraftParams),
    "gmail_send_email": (gmail_send_email, GmailSendEmailParams),
    "calendar_list_events": (calendar_list_events, CalendarListEventsParams),
    "calendar_create_event": (calendar_create_event, CalendarCreateEventParams),
    "drive_search_docs": (drive_search_docs, DriveSearchDocsParams),
    "drive_create_doc": (drive_create_doc, DriveCreateDocParams),
}


def google_tool_to_native_schema(tool_name: str) -> dict:
    """
    Converts one GOOGLE_WORKSPACE_TOOLS entry into this app's native tool
    schema shape -- {"type": "function", "function": {...}} -- matching
    api.chat.tools.ALL_TOOLS and
    connector_integrations.connector_manager.mcp_tool_to_native_schema's own
    output, so all three can sit in the same `tools` list handed to the LLM
    provider. The tool's own docstring (written for exactly this purpose --
    see each function's docstring above) becomes the schema `description`;
    `<Params>.model_json_schema()` becomes `parameters`.
    """
    func, params_model = GOOGLE_WORKSPACE_TOOLS[tool_name]
    return {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": (func.__doc__ or "").strip(),
            "parameters": params_model.model_json_schema(),
        },
    }


async def execute_google_workspace_tool(tool_name: str, arguments: dict, supabase: Client, user_id: str) -> str:
    """
    Validates `arguments` against `tool_name`'s Params model and executes
    it, returning a JSON string ready to drop straight into a `role: "tool"`
    message -- the same contract every other tool-execution path in this
    codebase already follows (api.chat.tools.execute_tool,
    mcp_manager.call_tool). Never raises: an unknown tool name, a Pydantic
    validation error (the LLM emitted arguments that don't match the
    schema), or an unexpected exception from the underlying Google API call
    all degrade to a JSON {"error": ...} string instead of blowing up the
    turn -- `arguments` itself has already been through
    api.chat.tools.repair_tool_arguments by the time it reaches here (see
    ChatService.stream_chat's tool-call loop), so this only needs to handle
    it not matching the target tool's *shape*.
    """
    entry = GOOGLE_WORKSPACE_TOOLS.get(tool_name)
    if entry is None:
        logger.warning("Google Workspace: execute_google_workspace_tool called for unknown tool '%s'.", tool_name)
        return json.dumps({"error": f"Unknown Google Workspace tool '{tool_name}'."})

    func, params_model = entry
    try:
        params = params_model(**arguments)
    except ValidationError as exc:
        logger.warning("Google Workspace: invalid arguments for tool '%s': %s", tool_name, exc)
        return json.dumps({"error": f"Invalid arguments for '{tool_name}': {exc}"})

    try:
        result = await func(supabase, user_id, params)
    except Exception:
        logger.exception("Google Workspace: tool '%s' raised unexpectedly for user %s.", tool_name, user_id)
        return json.dumps({"error": f"Google Workspace tool '{tool_name}' failed unexpectedly."})

    return json.dumps(result)
