"""
Voice agent tool adapter (Phase 2): thin @function_tool wrappers that let
the voice agent's LLM call the exact same retrieval and native-tool logic
the text chat pipeline already uses -- no business logic is reimplemented
here, only argument pass-through.

  - Retrieval: api.documents.services.get_relevant_context, the same hybrid
    (Supabase RPC + local BM25/cross-encoder rerank) retrieval function
    api/chat/services.py calls to ground text chat responses.
  - Native tools: every tool in api.chat.tools.ALL_TOOLS --
    calculator/get_current_time/get_weather/search_chat_history (BASE_TOOLS)
    plus duckduckgo_search/search_knowledge_base/list_documents/
    create_task/list_tasks/complete_task -- the exact functions text chat
    wraps for the same capabilities, called directly here by name instead
    of being reimplemented.
  - Google Workspace: the 8 native Gmail/Calendar/Drive functions in
    connector_integrations.google_tools, each exposed here as its own
    dedicated @function_tool (same one-per-capability shape as the
    BASE_TOOLS above) that forwards straight to
    google_tools.execute_google_workspace_tool for validation + dispatch --
    no argument handling is reimplemented here, same contract as every
    other tool in this file.

No MCP execution path here: this Toolset used to also expose a generic
execute_mcp_tool(tool_name, ...) escape hatch routed through
connector_integrations.connector_manager's singleton `mcp_manager` (the same
MCPClientManager api/chat/services.py's "Executing MCP Tool" branch still
uses for text chat). It's been removed now that every capability voice
actually needs -- native tools and Google Workspace alike -- has its own
dedicated @function_tool with a real, LLM-visible schema (LiveKit's Toolset
gives each @function_tool method its own static schema, not a per-session
dynamically-merged one the way text chat's get_merged_tool_schemas works,
so there was no way to keep that escape hatch generic without the model
having to guess a plausible-sounding tool_name blind). `mcp_manager` itself
is untouched and still connects to whatever static MCP servers are
configured (e.g. a generic MCP_SERVER_URLS entry like Brave Search) --
text chat still routes to it; only this Toolset's bridge into it is gone.
A future voice-facing MCP integration should get its own dedicated
@function_tool here, the same way each Google Workspace tool did, rather
than reintroducing a generic passthrough.

Built on livekit.agents.llm.Toolset rather than the older ai_callable/
FunctionContext pattern: as installed here (livekit-agents==1.6.10), that
API no longer exists (`llm.ai_callable` and `llm.FunctionContext` are both
gone) -- `@function_tool` on a Toolset subclass's methods, auto-collected
via Toolset.__init__, is the current equivalent, and Agent(tools=[...])
accepts a Toolset instance directly.

`supabase`/`user_id`/`session_id` scope every retrieval call the same way
api/chat/services.py scopes it for text chat. They can't be constructor
arguments -- voice/agent.py must be able to build `VoiceTools()` before a
LiveKit participant (and therefore a user identity) is known -- so they're
set once per job via bind_context(), after which every call is scoped to
that caller's own documents.
"""

import json
import logging
from typing import List, Optional

from livekit.agents import function_tool
from livekit.agents.llm import Toolset
from supabase import Client

from api.chat.tools import calculator as _calculator
from api.chat.tools import complete_task as _complete_task
from api.chat.tools import create_task as _create_task
from api.chat.tools import duckduckgo_search as _duckduckgo_search
from api.chat.tools import get_current_time as _get_current_time
from api.chat.tools import get_weather as _get_weather
from api.chat.tools import list_documents as _list_documents
from api.chat.tools import list_tasks as _list_tasks
from api.chat.tools import search_chat_history as _search_chat_history
from api.documents.services import format_retrieved_chunks, get_relevant_context
from connector_integrations.google_tools import execute_google_workspace_tool

logger = logging.getLogger(__name__)


class VoiceTools(Toolset):
    def __init__(self) -> None:
        super().__init__(id="voice_tools")
        self._supabase: Optional[Client] = None
        self._user_id: Optional[str] = None
        self._session_id: Optional[str] = None

    def bind_context(self, supabase: Optional[Client], user_id: str, session_id: str) -> None:
        """Called once from voice/agent.py's entrypoint after the LiveKit
        participant (and therefore the authenticated user_id) is known."""
        self._supabase = supabase
        self._user_id = user_id
        self._session_id = session_id

    @function_tool
    async def search_knowledge_base(self, query: str) -> str:
        """Search the user's uploaded documents and knowledge base for context
        relevant to a query. Call this before answering questions that depend
        on the user's own documents or prior conversation.

        Args:
            query: The search query to retrieve relevant document context for.
        """
        if not (self._supabase and self._user_id and self._session_id):
            logger.warning("Voice: search_knowledge_base called with no session context bound.")
            return json.dumps({"error": "Knowledge base retrieval is unavailable for this session."})

        chunks = await get_relevant_context(self._supabase, query, self._user_id, self._session_id)
        return format_retrieved_chunks(chunks)

    @function_tool
    async def list_documents(self) -> str:
        """Lists the exact names of every document the user has uploaded to
        their knowledge base. Call this when the user asks what documents
        you have access to, or what they've uploaded -- not for questions
        about what's inside a document, which search_knowledge_base answers
        instead.
        """
        if not (self._supabase and self._user_id):
            return json.dumps({"error": "No authenticated user to scope the document list to."})
        return await _list_documents(self._supabase, self._user_id, self._session_id or "")

    @function_tool
    async def duckduckgo_search(self, search_query: str) -> str:
        """Searches the live web. Defines an epistemic boundary: only call
        this when your own knowledge is genuinely insufficient or
        unreliable for the question, not as a default first step.

        WHEN TO USE: real-time external events, breaking news, live
        weather, prices, scores, or anything that can change after your
        training cutoff; or a highly niche/conflicting domain acronym or
        term where you are not confident which of several plausible
        meanings applies. WHEN NOT TO USE: foundational concepts,
        well-established definitions, or general world facts already in
        your own knowledge -- answer those directly instead.

        Args:
            search_query: The optimized search query to look up on the web.
        """
        return await _duckduckgo_search(search_query)

    @function_tool
    async def get_weather(self, city: str, unit: str = "celsius") -> str:
        """Retrieves real-time weather and temperature for a given city.
        Call this when the user explicitly asks for current weather,
        temperature, or conditions in a city -- not for general climate
        facts, seasons, or historical weather trivia.

        Args:
            city: The name of the city to get weather for.
            unit: The unit for temperature ('celsius' or 'fahrenheit'). Default is 'celsius'.
        """
        return await _get_weather(city, unit)

    @function_tool
    async def get_current_time(self, timezone: str = "UTC") -> str:
        """Fetches the current date and time for a given IANA timezone or
        city name. Call this when the user explicitly asks what the current
        date/time is somewhere -- not for historical dates or general date
        arithmetic that doesn't depend on right now.

        Args:
            timezone: The city name or IANA timezone string (e.g. 'Tokyo', 'London', 'America/New_York'). Default is 'UTC'.
        """
        return await _get_current_time(timezone)

    @function_tool
    async def calculator(self, expression: str) -> str:
        """Evaluates a mathematical expression. Call this for complex
        multi-step arithmetic or precise computation -- not for factual
        trivia, simple counting, or dates, even when the answer happens to
        be a number.

        Args:
            expression: A mathematical expression using numbers and basic operators (+, -, *, /, (), .).
        """
        return await _calculator(expression)

    @function_tool
    async def search_chat_history(self, query: str = "", limit: int = 5) -> str:
        """Searches past chat history across the user's sessions. Call this
        when the user asks what was said, discussed, or worked on in past
        conversations -- not for general capability questions or greetings.

        Args:
            query: Optional keyword/topic to search for. Leave empty to retrieve recent messages across past sessions.
            limit: Max number of messages to return. Default is 5.
        """
        if not self._supabase:
            return json.dumps({"result": "Chat history is currently unavailable."})
        return await _search_chat_history(self._supabase, self._session_id or "", query, limit)

    @function_tool
    async def create_task(
        self,
        title: str,
        due_at: Optional[str] = None,
        description: Optional[str] = None,
    ) -> str:
        """Creates a new task/reminder for the user, optionally with a due
        date/time. Call this when the user asks to be reminded of something,
        or to add or track a task or to-do item -- not for listing existing
        tasks or marking one done.

        Args:
            title: Short title describing the task or reminder.
            due_at: ISO 8601 UTC timestamp (YYYY-MM-DDTHH:MM:SSZ) the task is due at.
                Calculate this relative to the current date/time -- never guess a date
                without anchoring it to that reference point. Omit if there's no due date.
            description: Optional longer description or extra detail about the task.
        """
        if not (self._supabase and self._user_id):
            return json.dumps({"error": "No authenticated user to create a task for."})
        return await _create_task(self._supabase, self._user_id, title, due_at, description)

    @function_tool
    async def list_tasks(self, status: str = "pending") -> str:
        """Lists the user's tasks/reminders. Call this when the user asks
        what tasks/reminders they have, or to see their to-do list -- not
        for creating a new task or marking one done.

        Args:
            status: Filter by status: 'pending', 'completed', or 'all' for every task. Default is 'pending'.
        """
        if not (self._supabase and self._user_id):
            return json.dumps({"error": "No authenticated user to list tasks for."})
        return await _list_tasks(self._supabase, self._user_id, status)

    @function_tool
    async def complete_task(self, task_id: str) -> str:
        """Marks an existing task as completed. Call this when the user says
        they finished, did, or want to check off a specific task. Accepts
        either the task's id or just its title/name -- no need to call
        list_tasks first just to find the id.

        Args:
            task_id: The id of the task to complete, or -- if unknown -- the task's title
                or a distinctive part of it.
        """
        if not (self._supabase and self._user_id):
            return json.dumps({"error": "No authenticated user to complete a task for."})
        return await _complete_task(self._supabase, self._user_id, task_id)

    def _google_workspace_unavailable(self) -> Optional[str]:
        if not (self._supabase and self._user_id):
            return json.dumps({"error": "Google Workspace tools are unavailable for this session."})
        return None

    @function_tool
    async def gmail_search_recent(self, query: str, max_results: int = 10) -> str:
        """Search the user's Gmail for recent messages matching a query, returning
        each match's id, thread id, subject, sender, date, and snippet. Call this
        first to find a message before reading its full content or replying to
        it -- it does not return full message bodies.

        Args:
            query: Gmail search query using Gmail's own search operators, e.g.
                'from:boss@company.com is:unread', 'subject:invoice after:2024/01/01',
                or a plain keyword search.
            max_results: Maximum number of messages to return (1-50). Default is 10.
        """
        return self._google_workspace_unavailable() or await execute_google_workspace_tool(
            "gmail_search_recent", {"query": query, "max_results": max_results}, self._supabase, self._user_id
        )

    @function_tool
    async def gmail_read_thread(self, thread_id: str) -> str:
        """Read the full content of every message in a Gmail thread (subject,
        sender, date, and body text for each message), in order. Use this after
        gmail_search_recent has identified the thread the user is asking about.

        Args:
            thread_id: The Gmail thread id to read, as returned by gmail_search_recent.
        """
        return self._google_workspace_unavailable() or await execute_google_workspace_tool(
            "gmail_read_thread", {"thread_id": thread_id}, self._supabase, self._user_id
        )

    @function_tool
    async def gmail_create_draft(self, to: str, subject: str, body: str) -> str:
        """Create a draft email in the user's Gmail account without sending it.
        Use this when the user asks you to draft, prepare, or write an email for
        their review rather than send it immediately.

        Args:
            to: Recipient email address.
            subject: Email subject line.
            body: Plain-text email body.
        """
        return self._google_workspace_unavailable() or await execute_google_workspace_tool(
            "gmail_create_draft", {"to": to, "subject": subject, "body": body}, self._supabase, self._user_id
        )

    @function_tool
    async def gmail_send_email(self, to: str, subject: str, body: str) -> str:
        """Send an email immediately from the user's Gmail account. Only use
        this when the user has clearly asked you to send an email now -- prefer
        gmail_create_draft when they want to review it first.

        Args:
            to: Recipient email address.
            subject: Email subject line.
            body: Plain-text email body.
        """
        return self._google_workspace_unavailable() or await execute_google_workspace_tool(
            "gmail_send_email", {"to": to, "subject": subject, "body": body}, self._supabase, self._user_id
        )

    @function_tool
    async def calendar_list_events(
        self,
        time_min: Optional[str] = None,
        time_max: Optional[str] = None,
        max_results: int = 10,
    ) -> str:
        """List upcoming events on the user's primary Google Calendar, optionally
        bounded by a time range. Use this to check the user's schedule or find an
        event before modifying it -- always call this before telling the user
        whether they have any meetings, rather than guessing.

        Args:
            time_min: RFC3339 timestamp (e.g. '2024-06-01T00:00:00Z'). Defaults to now if omitted.
            time_max: RFC3339 timestamp. Omit for no upper bound.
            max_results: Maximum number of events to return (1-50). Default is 10.
        """
        return self._google_workspace_unavailable() or await execute_google_workspace_tool(
            "calendar_list_events",
            {"time_min": time_min, "time_max": time_max, "max_results": max_results},
            self._supabase,
            self._user_id,
        )

    @function_tool
    async def calendar_create_event(
        self,
        summary: str,
        start_time: str,
        end_time: str,
        description: Optional[str] = None,
        attendees: Optional[List[str]] = None,
        timezone: str = "UTC",
    ) -> str:
        """Create a new event on the user's primary Google Calendar, optionally
        inviting attendees. Use this when the user asks you to schedule a
        meeting or add something to their calendar.

        Args:
            summary: Event title.
            start_time: RFC3339 start timestamp, e.g. '2024-06-01T15:00:00-07:00'.
            end_time: RFC3339 end timestamp, e.g. '2024-06-01T16:00:00-07:00'.
            description: Optional event description/notes.
            attendees: Optional list of attendee email addresses to invite.
            timezone: IANA timezone name for start/end, e.g. 'America/Los_Angeles'. Default is 'UTC'.
        """
        return self._google_workspace_unavailable() or await execute_google_workspace_tool(
            "calendar_create_event",
            {
                "summary": summary,
                "start_time": start_time,
                "end_time": end_time,
                "description": description,
                "attendees": attendees,
                "timezone": timezone,
            },
            self._supabase,
            self._user_id,
        )

    @function_tool
    async def drive_search_docs(self, query: str, max_results: int = 10) -> str:
        """Search the user's Google Drive for files by name. Read-only -- use
        this to find a file's id before referencing it elsewhere; it does not
        return file content.

        Args:
            query: Text to search for in file names, e.g. 'Q3 budget'.
            max_results: Maximum number of files to return (1-50). Default is 10.
        """
        return self._google_workspace_unavailable() or await execute_google_workspace_tool(
            "drive_search_docs", {"query": query, "max_results": max_results}, self._supabase, self._user_id
        )

    @function_tool
    async def drive_create_doc(self, title: str, content: Optional[str] = None) -> str:
        """Create a new Google Doc in the user's Drive, optionally pre-filled
        with plain-text content. Use this when the user asks you to draft a
        document, notes, or a report as a Google Doc.

        Args:
            title: Title of the new Google Doc.
            content: Optional plain-text content to populate the document with.
        """
        return self._google_workspace_unavailable() or await execute_google_workspace_tool(
            "drive_create_doc", {"title": title, "content": content}, self._supabase, self._user_id
        )
