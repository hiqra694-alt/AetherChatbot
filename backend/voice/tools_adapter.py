"""
Voice agent tool adapter (Phase 2): thin @function_tool wrappers that let
the voice agent's LLM call the exact same retrieval, native-tool, and MCP
execution logic the text chat pipeline already uses -- no business logic is
reimplemented here, only argument pass-through.

  - Retrieval: api.documents.services.get_relevant_context, the same hybrid
    (Supabase RPC + local BM25/cross-encoder rerank) retrieval function
    api/chat/services.py calls to ground text chat responses.
  - Native tools: every tool in api.chat.tools.ALL_TOOLS --
    calculator/get_current_time/get_weather/search_chat_history (BASE_TOOLS)
    plus duckduckgo_search/search_knowledge_base/list_documents/
    create_task/list_tasks/complete_task -- the exact functions text chat
    wraps for the same capabilities, called directly here by name instead
    of being reimplemented.
  - MCP execution: mcp_integration.mcp_manager's singleton `mcp_manager`,
    the same MCPClientManager api/chat/services.py routes generic MCP tool
    calls through (see its "Executing MCP Tool" branch).

Native vs. MCP routing mirrors api/chat/services.py's own dispatcher (see
its "Execution Router" comment): a tool name in api.chat.tools.ALL_TOOL_NAMES
is one of *this app's* own tools and always runs natively; anything else is
assumed to come from a connected MCP server. The difference here is *where*
that distinction is made. Text chat exposes each tool (native or MCP) to the
model under its own real name via get_merged_tool_schemas, so the model
always calls a concrete, correctly-routed name and the router only ever sees
names the model was actually offered. Voice instead exposes one generic
execute_mcp_tool(tool_name, ...) escape hatch for MCP tools (LiveKit's
Toolset gives every @function_tool method its own static schema, not a
per-session dynamically-merged one, so mirroring get_merged_tool_schemas
exactly isn't a small change) -- which previously meant a native capability
voice had no dedicated tool for (e.g. weather) could only be reached, if the
model reached for it at all, by guessing a plausible-sounding tool_name and
routing it through that MCP-only escape hatch, where mcp_manager correctly
had no connected server advertising it and rejected the call. Fixed on both
sides: every BASE_TOOLS capability now has its own real native @function_tool
here (so the model has a correct, direct way to call it), and
execute_mcp_tool itself now rejects any tool_name that's actually native
before ever reaching mcp_manager (see its docstring below).

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
from typing import Optional

from livekit.agents import function_tool
from livekit.agents.llm import Toolset
from supabase import Client

from api.chat.tools import ALL_TOOL_NAMES
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
from mcp_integration.mcp_manager import mcp_manager

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

    @function_tool
    async def execute_mcp_tool(self, tool_name: str, arguments_json: str) -> str:
        """Execute a tool exposed by a connected MCP server (e.g. GitHub) by its
        exact name, passing through the arguments it expects. Only for tools
        provided by an external MCP connector -- never call this for this
        assistant's own built-in capabilities (weather, time, calculator,
        chat history, knowledge base search, task management), which each
        have their own dedicated tool above; call those directly by name
        instead.

        Args:
            tool_name: The exact name of the MCP tool to execute.
            arguments_json: The arguments to pass to the MCP tool, encoded as a
                JSON object string (e.g. '{"owner": "foo", "repo": "bar"}', or
                '{}' if the tool takes no arguments).
        """
        if tool_name in ALL_TOOL_NAMES:
            logger.warning(
                "Voice: execute_mcp_tool called with native tool name '%s' -- "
                "this assistant has a dedicated tool for that; call it directly instead.",
                tool_name,
            )
            return json.dumps({
                "error": (
                    f"'{tool_name}' is one of this assistant's own built-in tools, not an "
                    "MCP tool. Call it directly by its own name instead of through execute_mcp_tool."
                )
            })
        # arguments is typed as a JSON-encoded string, not `dict`, because MCP
        # tools accept arbitrary/unknown keys -- an open-ended object can't be
        # expressed under OpenAI strict-mode tool schemas, which require
        # `additionalProperties: false` on every object in the schema (see
        # livekit.agents.llm._strict.to_strict_json_schema). Pydantic renders
        # `dict`/`Dict[str, Any]` params with `additionalProperties: true`
        # already set, so livekit's strict-schema pass (which only fills in
        # `additionalProperties` when the key is absent) leaves it as `true`
        # and the strict-mode LLM endpoint rejects the tool schema outright.
        # A plain `str` param has no such object to close over, sidestepping
        # the conflict entirely; we decode it back to a dict here instead.
        try:
            arguments = json.loads(arguments_json)
        except (TypeError, ValueError):
            logger.warning("Voice: execute_mcp_tool got non-JSON arguments_json: %r", arguments_json)
            return json.dumps({"error": "arguments_json must be a valid JSON object string."})

        if not isinstance(arguments, dict):
            return json.dumps({"error": "arguments_json must decode to a JSON object."})

        return await mcp_manager.call_tool(tool_name, arguments)
