import asyncio
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import AsyncGenerator, Callable, Awaitable, List, Optional
from fastapi import HTTPException
from supabase import Client
from api.chat.schemas import Message, ProviderEnum
from providers.factory import ProviderFactory
from providers.base import BaseProvider
from api.chat.tools import ALL_TOOLS, ALL_TOOL_NAMES, execute_tool, repair_tool_arguments
from api.documents.schemas import RetrievedChunk
from api.documents.services import format_retrieved_chunks, get_relevant_context
from api.memory.services import format_memory_for_prompt, get_user_memory
from mcp_integration.mcp_manager import (
    mcp_manager,
    get_merged_tool_schemas,
    mcp_tool_to_native_schema,
    filter_workspace_tools,
    LEGACY_GOOGLE_WORKSPACE_CONNECTOR_ID,
)
from mcp_integration.intent_router import select_relevant_tools

logger = logging.getLogger(__name__)

# Every tool this app knows how to execute, in the order offered to the
# model. Sourced from api.chat.tools.ALL_TOOLS (the single registry) rather
# than listed again here, so a tool added to tools.py is automatically
# offered without a second place to remember to update it.
ACTIVE_TOOLS = list(ALL_TOOLS)

def _is_transient_tool_formatting_error(err_msg: str) -> bool:
    """
    Matches the provider error strings that indicate a flaky/malformed tool
    call emission or an invalid tool-call attempt against the current
    request's tool set (Groq/Llama occasionally misformats a function call,
    or tries to call a tool that wasn't offered this turn -- most commonly
    on a post-tool synthesis turn, where turn_tools is deliberately empty)
    rather than a hard failure -- these are worth a single silent retry
    (with tool-calling forced off) before degrading gracefully to whatever
    text has already streamed, rather than surfacing an error to the user.
    """
    return (
        "tool call validation failed" in err_msg
        or "attempted to call tool" in err_msg
        or "which was not in request.tools" in err_msg
        or "not in request's tools" in err_msg
        or "failed to call a function" in err_msg
        or "adjust your prompt" in err_msg
        or "failed_generation" in err_msg
    )

# Single source of truth for the assistant's persona/instructions. Both the
# plain-chat path and the RAG-grounded path (ChatService._build_system_prompt)
# build on this one string rather than each hardcoding their own copy.
#
# Every tool offered to the model (see active_tools in stream_chat) is
# documented explicitly below. A prior, shorter version of this prompt only
# described search_knowledge_base, leaving the model to guess at the other
# tools' intent and — for a small tool-calling model like Groq's
# llama-3.1-8b-instant — to narrate its uncertainty ("I will use the X tool",
# "I don't know what this has to do with the provided functions") directly
# into the user-facing answer instead of just picking a tool or answering
# plainly. The anti-narration and no-tool-fits rules below exist specifically
# to close that gap.
SYSTEM_PROMPT = Message(
    role="system",
    content=(
        "You are AetherChat, a helpful AI assistant.\n\n"
        "## Answering general questions\n"
        "Answer general knowledge, technical, and conversational questions directly from your own "
        "knowledge and conversationally. Most turns need no tool at all — if none of the tools below "
        "clearly apply to the user's request, just answer normally. Never say things like 'this doesn't "
        "match the provided functions' — if no tool fits, that simply means the answer doesn't require one.\n\n"
        "## Available tools\n"
        "- list_documents: use ONLY when the user asks which files/documents exist (e.g. 'what documents "
        "do you have access to', 'what have I uploaded'). It returns filenames only, never their content — "
        "never call it to answer a question about what is inside a document.\n"
        "- search_knowledge_base: use for ANY question about the content of the user's uploaded documents — "
        "facts, topics, projects, skills, dates, or anything found INSIDE a document rather than just its "
        "filename (e.g. 'according to my CV', 'summarize my report', 'what projects are in my CV'). If the "
        "question is about what's written in a document rather than which documents exist, this is the "
        "right tool, not list_documents.\n"
        "- duckduckgo_search: always use this for real-time external events, breaking news, live prices, "
        "scores, or a highly niche/conflicting domain acronym you're not confident about — if the answer "
        "depends on current, real-world information rather than your own training, call it rather than "
        "guessing. Do not call it for foundational computer science concepts, standard definitions, or "
        "general world facts already in your own knowledge; see the Epistemic humility rule below for the "
        "ambiguous case.\n"
        "- calculator: use only for complex multi-step arithmetic, equation solving, or financial/"
        "statistical formulas. Never use it for factual trivia, sports rules or player counts, general "
        "knowledge, simple single-step counting, or dates just because the answer happens to be a number.\n"
        "- get_current_time: use only when the user asks for the current date/time in some location.\n"
        "- get_weather: always use this when the user asks for current weather/temperature/conditions in a "
        "city — never guess or answer from general climate knowledge instead.\n"
        "- search_chat_history: use only when the user explicitly asks what was said, discussed, or worked "
        "on in past conversation sessions.\n"
        "- create_task: use when the user asks you to remind them of something or add/track a task or "
        "to-do item. Never use it for questions about existing tasks.\n"
        "- list_tasks: use when the user asks what tasks/reminders they have, pending or completed.\n"
        "- complete_task: use when the user says they finished or want to check off a specific task. Pass "
        "the task's id if you already have it from an earlier list_tasks call; otherwise just pass the "
        "task's title/name directly -- an exact id lookup isn't required.\n\n"
        "## CRITICAL TOOL ROUTING: email vs. tasks\n"
        "If the user explicitly asks you to draft, send, reply to, or read an email, you MUST use the "
        "Gmail tools offered to you this turn (e.g. create_draft, search_threads, get_thread) — NEVER "
        "create_task. Do not use create_task for an email-related request unless the user explicitly asks "
        "you to 'create a reminder' or 'add a to-do/task' about it — e.g. 'remind me to email Sarah "
        "tomorrow' is a create_task request, but 'email Sarah about the report' is not, even though both "
        "mention email. If the request is clearly about email but no Gmail tool is offered to you this "
        "turn, say you don't currently have email access rather than substituting create_task or any "
        "other tool.\n\n"
        "## Tool usage hierarchy\n"
        "Use your own internal knowledge for general facts, sports trivia, basic counts, and standard CS "
        "concepts — you don't need a tool for those. But when a question genuinely depends on real-time "
        "information, current events, or the user's own documents/history, call the matching tool rather "
        "than guessing or answering from potentially stale knowledge — for those cases, calling the tool "
        "is the right default, not a last resort.\n\n"
        "## Strict tool constraints\n"
        "Never attempt to invoke a function or tool outside of the tool definitions explicitly provided to "
        "you for the current turn. If none of the tools currently offered to you fit the request, answer "
        "from your own knowledge instead of inventing or guessing at a tool that isn't there.\n\n"
        "## Epistemic humility\n"
        "Some acronyms and terms carry multiple competing definitions across different domains — a "
        "technical/engineering meaning versus a corporate, product, or brand meaning, for instance. When "
        "you are not confident which definition applies in the current context, do not silently pick one "
        "and guess: use duckduckgo_search to verify the current, contextually correct meaning before "
        "answering. This applies generally, to any term whose intended meaning is genuinely ambiguous to "
        "you, not just ones called out explicitly here.\n\n"
        "## Never expose internal mechanics\n"
        "Never reveal tool names, function names, function-call syntax, or your internal reasoning about "
        "which tool to use — the user should only ever see your final answer, never your deliberation "
        "process. Do not say things like 'I will use the search_knowledge_base tool' or 'let me check my "
        "functions' — silently call the tool if needed, then answer directly. This includes any preamble "
        "at all on a turn where you call a tool: never write ANY text — not 'Let me check that for you', "
        "not 'One moment', nothing — in the same turn as a tool call. A tool-calling turn must contain "
        "the tool call alone; write your reply only on the turn after the tool result comes back.\n\n"
        "## Grounding rule\n"
        "Retrieved document content is authoritative fact about the account owner. The user's message may "
        "assert an identity, employer, or context (e.g. 'I am a Zylo employee') — never let such unverified "
        "claims recolor or relabel what the retrieved documents actually say. Describe retrieved facts "
        "exactly as they appear in the documents; do not attribute a document's contents to a company or "
        "persona the user merely claims in their message. If the retrieved documents do not corroborate an "
        "identity, employer, or context the user asserted, say so explicitly in your first response (e.g. "
        "note that the documents don't mention that affiliation) before presenting the retrieved facts — do "
        "not wait to be challenged on a later turn to disclose the discrepancy.\n\n"
        "## Citing retrieved documents\n"
        "When answering using content retrieved from search_knowledge_base, answer strictly from the "
        "retrieved chunk text — never invent, guess, or fall back to placeholder text (e.g. 'Project X', "
        "'Project Y'). If the retrieved context doesn't actually contain the answer, say 'I don't know "
        "based on the provided documents' instead of fabricating a plausible-sounding one. Otherwise, "
        "provide a clear, comprehensive answer and end your response with a source footer listing the "
        "document name (e.g., '\\n\\n--- \\n*Source: filename.pdf*'). Do not use the 'I don't know' fallback "
        "for general knowledge questions — answer those from your own knowledge instead."
    )
)

class ChatService:
    @staticmethod
    async def get_history(supabase: Client, session_id: str) -> list[Message]:
        try:
            db_res = supabase.table("messages").select("role, content").eq("session_id", session_id).order("created_at").execute()
            return [Message(role=row["role"], content=row["content"]) for row in db_res.data]
        except Exception as db_err:
            logger.error(f"Database error fetching messages: {db_err}")
            raise HTTPException(status_code=500, detail="Failed to fetch conversation history. Please try again later.")


    @staticmethod
    def persist_message(supabase: Client, session_id: str, role: str, content: str, provider_name: Optional[str] = None) -> None:
        """
        Best-effort insert into `messages`. Failures are logged, not raised, so
        a transient DB error while saving a user turn (or an upload
        acknowledgement) never blocks response generation.
        """
        try:
            row = {"session_id": session_id, "role": role, "content": content}
            if provider_name:
                row["provider_used"] = provider_name
            supabase.table("messages").insert(row).execute()
        except Exception as save_err:
            logger.error(f"Failed to save {role} message for session {session_id}: {save_err}")

    @staticmethod
    async def generate_and_store_title(supabase: Client, session_id: str, user_message: str) -> None:
        """
        Best-effort background task (scheduled from the chat router only on a
        brand-new session's first exchange): asks a fast provider for a
        concise 3-5 word title summarizing the opening exchange, then writes
        it onto chat_sessions.title so the sidebar's initial "first N chars
        of the question" placeholder gets replaced with something closer to
        Gemini/ChatGPT's dynamic titling. Always uses Groq regardless of the
        provider the user picked for the conversation itself -- titling is a
        cheap, latency-insensitive side task that shouldn't depend on
        whichever (possibly slower/pricier) provider is driving the actual
        chat. Never raises: a failure here must never surface to the user or
        affect the conversation, only leave the placeholder title in place.
        """
        try:
            assistant_res = (
                supabase.table("messages")
                .select("content")
                .eq("session_id", session_id)
                .eq("role", "assistant")
                .order("created_at")
                .limit(1)
                .execute()
            )
            assistant_message = assistant_res.data[0]["content"] if assistant_res.data else ""
            # Strip the embedded sources tag (see stream_chat's finally block)
            # before feeding this into the titling prompt -- it's raw JSON,
            # not conversation content.
            assistant_message = re.sub(r"<aether-sources>[\s\S]*?</aether-sources>", "", assistant_message).strip()

            title_provider = ProviderFactory.get_provider("groq")
            title_prompt = [
                Message(
                    role="system",
                    content=(
                        "Generate a short chat title summarizing the conversation below. "
                        "Respond with ONLY the title itself: 3-5 words, no quotes, no punctuation "
                        "at the end, no preamble."
                    ),
                ),
                Message(
                    role="user",
                    content=f"User: {user_message[:500]}\nAssistant: {assistant_message[:500]}",
                ),
            ]

            title = ""
            async for chunk in title_provider.stream_response(title_prompt, tools=None):
                if isinstance(chunk, str):
                    title += chunk

            title = title.strip().strip('"').strip("'").strip()
            if not title:
                return
            if len(title) > 60:
                title = title[:60].rstrip()

            supabase.table("chat_sessions").update({"title": title}).eq("id", session_id).execute()
        except Exception as title_err:
            logger.warning(f"Failed to auto-generate title for session {session_id}: {title_err}")

    @staticmethod
    def get_provider(provider_enum: ProviderEnum) -> BaseProvider:
        try:
            return ProviderFactory.get_provider(provider_enum.value)
        except ValueError as val_err:
            raise HTTPException(status_code=400, detail=str(val_err))
        except Exception as init_err:
            logger.error(f"Failed to initialize provider {provider_enum.value}: {init_err}")
            raise HTTPException(status_code=500, detail="AI Provider Initialization Error. Please check your configuration.")

    @staticmethod
    async def _build_memory_block(supabase: Client, user_id: Optional[str]) -> str:
        """
        Fetches the single, continuously-evolving narrative profile stored in
        `user_memory` for `user_id` (Gemini-style global memory, independent
        of session_id) and renders it as a system-prompt block. Runs on every
        turn, not just when a file is attached, since this profile is meant
        to be available across all of the user's chats. Retrieval failures
        (no profile yet, DB hiccup) are swallowed and degrade to "no memory
        this turn" -- memory must never be a point of failure for chat, same
        contract as document retrieval below.
        """
        if not user_id:
            return ""

        try:
            profile = await get_user_memory(supabase, user_id)
        except Exception as mem_err:
            logger.warning(f"User memory retrieval failed, continuing without it: {mem_err}")
            return ""

        narrative = format_memory_for_prompt(profile)
        if not narrative:
            return ""

        return "\n\n<user_memory>\n" + narrative + "\n</user_memory>"

    @staticmethod
    async def _fetch_document_context(
        supabase: Client,
        user_id: Optional[str],
        session_id: str,
        user_message: str,
        document_name: Optional[str],
    ) -> List[RetrievedChunk]:
        """
        BRANCH A ONLY (a file was attached in this request, so `document_name`
        is set): retrieves that document's chunks -- scoped to both `user_id`
        and `session_id` (Context Isolation) -- so generation is
        deterministically grounded in the just-uploaded file.

        The underlying vector search issues a Voyage AI embedding call and a
        Supabase RPC round trip, either of which can fail outright (rate
        limit, network error) or stall and get reset by an intermediary --
        observed in production as an HTTP/2 StreamReset while querying
        session-scoped chunks, which previously escaped as an unhandled 500
        on the chat endpoint. Both failure modes are caught here and degrade
        to an empty context (with a logged warning) rather than propagating
        -- retrieval must never crash the chat stream.
        """
        if not (user_id and document_name and user_message.strip()):
            return []

        try:
            return await get_relevant_context(supabase, user_message, user_id, session_id, document_name=document_name)
        except Exception as context_err:
            logger.warning(f"Document context retrieval failed, continuing without it: {context_err}")
            return []

    @staticmethod
    def _build_temporal_block() -> str:
        """
        Live "what time is it right now" context. Computed fresh on every
        call -- never cached at import/app-startup -- so relative phrasing
        in a user's request ("remind me tomorrow at 5pm", "in an hour") has
        an exact, per-request reference point to resolve against, rather
        than drifting stale for the lifetime of the server process.
        """
        now = datetime.now(timezone.utc)
        friendly = now.strftime("%A, %B %d, %Y at %I:%M %p UTC")
        iso = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        return (
            "\n\n<current_time>\n"
            f"The current date and time is {friendly} ({iso}). Use this exact reference point to "
            "compute all relative dates and times for tasks and reminders.\n"
            "</current_time>"
        )

    @staticmethod
    async def _build_system_prompt(
        supabase: Client,
        user_id: Optional[str],
        session_id: str,
        user_message: str,
        document_name: Optional[str] = None,
    ) -> Message:
        """
        Builds this turn's system prompt as SYSTEM_PROMPT plus:

        1. Current-time context (see _build_temporal_block): always
           injected, computed fresh per request.

        2. User memory (see _build_memory_block): the user's continuous
           narrative profile, fetched on every turn regardless of whether a
           file is attached.

        3. Document context (see _fetch_document_context), BRANCH A ONLY (a
           file was attached in this request): injected so generation is
           deterministically grounded in the just-uploaded file. BRANCH B (no
           file attached) does NOT fetch document context here; retrieval in
           that case is agentic, only happening if the model itself calls
           the `search_knowledge_base` tool (see api.chat.tools), so a plain
           conversational turn never pays for an unnecessary Voyage/Supabase
           round trip.

        Blocks 2 and 3 are fetched concurrently via asyncio.gather rather
        than sequentially -- the user-memory DB read and the document vector
        search don't depend on each other's results, so running them one
        after another only adds latency before the first token can stream.
        Both swallow their own retrieval failures so grounding stays
        strictly additive, never a point of failure for chat.

        Never mutates the shared SYSTEM_PROMPT singleton, since that object
        is reused across every concurrent request/user -- a fresh Message is
        built every call instead (the temporal block alone differs turn to
        turn, so there is no "nothing changed" case to fast-path anymore).
        """
        memory_block, chunks = await asyncio.gather(
            ChatService._build_memory_block(supabase, user_id),
            ChatService._fetch_document_context(supabase, user_id, session_id, user_message, document_name),
        )

        additions = ChatService._build_temporal_block() + memory_block
        if chunks:
            context_blocks = format_retrieved_chunks(chunks)
            additions += f"\n\n<knowledge_base_context>\n{context_blocks}\n</knowledge_base_context>"

        return Message(role="system", content=f"{SYSTEM_PROMPT.content}{additions}")

    @staticmethod
    async def stream_chat(
        provider_instance: BaseProvider,
        history: list[Message],
        supabase: Client,
        session_id: str,
        provider_name: str,
        request_is_disconnected: Callable[[], Awaitable[bool]],
        user_id: Optional[str] = None,
        scoped_document_name: Optional[str] = None,
        enabled_connectors: Optional[List[str]] = None,
        google_access_token: Optional[str] = None,
        github_access_token: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        # Debug instrumentation: confirms exactly what the frontend actually
        # sent this turn for connector gating, before any of the
        # enabled_connectors/token logic below runs -- never logs the raw
        # token values themselves, only whether one was present, since these
        # are live OAuth access tokens. If enabled_connectors here doesn't
        # match what the connector toggle UI shows as "on", the bug is in
        # the frontend's FormData construction (src/app/page.tsx's
        # handleSendMessage) or the /api/chat multipart parsing
        # (api/chat/routers.py), not in mcp_manager.
        logger.error(
            "CHAT REQUEST START: session=%s user=%s enabled_connectors=%r "
            "google_access_token=%s github_access_token=%s scoped_document_name=%r",
            session_id, user_id, enabled_connectors,
            "present" if google_access_token else "absent",
            "present" if github_access_token else "absent",
            scoped_document_name,
        )

        search_results_json = []
        latest_user_message = next((m.content for m in reversed(history) if m.role == "user" and m.content), "")

        # BRANCH A (a file attached) never offers any tool this turn -- see
        # below -- so a Workspace/GitHub session is only ever worth opening
        # for BRANCH B, where a tool might actually get called. Passing None
        # here is exactly what create_workspace_session/create_github_session
        # treat as "don't connect at all" (mcp_integration.mcp_manager), so
        # BRANCH A never pays for the extra SSE round trip even if the
        # request did carry a token.
        effective_google_token = None if scoped_document_name else google_access_token
        effective_github_token = None if scoped_document_name else github_access_token

        # Gates the server-side refresh-token fallback (mcp_manager
        # .create_workspace_session / _refresh_google_access_token): only
        # worth a real network round trip to Google's token endpoint when
        # this turn could actually use the result -- i.e. not BRANCH A (see
        # effective_google_token above), and only when the user actually
        # toggled a Google connector on this turn. enabled_connectors=None
        # (nothing sent -- an old caller, or backward-compat default) offers
        # every connector, so it's treated as "requested" here too, same as
        # get_merged_tool_schemas' own None contract.
        google_workspace_requested = enabled_connectors is None or any(
            c == LEGACY_GOOGLE_WORKSPACE_CONNECTOR_ID or c.startswith("google_") for c in enabled_connectors
        )
        refresh_supabase = supabase if (not scoped_document_name and google_workspace_requested) else None
        refresh_user_id = user_id if (not scoped_document_name and google_workspace_requested) else None

        # Per-request, per-user Google Workspace (Phase 4) and GitHub
        # (Phase 6) sessions: opened fresh for this one request, each
        # authenticated with this user's own OAuth token, and guaranteed
        # closed by their respective context managers before this `async
        # with` exits below -- on normal completion, on an exception, or if
        # the client disconnects and this generator is closed early. Never
        # touch mcp_manager's singleton `_connections` (see
        # create_workspace_session's docstring), so neither can ever leak
        # into another concurrent user's request.
        async with mcp_manager.create_workspace_session(
            effective_google_token, supabase=refresh_supabase, user_id=refresh_user_id
        ) as workspace_connection, \
                mcp_manager.create_github_session(effective_github_token) as github_connection, \
                mcp_manager.create_gmail_mcp_session(
                    supabase=refresh_supabase, user_id=refresh_user_id
                ) as gmail_mcp_connection:
            workspace_tool_names: set = set()
            github_tool_names: set = set()
            gmail_mcp_tool_names: set = set()

            if scoped_document_name:
                # BRANCH A: a file was attached in this request. Retrieval already
                # happened deterministically in _build_system_prompt below, so no
                # tool-calling decision is needed this turn — pass no tools at all.
                active_tools = []
            else:
                # BRANCH B: no file attached. Offer search_knowledge_base and
                # duckduckgo_search so the model can pull from the user's documents
                # or the live web agentically — routing between them (and deciding
                # whether either is needed at all) is left entirely to the model's
                # own tool-calling judgment, guided by SYSTEM_PROMPT. There is no
                # manual toggle: gating a tool on/off per-request based on a UI
                # flag, on top of the model's own tool selection, doubled up the
                # routing decision and made the model second-guess itself more
                # often ("tool paralysis"), not less.
                #
                # Merged fresh on every turn (rather than once at import time)
                # with whatever MCP servers are currently connected -- see
                # mcp_integration.mcp_manager.get_merged_tool_schemas -- so a
                # server that (re)connects after startup is picked up without
                # restarting the app. Reduces to native-only automatically when
                # no MCP server is connected, since the merge is a strict
                # superset of ACTIVE_TOOLS.
                #
                # enabled_connectors (from the request's per-connector toggles,
                # if any) only ever narrows which MCP connectors' tools are
                # offered -- native tools stay on regardless. None (nothing
                # sent by the frontend) offers every connected connector, the
                # same as before this option existed.
                active_tools = get_merged_tool_schemas(ACTIVE_TOOLS, mcp_manager, enabled_connectors=enabled_connectors)

                # Prevent aggressive triggering of chat history for short ambiguous inputs
                if len(latest_user_message.strip()) < 5:
                    active_tools = [t for t in active_tools if t.get("function", {}).get("name") != "search_chat_history"]

                # Google Workspace (Phase 4), GitHub (Phase 6), and Gmail MCP
                # (Phase 1) tools, from the per-request sessions opened
                # above -- entirely independent of the singleton-registered
                # static connectors merged just above. A name that's already
                # offered (native, an enabled static connector, or another
                # per-request connector merged first) wins, same "first
                # registration wins" precedent get_merged_tool_schemas
                # already applies to native-vs-static collisions. All three
                # loops share one existing_names set so a collision between
                # any of them is caught too, not just against native/static
                # tools.
                if workspace_connection is not None or github_connection is not None or gmail_mcp_connection is not None:
                    existing_names = {t["function"]["name"] for t in active_tools}

                    if workspace_connection is not None:
                        # Phase 7: narrowed to whichever granular Gmail/Docs/
                        # Calendar/Drive sub-connector(s) the frontend
                        # toggled on this turn, rather than every tool the
                        # Workspace session happens to have discovered -- see
                        # filter_workspace_tools for the exact contract
                        # (None/legacy "google_workspace" still offer
                        # everything, for backward compatibility).
                        for tool in filter_workspace_tools(workspace_connection.tools, enabled_connectors):
                            if tool.name in existing_names:
                                logger.warning(
                                    f"MCP: Google Workspace tool '{tool.name}' collides with an already-offered "
                                    "tool name; keeping the existing one."
                                )
                                continue
                            active_tools.append(mcp_tool_to_native_schema(tool))
                            workspace_tool_names.add(tool.name)
                            existing_names.add(tool.name)

                    if github_connection is not None:
                        for tool in github_connection.tools:
                            if tool.name in existing_names:
                                logger.warning(
                                    f"MCP: GitHub tool '{tool.name}' collides with an already-offered "
                                    "tool name; keeping the existing one."
                                )
                                continue
                            active_tools.append(mcp_tool_to_native_schema(tool))
                            github_tool_names.add(tool.name)
                            existing_names.add(tool.name)

                    if gmail_mcp_connection is not None:
                        # The managed Gmail MCP server advertises 21 full
                        # tool schemas (~14k tokens) -- offering all of them
                        # on every turn blew past Groq's 12,000 TPM limit
                        # and got the whole request rejected with an HTTP
                        # 413. Narrowed here to the top few tools actually
                        # relevant to this turn's message instead of a
                        # permanent static whitelist, so every tool stays
                        # reachable across turns depending on what's asked --
                        # see mcp_integration.intent_router for the scoring.
                        # A no-op for any Gmail MCP server that ever
                        # advertises 5 or fewer tools (select_relevant_tools'
                        # own short-circuit).
                        for tool in select_relevant_tools(latest_user_message, gmail_mcp_connection.tools):
                            if tool.name in existing_names:
                                logger.warning(
                                    f"MCP: Gmail MCP tool '{tool.name}' collides with an already-offered "
                                    "tool name; keeping the existing one."
                                )
                                continue
                            active_tools.append(mcp_tool_to_native_schema(tool))
                            gmail_mcp_tool_names.add(tool.name)
                            existing_names.add(tool.name)

            system_message = await ChatService._build_system_prompt(
                supabase, user_id, session_id, latest_user_message, document_name=scoped_document_name
            )

            if not history or history[0].role != "system":
                history.insert(0, system_message)
            else:
                history[0] = system_message

            full_text = ""
            has_called_tool = False
            try:
                for _ in range(5):
                    # Once a tool has already run once this request, the remaining
                    # turns are pure synthesis: the model just needs to read the
                    # tool result(s) already in `history` and write the answer, not
                    # decide whether to call yet another tool. Re-offering the full
                    # tool schema set on that turn forces a small model to keep
                    # splitting its attention between "should I call a tool" and
                    # "write the grounded answer" even when the routing decision is
                    # already made — a real contributor to synthesis turns
                    # ignoring retrieved content and fabricating placeholders.
                    turn_tools = [] if has_called_tool else active_tools
                    turn_tool_names = {t.get("function", {}).get("name") for t in turn_tools}
                    tool_calls_received = None
                    turn_text = ""
                    # Length of full_text before this turn streamed anything --
                    # lets a later retraction (see below) trim back exactly this
                    # turn's contribution without disturbing any prior turn's
                    # already-confirmed-final text.
                    full_text_len_before_turn = len(full_text)

                    # A transient tool-formatting error is retried once, but only
                    # while nothing has been streamed to the user yet this turn --
                    # retrying after partial output would duplicate visible text.
                    # This is what turns an occasional flaky generation (more
                    # likely on longer follow-up messages / longer histories,
                    # where the model has more context to misformat a call over)
                    # into a silent retry instead of an error on the first try.
                    for attempt in range(2):
                        try:
                            stream = provider_instance.stream_response(history, tools=turn_tools)
                            async for chunk in stream:
                                if await request_is_disconnected():
                                    logger.info("Client disconnected during stream. Terminating.")
                                    return

                                if isinstance(chunk, dict) and chunk.get("type") == "tool_calls":
                                    tool_calls_received = (tool_calls_received or []) + list(chunk.get("tool_calls") or [])
                                elif isinstance(chunk, dict) and chunk.get("type") == "text_tool_call":
                                    # Dual-mode fallback: the provider (currently
                                    # only Groq/llama-3.3-70b-versatile) emitted a
                                    # tool call as a literal `<tool_name>{...}</tool_name>`
                                    # text tag instead of a native tool_calls object.
                                    # The provider adapter already keeps that raw tag
                                    # out of the streamed content and only extracts
                                    # names offered to it this turn -- but this is
                                    # re-validated here against turn_tool_names
                                    # (rather than trusted blindly) so any provider
                                    # emitting this event type is held to the same
                                    # "never invoke a tool outside what was
                                    # explicitly offered this turn" rule. A valid
                                    # match is normalized into the same shape a
                                    # native call would have (with a synthetic id,
                                    # since text tags carry none) so the rest of
                                    # this loop -- execution, appending history, and
                                    # the synthesis turn -- proceeds identically to
                                    # the native tool-calling path.
                                    tag_name = chunk.get("name", "")
                                    if tag_name not in turn_tool_names:
                                        logger.warning(
                                            f"Discarding text-tag tool call for '{tag_name}': not among the "
                                            "tools offered this turn."
                                        )
                                    else:
                                        logger.warning(
                                            f"Provider emitted tool call '{tag_name}' as a text tag "
                                            "instead of a native tool_calls object; using text-tag fallback."
                                        )
                                        synthetic_call = {
                                            "id": f"text_tag_{uuid.uuid4().hex[:8]}",
                                            "type": "function",
                                            "function": {
                                                "name": tag_name,
                                                "arguments": chunk.get("arguments") or "{}",
                                            },
                                        }
                                        tool_calls_received = (tool_calls_received or []) + [synthetic_call]
                                elif isinstance(chunk, str):
                                    turn_text += chunk
                                    full_text += chunk
                                    yield f"data: {json.dumps({'content': chunk})}\n\n"
                            break
                        except Exception as turn_err:
                            err_msg = str(turn_err).lower()
                            if _is_transient_tool_formatting_error(err_msg):
                                if attempt == 0 and not turn_text:
                                    tool_calls_received = None
                                    if has_called_tool:
                                        # Includes the "attempted to call tool X
                                        # which was not in request.tools"
                                        # validation error -- this is a post-tool
                                        # synthesis turn, where turn_tools is
                                        # deliberately empty but the model still
                                        # tries to call something. Retrying with
                                        # tool-calling explicitly forced off
                                        # (rather than just repeating the same
                                        # request) is what actually resolves it
                                        # instead of reproducing the same
                                        # rejection.
                                        logger.warning(
                                            f"Tool-call validation error on synthesis turn, retrying once "
                                            f"with tool-calling forced off: {turn_err}"
                                        )
                                        turn_tools = []
                                        turn_tool_names = set()
                                    else:
                                        # This is a genuine first attempt at
                                        # calling a tool (has_called_tool is
                                        # still False), not a synthesis-turn
                                        # artifact -- most likely Llama flaking on
                                        # the call's formatting under load or a
                                        # long context. Keep turn_tools intact so
                                        # the retry gives the model a real second
                                        # chance to call the tool correctly,
                                        # instead of guaranteeing it can't by
                                        # stripping tools out from under it.
                                        logger.warning(
                                            f"Transient tool-formatting error on first attempt, retrying "
                                            f"turn once with tools still enabled: {turn_err}"
                                        )
                                    continue
                                # Either the retry above also hit a validation
                                # error, or text had already started streaming
                                # when the error occurred (so retrying would
                                # duplicate visible output). Either way this is a
                                # known, benign tool-calling hiccup, not a real
                                # failure -- degrade gracefully to whatever text
                                # has already streamed instead of surfacing an
                                # internal API validation error to the user.
                                logger.warning(
                                    f"Tool-call validation error persisted; degrading turn to plain text: {turn_err}"
                                )
                                tool_calls_received = None
                                break
                            raise

                    if tool_calls_received:
                        if turn_text.strip():
                            # This turn streamed real content before also (or
                            # instead) deciding to call a tool -- e.g. llama-
                            # 3.3-70b-versatile narrating "Let me check the
                            # weather..." ahead of the native tool_calls delta,
                            # despite SYSTEM_PROMPT forbidding it. That text has
                            # already reached the browser live (see the `elif
                            # isinstance(chunk, str)` yield above), so it can't be
                            # un-sent -- instead, tell the client to wipe it from
                            # the message bubble before the synthesis turn's real
                            # answer starts arriving, and drop it from full_text
                            # so the persisted message doesn't carry it either.
                            full_text = full_text[:full_text_len_before_turn]
                            yield f"data: {json.dumps({'retract': True})}\n\n"
                        has_called_tool = True
                        history.append(Message(role="assistant", content=turn_text if turn_text.strip() else None, tool_calls=tool_calls_received))

                        for tc in tool_calls_received:
                            tool_call_id = tc["id"]
                            tool_name = tc["function"]["name"]
                            # Tolerates dirty/truncated JSON in the accumulated
                            # arguments string (see repair_tool_arguments) instead
                            # of silently dropping every argument -- including
                            # required ones like `city` or `search_query` -- the
                            # moment a strict json.loads first fails.
                            tool_args = repair_tool_arguments(tool_name, tc["function"]["arguments"])

                            # Explicit record of the LLM's own tool choice, independent
                            # of which branch below ends up executing it -- lets a
                            # misrouting bug (e.g. create_task called for an email
                            # request) be diagnosed straight from logs: was it the model
                            # picking the wrong tool, or this router dispatching a
                            # correctly-chosen tool to the wrong connection?
                            logger.info(f"LLM invoked tool: {tool_name} with arguments: {tool_args}")

                            # Execution Router: a tool name in ALL_TOOL_NAMES is
                            # one of this app's own registered tools (see
                            # api.chat.tools) and always runs through the
                            # existing native dispatcher, unchanged. A name in
                            # workspace_tool_names/github_tool_names/
                            # gmail_mcp_tool_names came from this request's own
                            # per-user Google Workspace/GitHub/Gmail MCP
                            # session and is routed through that same
                            # temporary, authenticated connection -- never
                            # through the shared singleton manager, which has
                            # no knowledge of any of these sessions at all.
                            # Anything else was offered because
                            # get_merged_tool_schemas added it from a
                            # singleton-registered static MCP connector (a
                            # shared-PAT GitHub, Brave Search, ...), so it's
                            # routed through mcp_manager instead. The model
                            # itself never needs to know which kind of tool it
                            # called.
                            if tool_name in ALL_TOOL_NAMES:
                                logger.info(f"Executing Native Tool: {tool_name}")
                                result_str = await execute_tool(tool_name, tool_args, supabase, session_id, user_id)
                            elif tool_name in workspace_tool_names:
                                logger.info(f"Executing Google Workspace Tool: {tool_name}")
                                result_str = await workspace_connection.call_tool(tool_name, tool_args)
                            elif tool_name in github_tool_names:
                                logger.info(f"Executing per-user GitHub Tool: {tool_name}")
                                result_str = await github_connection.call_tool(tool_name, tool_args)
                            elif tool_name in gmail_mcp_tool_names:
                                logger.info(f"Executing Gmail MCP Tool: {tool_name}")
                                result_str = await gmail_mcp_connection.call_tool(tool_name, tool_args)
                            else:
                                logger.info(f"Executing MCP Tool: {tool_name}")
                                result_str = await mcp_manager.call_tool(tool_name, tool_args)

                            if tool_name == "duckduckgo_search":
                                try:
                                    parsed = json.loads(result_str)
                                    if isinstance(parsed, list):
                                        search_results_json.extend(parsed)
                                        yield f"data: {json.dumps({'sources': parsed})}\n\n"
                                except Exception:
                                    pass

                            history.append(Message(
                                role="tool",
                                content=result_str,
                                tool_call_id=tool_call_id,
                                name=tool_name
                            ))
                    else:
                        break

                if not await request_is_disconnected():
                    yield "data: [DONE]\n\n"

            except Exception as stream_err:
                logger.error(f"Error during stream generation: {stream_err}")
                err_msg = str(stream_err).lower()
                if "tool call validation failed" in err_msg or "attempted to call tool" in err_msg:
                    user_friendly_err = "I am unable to process that tool request right now. Please try rephrasing your message."
                elif "failed to call a function" in err_msg or "adjust your prompt" in err_msg:
                    user_friendly_err = "I had trouble formatting a tool call for that request. Please try rephrasing your message."
                elif "rate_limit_exceeded" in err_msg or "429" in err_msg:
                    user_friendly_err = "The AI provider is currently experiencing high traffic. Please try again in a moment."
                elif _is_transient_tool_formatting_error(err_msg):
                    user_friendly_err = "I'm having trouble processing that request right now. Please try rephrasing."
                else:
                    user_friendly_err = "An unexpected error occurred while communicating with the AI provider. Please try again."
                yield f"data: {json.dumps({'error': user_friendly_err})}\n\n"
            finally:
                if full_text.strip():
                    if search_results_json:
                        # Embed sources for zero-migration UI rendering on fetch
                        full_text += f"\n\n<aether-sources>{json.dumps(search_results_json)}</aether-sources>"
                    try:
                        supabase.table("messages").insert({
                            "session_id": session_id,
                            "role": "assistant",
                            "content": full_text,
                            "provider_used": provider_name
                        }).execute()
                        logger.info(f"Successfully saved assistant response for session {session_id}")
                    except Exception as save_err:
                        logger.error(f"Failed to save assistant message to DB: {save_err}")
