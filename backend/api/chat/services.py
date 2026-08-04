import json
import logging
from typing import AsyncGenerator, Callable, Awaitable, Optional
from fastapi import HTTPException
from supabase import Client
from api.chat.schemas import Message, ProviderEnum
from providers.factory import ProviderFactory
from providers.base import BaseProvider
from api.chat.tools import (
    BASE_TOOLS,
    DUCKDUCKGO_SEARCH_TOOL,
    LIST_DOCUMENTS_TOOL,
    SEARCH_KNOWLEDGE_BASE_TOOL,
    execute_tool,
)
from api.documents.services import format_retrieved_chunks, get_relevant_context

logger = logging.getLogger(__name__)

def _is_transient_tool_formatting_error(err_msg: str) -> bool:
    """
    Matches the provider error strings that indicate a flaky/malformed tool
    call emission (Groq/Llama occasionally misformats a function call under
    load or on longer contexts) rather than a hard failure -- these are worth
    a single silent retry before surfacing an error to the user.
    """
    return (
        "tool call validation failed" in err_msg
        or "attempted to call tool" in err_msg
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
        "- duckduckgo_search: use autonomously for current events, real-time facts, or anything you would "
        "not reliably know — decide on your own whether a question needs live web results, the same way "
        "you decide between any other tool. Do not call it for foundational concepts or general knowledge "
        "you already know with confidence; see the Epistemic humility rule below for the ambiguous case.\n"
        "- calculator: use only for explicit arithmetic expressions.\n"
        "- get_current_time: use only when the user asks for the current date/time in some location.\n"
        "- get_weather: use only when the user asks for current weather/temperature in a city.\n"
        "- search_chat_history: use only when the user explicitly asks about earlier messages or past "
        "conversations.\n\n"
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
        "functions' — silently call the tool if needed, then answer directly.\n\n"
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
    def get_provider(provider_enum: ProviderEnum) -> BaseProvider:
        try:
            return ProviderFactory.get_provider(provider_enum.value)
        except ValueError as val_err:
            raise HTTPException(status_code=400, detail=str(val_err))
        except Exception as init_err:
            logger.error(f"Failed to initialize provider {provider_enum.value}: {init_err}")
            raise HTTPException(status_code=500, detail="AI Provider Initialization Error. Please check your configuration.")

    @staticmethod
    async def _build_system_prompt(
        supabase: Client,
        user_id: Optional[str],
        user_message: str,
        document_name: Optional[str] = None,
    ) -> Message:
        """
        BRANCH A ONLY (a file was attached in this request, so `document_name`
        is set): immediately retrieves that document's chunks and injects
        them into the system prompt so generation is deterministically
        grounded in the just-uploaded file — no model decision needed.

        BRANCH B (no file attached, `document_name` is None): returns the
        base SYSTEM_PROMPT unchanged and does NOT call get_relevant_context.
        Retrieval in that case is agentic — it only happens if the model
        itself decides to call the `search_knowledge_base` tool (see
        api.chat.tools.search_knowledge_base), so a plain conversational
        turn never pays for an unnecessary Voyage/Supabase round trip.

        Never mutates the shared SYSTEM_PROMPT singleton, since that object
        is reused across every request. Retrieval failures (no documents
        yet, Voyage/RPC errors) are swallowed and fall back to the base
        prompt so RAG grounding is strictly additive, never a point of
        failure for chat.
        """
        if not user_id or not document_name or not user_message.strip():
            return SYSTEM_PROMPT

        try:
            chunks = await get_relevant_context(supabase, user_message, user_id, document_name=document_name)
        except Exception as context_err:
            logger.warning(f"Document context retrieval failed, continuing without it: {context_err}")
            return SYSTEM_PROMPT

        if not chunks:
            return SYSTEM_PROMPT

        context_blocks = format_retrieved_chunks(chunks)
        grounded_content = (
            f"{SYSTEM_PROMPT.content}\n\n"
            "<knowledge_base_context>\n"
            f"{context_blocks}\n"
            "</knowledge_base_context>"
        )
        return Message(role="system", content=grounded_content)

    @staticmethod
    async def stream_chat(
        provider_instance: BaseProvider,
        history: list[Message],
        supabase: Client,
        session_id: str,
        provider_name: str,
        request_is_disconnected: Callable[[], Awaitable[bool]],
        user_id: Optional[str] = None,
        scoped_document_name: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        search_results_json = []
        latest_user_message = next((m.content for m in reversed(history) if m.role == "user" and m.content), "")

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
            active_tools = list(BASE_TOOLS) + [SEARCH_KNOWLEDGE_BASE_TOOL, LIST_DOCUMENTS_TOOL, DUCKDUCKGO_SEARCH_TOOL]

            # Prevent aggressive triggering of chat history for short ambiguous inputs
            if len(latest_user_message.strip()) < 5:
                active_tools = [t for t in active_tools if t.get("function", {}).get("name") != "search_chat_history"]

        system_message = await ChatService._build_system_prompt(
            supabase, user_id, latest_user_message, document_name=scoped_document_name
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
                tool_calls_received = None
                turn_text = ""

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
                                tool_calls_received = chunk.get("tool_calls")
                            elif isinstance(chunk, str):
                                turn_text += chunk
                                full_text += chunk
                                yield f"data: {json.dumps({'content': chunk})}\n\n"
                        break
                    except Exception as turn_err:
                        if attempt == 0 and not turn_text and _is_transient_tool_formatting_error(str(turn_err).lower()):
                            logger.warning(f"Transient tool-formatting error, retrying turn once: {turn_err}")
                            tool_calls_received = None
                            continue
                        raise

                if tool_calls_received:
                    has_called_tool = True
                    history.append(Message(role="assistant", content=turn_text if turn_text.strip() else None, tool_calls=tool_calls_received))

                    for tc in tool_calls_received:
                        tool_call_id = tc["id"]
                        tool_name = tc["function"]["name"]
                        try:
                            tool_args = json.loads(tc["function"]["arguments"])
                        except json.JSONDecodeError:
                            tool_args = {}
                            
                        # Execute the tool using the master dispatcher
                        result_str = await execute_tool(tool_name, tool_args, supabase, session_id, user_id)
                        
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
