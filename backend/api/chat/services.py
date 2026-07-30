import json
import logging
from typing import AsyncGenerator, Callable, Awaitable, Optional
from fastapi import HTTPException
from supabase import Client
from api.chat.schemas import Message, ProviderEnum
from providers.factory import ProviderFactory
from providers.base import BaseProvider
from api.chat.tools import BASE_TOOLS, DUCKDUCKGO_SEARCH_TOOL, SEARCH_KNOWLEDGE_BASE_TOOL, execute_tool
from api.documents.services import get_relevant_context

logger = logging.getLogger(__name__)

# Single source of truth for the assistant's persona/instructions. Kept short
# and token-efficient — long, multi-rule prompts were confusing Groq/Llama's
# function-calling parser. Both the plain-chat path and the RAG-grounded path
# (ChatService._build_system_prompt) build on this one string rather than
# each hardcoding their own copy.
SYSTEM_PROMPT = Message(
    role="system",
    content=(
        "You are AetherChat, a helpful AI assistant.\n"
        "- Answer general knowledge, technical, and conversational questions directly from your own "
        "knowledge. Only use the search_knowledge_base tool when the question explicitly requires "
        "information from the user's own private uploaded documents.\n"
        "- When answering from retrieved documents, provide clear, comprehensive answers and end "
        "your response with a source footer listing the document name (e.g., '\\n\\n--- \\n*Source: filename.pdf*').\n"
        "- If the user's question specifically asked about their uploaded documents and the retrieved "
        "context lacks the answer, state: 'I don't know based on the provided documents.' Do not use "
        "this fallback for general knowledge questions — answer those from your own knowledge instead."
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

        context_blocks = "\n\n".join(
            f"--- START OF CHUNK FROM: {chunk.document_name} ---\n"
            f"{chunk.chunk_text}\n"
            "--- END OF CHUNK ---"
            for chunk in chunks
        )
        grounded_content = (
            f"{SYSTEM_PROMPT.content}\n\n"
            "--- RETRIEVED KNOWLEDGE BASE CONTEXT ---\n"
            f"{context_blocks}\n"
            "--- END OF CONTEXT ---"
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
        use_web_search: bool = False,
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
            # BRANCH B: no file attached. Offer search_knowledge_base so the
            # model can pull from the user's documents agentically, only
            # paying the Voyage/Supabase round trip if it decides to.
            active_tools = list(BASE_TOOLS) + [SEARCH_KNOWLEDGE_BASE_TOOL]
            if use_web_search:
                active_tools.append(DUCKDUCKGO_SEARCH_TOOL)

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
        try:
            for _ in range(5):
                stream = provider_instance.stream_response(history, tools=active_tools)
                tool_calls_received = None
                turn_text = ""
                
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
                
                if tool_calls_received:
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
                user_friendly_err = "I am unable to process that tool request. Web Search is currently turned off. Please toggle Web Search ON at the bottom of the chat if you'd like live up-to-date information."
            elif "failed to call a function" in err_msg or "adjust your prompt" in err_msg:
                user_friendly_err = "I had trouble formatting a tool call for that request. Please try rephrasing your message, or turn off Web Search if it's enabled."
            elif "rate_limit_exceeded" in err_msg or "429" in err_msg:
                user_friendly_err = "The AI provider is currently experiencing high traffic. Please try again in a moment."
            elif "failed_generation" in err_msg:
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
