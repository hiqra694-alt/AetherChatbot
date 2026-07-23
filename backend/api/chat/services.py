import json
import logging
from typing import AsyncGenerator, Callable, Awaitable
from fastapi import HTTPException
from supabase import Client
from api.chat.schemas import Message, ProviderEnum
from providers.factory import ProviderFactory
from providers.base import BaseProvider
from api.chat.tools import BASE_TOOLS, DUCKDUCKGO_SEARCH_TOOL, execute_tool

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = Message(
    role="system",
    content=(
        "You are AetherChat, a smart, helpful AI assistant.\n\n"
        "CRITICAL INSTRUCTIONS & TOOL RULES:\n"
        "1. TOOL RESTRICTION: You may ONLY call tools explicitly listed in your active tools menu. NEVER invoke or hallucinate unlisted tool names.\n\n"
        "2. WEB SEARCH DISABLED BEHAVIOR: If the user asks for real-time news, current events, or external up-to-date information, and the web search tool is NOT in your active tools menu, answer conversationally letting the user know Web Search is currently disabled and can be toggled on at the bottom of the chat interface.\n\n"
        "3. TOOL RESPONSE HANDLING: Integrate tool outputs into clean, natural conversational responses. Do NOT output raw JSON strings, code blocks, function names, or internal architecture details.\n\n"
        "4. NO UNPROMPTED TOOL ERRORS: Do NOT append error messages or system status updates unless a tool explicitly fails during a requested action. If a tool returns a result, summarize it accurately and directly answer the user's question."
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
            raise HTTPException(status_code=500, detail=f"Failed to fetch conversation history: {str(db_err)}")


    @staticmethod
    def get_provider(provider_enum: ProviderEnum) -> BaseProvider:
        try:
            return ProviderFactory.get_provider(provider_enum.value)
        except ValueError as val_err:
            raise HTTPException(status_code=400, detail=str(val_err))
        except Exception as init_err:
            logger.error(f"Failed to initialize provider {provider_enum.value}: {init_err}")
            raise HTTPException(status_code=500, detail=f"AI Provider Initialization Error: {str(init_err)}")

    @staticmethod
    async def stream_chat(
        provider_instance: BaseProvider, 
        history: list[Message], 
        supabase: Client, 
        session_id: str, 
        provider_name: str,
        request_is_disconnected: Callable[[], Awaitable[bool]],
        use_web_search: bool = False
    ) -> AsyncGenerator[str, None]:
        search_results_json = []
        active_tools = list(BASE_TOOLS)
        if use_web_search:
            active_tools.append(DUCKDUCKGO_SEARCH_TOOL)

        if not history or history[0].role != "system":
            history.insert(0, SYSTEM_PROMPT)
        else:
            history[0] = SYSTEM_PROMPT

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
                        result_str = await execute_tool(tool_name, tool_args, supabase, session_id)
                        
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
            err_msg = str(stream_err)
            if "tool call validation failed" in err_msg or "attempted to call tool" in err_msg:
                user_friendly_err = "I am unable to process that tool request. Web Search is currently turned off. Please toggle Web Search ON at the bottom of the chat if you'd like live up-to-date information."
            else:
                user_friendly_err = err_msg
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
