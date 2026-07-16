import json
import logging
from typing import AsyncGenerator, Callable, Awaitable
from fastapi import HTTPException
from supabase import Client
from api.chat.schemas import Message, ProviderEnum
from providers.factory import ProviderFactory
import httpx
from bs4 import BeautifulSoup
import urllib.parse

logger = logging.getLogger(__name__)

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

        if use_web_search and history:
            last_message = history[-1].content
            try:
                logger.info(f"Performing Web Search for query: {last_message}")
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                }
                async with httpx.AsyncClient() as client:
                    resp = await client.post("https://html.duckduckgo.com/html/", data={"q": last_message}, headers=headers, timeout=10.0)
                    
                soup = BeautifulSoup(resp.text, 'html.parser')
                results = []
                
                for div in soup.find_all('div', class_='result'):
                    title_tag = div.find('h2', class_='result__title')
                    snippet_tag = div.find('a', class_='result__snippet')
                    url_tag = div.find('a', class_='result__url')
                    
                    if title_tag and snippet_tag and url_tag:
                        a_tag = title_tag.find('a')
                        if a_tag:
                            title = a_tag.get_text(strip=True)
                            snippet = snippet_tag.get_text(strip=True)
                            raw_url = url_tag.get('href', '')
                            
                            url = raw_url
                            if raw_url.startswith('/l/?uddg='):
                                parsed_url = urllib.parse.parse_qs(urllib.parse.urlparse(raw_url).query)
                                if 'uddg' in parsed_url:
                                    url = parsed_url['uddg'][0]
                            
                            results.append({
                                "title": title,
                                "url": url,
                                "snippet": snippet
                            })
                            
                    if len(results) >= 5:
                        break
                
                search_results_json = results
                
                if results:
                    context_str = "\n".join([f"[{i+1}] {res['title']}\nURL: {res['url']}\nSnippet: {res['snippet']}\n" for i, res in enumerate(results)])
                    injection = f"\n\n[SYSTEM NOTE: The following are real-time web search results for the user's query. Use them to answer accurately. Cite your sources using inline citations like [1] or [2] next to the relevant facts. At the end of your response, output a 'Sources' section listing the references. If the results are irrelevant, ignore them.]\n{context_str}"
                    history[-1].content += injection
            except Exception as search_err:
                logger.error(f"Web Search failed: {search_err}")
                # We do not raise an error, just gracefully degrade to standard generation.

        full_text = ""
        try:
            if search_results_json:
                yield f"data: {json.dumps({'sources': search_results_json})}\n\n"

            async for chunk in provider_instance.stream_response(history):
                if await request_is_disconnected():
                    logger.info("Client disconnected during stream. Terminating.")
                    break
                    
                full_text += chunk
                yield f"data: {json.dumps({'content': chunk})}\n\n"
            
            if not await request_is_disconnected():
                yield "data: [DONE]\n\n"
                
        except Exception as stream_err:
            logger.error(f"Error during stream generation: {stream_err}")
            yield f"data: {json.dumps({'error': str(stream_err)})}\n\n"
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
