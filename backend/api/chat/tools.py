import json
import logging
import httpx
import urllib.parse
from typing import Optional
from bs4 import BeautifulSoup
from datetime import datetime
import zoneinfo
from supabase import Client

from api.documents.services import format_retrieved_chunks, get_relevant_context, list_user_documents

logger = logging.getLogger(__name__)

DUCKDUCKGO_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "duckduckgo_search",
        "description": (
            "Searches the live web. Defines an epistemic boundary: only call this when your own "
            "knowledge is genuinely insufficient or unreliable for the question, not as a default "
            "first step. "
            "WHEN TO USE: (1) real-time or time-sensitive external data — news, current events, "
            "weather, prices, scores, or anything that can change after your training cutoff; "
            "(2) an acronym, term, or name that is genuinely ambiguous or niche, where you are not "
            "confident which of several plausible meanings applies (e.g. it could refer to a "
            "company/product/brand as easily as a technical concept) and answering without checking "
            "would risk guessing. "
            "WHEN NOT TO USE: do NOT call this for foundational computer science or technical "
            "concepts, well-established definitions, or general knowledge you already know with "
            "confidence — answer those directly from your own knowledge instead."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "search_query": {
                    "type": "string",
                    "description": "The optimized search query to look up on DuckDuckGo."
                }
            },
            "required": ["search_query"]
        }
    }
}

CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluates a basic arithmetic expression. Enforce strict usage for explicit arithmetic requests only.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A mathematical expression using numbers and basic operators (+, -, *, /, (), .)."
                }
            },
            "required": ["expression"]
        }
    }
}

GET_TIME_TOOL = {
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "Fetches the current date and time for a given IANA timezone.",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "The city name or IANA timezone string to get the current time for (e.g. 'Tokyo', 'London', 'America/New_York'). Default is 'UTC'."
                }
            },
            "required": ["timezone"]
        }
    }
}

GET_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Retrieves real-time weather and temperature for a given city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The name of the city to get weather for."
                },
                "unit": {
                    "type": "string",
                    "description": "The unit for temperature ('celsius' or 'fahrenheit'). Default is 'celsius'."
                }
            },
            "required": ["city"]
        }
    }
}

SEARCH_CHAT_HISTORY_TOOL = {
    "type": "function",
    "function": {
        "name": "search_chat_history",
        "description": "Searches past chat history across user sessions. Use ONLY when the user explicitly asks about past topics, prior conversations, or earlier messages. Do NOT invoke for general capability questions, greetings, or short ambiguous inputs.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Optional query string to search for specific topics or keywords in past chat sessions. Leave empty or pass empty string to retrieve recent messages across past sessions."
                },
                "limit": {
                    "type": "integer",
                    "description": "Max number of messages to return. Default is 5."
                }
            },
            "required": []
        }
    }
}

SEARCH_KNOWLEDGE_BASE_TOOL = {
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": "Searches the user's private uploaded Knowledge Base (e.g. their CV, academic papers, or other uploaded documents). Use ONLY when the question explicitly refers to the user's own uploaded content (e.g. 'according to my CV', 'what does the document say', 'summarize my report'). Do NOT invoke for general knowledge, definitions, coding/technical concepts, or conversational questions you can already answer yourself (e.g. 'what is a REST API') — answer those directly instead.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up in the user's uploaded documents."
                }
            },
            "required": ["query"]
        }
    }
}

LIST_DOCUMENTS_TOOL = {
    "type": "function",
    "function": {
        "name": "list_documents",
        "description": "Lists the exact names of every document the user has uploaded to their Knowledge Base. Use this whenever the user asks what documents/files you have access to, or which documents they've uploaded. Do NOT use search_knowledge_base for this — it only returns a partial, similarity-ranked set of chunks and cannot reliably enumerate every uploaded document.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
}

BASE_TOOLS = [CALCULATOR_TOOL, GET_TIME_TOOL, GET_WEATHER_TOOL, SEARCH_CHAT_HISTORY_TOOL]

async def duckduckgo_search(search_query: str) -> str:
    try:
        logger.info(f"Performing Web Search for query: {search_query}")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post("https://html.duckduckgo.com/html/", data={"q": search_query}, headers=headers, timeout=10.0)
            
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
                
        if not results:
            return json.dumps({"error": "No relevant search results found."})
            
        return json.dumps(results)
    except Exception as search_err:
        logger.error(f"Web Search failed: {search_err}")
        return json.dumps({"error": "Search failed or blocked"})

async def calculator(expression: str) -> str:
    try:
        allowed_chars = set("0123456789+-*/(). ")
        if not all(char in allowed_chars for char in expression):
            return json.dumps({"error": "Invalid characters in mathematical expression. Only basic arithmetic allowed."})
            
        # evaluate the expression safely
        result = eval(expression, {"__builtins__": None}, {})
        return json.dumps({"result": str(result)})
    except Exception as e:
        logger.error(f"Calculator failed for expression '{expression}': {e}")
        return json.dumps({"error": "Failed to evaluate expression."})

async def get_current_time(timezone: str = "UTC") -> str:
    try:
        # Map common city names to IANA timezones
        city_map = {
            "tokyo": "Asia/Tokyo",
            "london": "Europe/London",
            "new york": "America/New_York",
            "paris": "Europe/Paris",
            "sydney": "Australia/Sydney",
            "berlin": "Europe/Berlin",
            "dubai": "Asia/Dubai",
            "singapore": "Asia/Singapore",
            "los angeles": "America/Los_Angeles",
            "san francisco": "America/Los_Angeles",
            "chicago": "America/Chicago",
            "toronto": "America/Toronto",
            "seattle": "America/Los_Angeles",
            "mumbai": "Asia/Kolkata",
            "delhi": "Asia/Kolkata",
            "beijing": "Asia/Shanghai",
            "shanghai": "Asia/Shanghai",
            "hong kong": "Asia/Hong_Kong",
        }
        
        normalized_tz = timezone.lower().strip()
        resolved_tz = city_map.get(normalized_tz, timezone)
        
        tz = zoneinfo.ZoneInfo(resolved_tz)
        now = datetime.now(tz)
        formatted_time = now.strftime("%Y-%m-%d %H:%M:%S %Z, %A")
        return json.dumps({"time": formatted_time})
    except Exception as e:
        logger.error(f"Get time failed for timezone '{timezone}': {e}")
        return json.dumps({"status": "error", "message": "Location not found"})

async def get_weather(city: str, unit: str = "celsius") -> str:
    try:
        async with httpx.AsyncClient() as client:
            # Geocode the city
            geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={urllib.parse.quote(city)}&count=1&language=en&format=json"
            geo_resp = await client.get(geo_url, timeout=10.0)
            geo_data = geo_resp.json()
            
            if "results" not in geo_data or not geo_data["results"]:
                return json.dumps({"error": f"Could not find coordinates for city: {city}"})
                
            lat = geo_data["results"][0]["latitude"]
            lon = geo_data["results"][0]["longitude"]
            city_name = geo_data["results"][0]["name"]
            
            # Fetch weather
            temp_unit = "celsius" if unit.lower() == "celsius" else "fahrenheit"
            weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&temperature_unit={temp_unit}"
            weather_resp = await client.get(weather_url, timeout=10.0)
            weather_data = weather_resp.json()
            
            if "current_weather" not in weather_data:
                return json.dumps({"error": "Failed to retrieve current weather data."})
                
            current_weather = weather_data["current_weather"]
            return json.dumps({
                "city": city_name,
                "temperature": current_weather["temperature"],
                "unit": temp_unit,
                "windspeed": current_weather["windspeed"],
                "winddirection": current_weather["winddirection"]
            })
    except Exception as e:
        logger.error(f"Get weather failed for city '{city}': {e}")
        return json.dumps({"error": "Failed to fetch weather."})

async def search_chat_history(supabase: Client, session_id: str = "", query: str = "", limit: int = 5) -> str:
    try:
        query_str = (query or "").strip()
        meta_phrases = {
            "previous sessions", "past sessions", "history", "all", "everything",
            "previous chat", "past conversations", "what did we talk about", "what have i talked about"
        }
        is_meta_query = not query_str or query_str.lower() in meta_phrases

        if not is_meta_query:
            res = (
                supabase.table("messages")
                .select("role, content, created_at")
                .ilike("content", f"%{query_str}%")
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            if res.data:
                results = [{"role": row["role"], "content": row["content"]} for row in reversed(res.data)]
                return json.dumps({"results": results})

        # Fallback: retrieve the most recent messages across all sessions of the user
        res = (
            supabase.table("messages")
            .select("role, content, created_at")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        if not res.data:
            return json.dumps({"result": "No past chat history found for this user."})

        results = [{"role": row["role"], "content": row["content"]} for row in reversed(res.data)]
        return json.dumps({"results": results})
    except Exception as e:
        logger.error(f"Search chat history failed: {e}")
        return json.dumps({"result": "Chat history is currently unavailable."})

async def search_knowledge_base(supabase: Client, user_id: Optional[str], query: str) -> str:
    """
    Agentic RAG entry point: only invoked when the model itself decides (via
    a tool call) that the user's uploaded documents are relevant, rather than
    running retrieval unconditionally on every turn. Searches across all of
    the user's documents (document_name=None) since this path is for
    conversational turns with no file attached in the current request.
    """
    if not user_id:
        return json.dumps({"error": "No authenticated user to scope the knowledge base search to."})

    try:
        chunks = await get_relevant_context(supabase, query, user_id, document_name=None)
    except Exception as search_err:
        logger.error(f"search_knowledge_base failed for query '{query}': {search_err}")
        return json.dumps({"error": "Failed to search the knowledge base."})

    if not chunks:
        return json.dumps({"result": "No relevant information found in the knowledge base."})

    return format_retrieved_chunks(chunks)

async def list_documents(supabase: Client, user_id: Optional[str]) -> str:
    """
    Deterministic enumeration of the user's uploaded documents, backed by the
    same list_user_documents() query as GET /api/documents (RLS + explicit
    user_id-scoped). Exists so "what documents do you have access to?" is
    answered from real rows instead of the model inferring/completing a list
    from a lossy top-k semantic search, which is what previously produced
    hallucinated filenames.
    """
    if not user_id:
        return json.dumps({"error": "No authenticated user to scope the document list to."})

    try:
        documents = await list_user_documents(supabase, user_id)
    except Exception as list_err:
        logger.error(f"list_documents failed for user {user_id}: {list_err}")
        return json.dumps({"error": "Failed to list documents."})

    if not documents:
        return json.dumps({"result": "No documents have been uploaded yet."})

    return json.dumps({
        "documents": [doc.document_name for doc in documents]
    })

async def execute_tool(tool_name: str, tool_args: dict, supabase: Client, session_id: str, user_id: Optional[str] = None) -> str:
    if tool_name == "duckduckgo_search":
        return await duckduckgo_search(tool_args.get("search_query", ""))
    elif tool_name == "calculator":
        return await calculator(tool_args.get("expression", ""))
    elif tool_name == "get_current_time":
        return await get_current_time(tool_args.get("timezone", "UTC"))
    elif tool_name == "get_weather":
        return await get_weather(tool_args.get("city", ""), tool_args.get("unit", "celsius"))
    elif tool_name == "search_chat_history":
        return await search_chat_history(supabase, session_id, tool_args.get("query", ""), tool_args.get("limit", 5))
    elif tool_name == "search_knowledge_base":
        return await search_knowledge_base(supabase, user_id, tool_args.get("query", ""))
    elif tool_name == "list_documents":
        return await list_documents(supabase, user_id)
    else:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
