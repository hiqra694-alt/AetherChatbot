import json
import logging
import re
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
            "WHEN TO USE: real-time external events, breaking news, live weather, prices, scores, "
            "or anything that can change after your training cutoff; or a highly niche/conflicting "
            "domain acronym or term where you are not confident which of several plausible meanings "
            "applies (e.g. it could refer to a company/product/brand as easily as a technical "
            "concept) and answering without checking would risk guessing. "
            "WHEN NOT TO USE: foundational computer science concepts, standard/well-established "
            "definitions, or general world facts already present in your own knowledge — answer "
            "those directly instead."
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
        "description": (
            "Evaluates a mathematical expression. "
            "WHEN TO USE: complex multi-step arithmetic, equation solving, or financial/statistical "
            "formulas that require precise computation. "
            "WHEN NOT TO USE: factual trivia, sports rules or player counts, general knowledge, "
            "simple single-step counting, or dates — answer those directly instead, never route "
            "them through arithmetic evaluation just because the answer happens to be a number."
        ),
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
        "description": (
            "Fetches the current date and time for a given IANA timezone. "
            "WHEN TO USE: the user explicitly asks what the current date/time is in some location. "
            "WHEN NOT TO USE: historical dates, dates mentioned in documents/chat history, or "
            "general date arithmetic that doesn't depend on right now."
        ),
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
        "description": (
            "Retrieves real-time weather and temperature for a given city. "
            "WHEN TO USE: the user explicitly asks for current weather/temperature/conditions in a "
            "city. WHEN NOT TO USE: general climate facts, seasons, or historical weather trivia."
        ),
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
        "description": (
            "Searches past chat history across user sessions. "
            "WHEN TO USE: explicit questions asking what was said, discussed, or worked on in past "
            "conversation sessions (e.g. 'what did we talk about', 'what were we working on last "
            "time'). "
            "WHEN NOT TO USE: general capability questions, greetings, or short ambiguous inputs."
        ),
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
        "description": (
            "Searches the user's private uploaded Knowledge Base (e.g. their CV, academic papers, "
            "or other uploaded documents). "
            "WHEN TO USE: specific questions about uploaded personal/corporate documents, CVs, or "
            "research papers (e.g. 'according to my CV', 'what does the document say', 'summarize "
            "my report'). "
            "WHEN NOT TO USE: general knowledge, definitions, coding/technical concepts, or "
            "conversational questions you can already answer yourself (e.g. 'what is a REST API') "
            "— answer those directly instead."
        ),
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
        "description": (
            "Lists the exact names of every document the user has uploaded to their Knowledge Base. "
            "WHEN TO USE: specific questions about uploaded personal/corporate documents, CVs, or "
            "research papers — specifically, enumerating which ones exist (e.g. 'what documents do "
            "you have access to', 'what have I uploaded'). "
            "WHEN NOT TO USE: questions about what's inside a document — use search_knowledge_base "
            "for that instead. This tool returns filenames only, never content, and "
            "search_knowledge_base only returns a partial, similarity-ranked set of chunks and "
            "cannot reliably enumerate every uploaded document, so neither substitutes for the other."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
}

BASE_TOOLS = [CALCULATOR_TOOL, GET_TIME_TOOL, GET_WEATHER_TOOL, SEARCH_CHAT_HISTORY_TOOL]

# Master registry of every tool schema this app defines, regardless of
# whether a given request/turn actually offers all of them to the model
# (see ChatService.ACTIVE_TOOLS and the per-turn active_tools/turn_tools
# selection in services.stream_chat, which choose a subset of this list).
# This is also the single source of truth the Groq provider's text-tag
# interceptor (providers/groq.py) validates a hallucinated
# `<tool_name>...</tool_name>` tag's name against, and that
# _TOOL_NAME_PATTERN there is derived from — so a new tool added here never
# silently falls out of sync with either safety net.
ALL_TOOLS = BASE_TOOLS + [DUCKDUCKGO_SEARCH_TOOL, SEARCH_KNOWLEDGE_BASE_TOOL, LIST_DOCUMENTS_TOOL]
ALL_TOOL_NAMES = frozenset(t["function"]["name"] for t in ALL_TOOLS)

# Looked up by repair_tool_arguments' regex fallback to know which property
# names/types to go looking for in a raw string that failed every JSON
# repair attempt -- keyed by tool name rather than re-declared, so a tool's
# schema (and this fallback) can never drift out of sync with each other.
_TOOL_PARAM_SCHEMAS = {t["function"]["name"]: t["function"]["parameters"] for t in ALL_TOOLS}


def _balancing_closers(text: str) -> str:
    """The `}`/`]` characters needed to close every currently-unbalanced
    `{`/`[` in `text`, in the right order (braces before brackets isn't
    strictly correct for arbitrary nesting, but every tool's arguments
    object here is always brace-first at the top level, so this is
    sufficient for what it's used to repair)."""
    closers = '}' * max(text.count('{') - text.count('}'), 0)
    closers += ']' * max(text.count('[') - text.count(']'), 0)
    return closers


def _json_repair_candidates(text: str):
    """
    Yields increasingly-lenient rewrites of `text` to attempt json.loads
    against, in order:

    1. The raw text itself.
    2. With an unterminated string value closed (an odd number of
       unescaped `"` means the text ends mid-string-value -- e.g.
       '{"city": "San Fran') and a trailing dangling comma dropped, then
       unclosed braces/brackets balanced. Recovers a value truncated
       mid-way through by a token limit or flaky small-model generation.
    3. With the entire trailing dangling key (name present but colon/value
       cut off entirely, or cut off partway through the value) trimmed off
       instead, then balanced. Recovers the case where #2's repair would
       produce a dangling `"key"` with no `: value` at all, which still
       isn't valid JSON.

    Each candidate is tried against the ORIGINAL text, not chained off the
    previous one, so an earlier candidate's repair attempt that turns out
    to still be invalid JSON never corrupts input to a later one.
    """
    yield text

    quote_count = len(re.findall(r'(?<!\\)"', text))
    closed_string = text + '"' if quote_count % 2 == 1 else text
    closed_string = re.sub(r',\s*$', '', closed_string)
    if closed_string != text:
        yield closed_string + _balancing_closers(closed_string)

    trimmed = re.sub(r',?\s*"[^"]*"?\s*:?\s*"?[^"}\]]*$', '', text)
    trimmed = re.sub(r',\s*$', '', trimmed).rstrip()
    if trimmed and trimmed != text:
        yield trimmed + _balancing_closers(trimmed)


def _try_parse_json_object(text: str) -> Optional[dict]:
    """Returns the first candidate from _json_repair_candidates that parses
    as a JSON object, or None if none of them do."""
    for candidate in _json_repair_candidates(text):
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def repair_tool_arguments(tool_name: str, raw_arguments: str) -> dict:
    """
    Parses a tool call's raw `arguments` string into a dict, tolerating the
    dirty/truncated JSON llama-3.3-70b-versatile occasionally emits (a
    trailing comma, or an object cut off mid-value by a token limit)
    instead of discarding every argument the instant a bare `json.loads`
    first fails -- that previously turned a call like
    get_weather('{"city": "Lahon') into an empty {} and a guaranteed
    "Could not find coordinates" response, even though the city name was
    sitting right there in the raw text.

    If no repair attempt produces valid JSON at all, falls back to
    regex-extracting each of the tool's own declared parameters (by name
    and declared type, from this app's own schema in ALL_TOOLS) directly
    out of the raw string, so a required field like `city` or
    `search_query` still reaches the tool. Never raises -- an
    unrecoverable string returns whatever partial dict could be salvaged,
    possibly empty.
    """
    raw_arguments = (raw_arguments or "").strip()
    if not raw_arguments:
        return {}

    parsed = _try_parse_json_object(raw_arguments)
    if parsed is not None:
        return parsed

    schema = _TOOL_PARAM_SCHEMAS.get(tool_name) or {}
    properties = schema.get("properties", {}) if isinstance(schema, dict) else {}

    recovered = {}
    for prop_name, prop_schema in properties.items():
        prop_type = (prop_schema or {}).get("type")
        if prop_type in ("integer", "number"):
            m = re.search(rf'"{re.escape(prop_name)}"\s*:\s*(-?\d+(?:\.\d+)?)', raw_arguments)
            if m:
                recovered[prop_name] = int(float(m.group(1))) if prop_type == "integer" else float(m.group(1))
        else:
            m = re.search(rf'"{re.escape(prop_name)}"\s*:\s*"([^"]*)', raw_arguments)
            if m:
                recovered[prop_name] = m.group(1)
    return recovered

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

async def search_knowledge_base(supabase: Client, user_id: Optional[str], session_id: str, query: str) -> str:
    """
    Agentic RAG entry point: only invoked when the model itself decides (via
    a tool call) that the user's uploaded documents are relevant, rather than
    running retrieval unconditionally on every turn. Searches across all
    documents uploaded in the current chat `session_id` (document_name=None,
    still scoped to this session -- Context Isolation) since this path is for
    conversational turns with no file attached in the current request.
    """
    if not user_id:
        return json.dumps({"error": "No authenticated user to scope the knowledge base search to."})

    try:
        chunks = await get_relevant_context(supabase, query, user_id, session_id, document_name=None)
    except Exception as search_err:
        logger.error(f"search_knowledge_base failed for query '{query}': {search_err}")
        return json.dumps({"error": "Failed to search the knowledge base."})

    if not chunks:
        return json.dumps({"result": "No relevant information found in the knowledge base."})

    return format_retrieved_chunks(chunks)

async def list_documents(supabase: Client, user_id: Optional[str], session_id: str) -> str:
    """
    Deterministic enumeration of the documents uploaded in the current chat
    session, backed by the same list_user_documents() query as GET
    /api/documents (RLS + explicit user_id/session_id-scoped). Exists so
    "what documents do you have access to?" is answered from real rows
    instead of the model inferring/completing a list from a lossy top-k
    semantic search, which is what previously produced hallucinated
    filenames.
    """
    if not user_id:
        return json.dumps({"error": "No authenticated user to scope the document list to."})

    try:
        documents = await list_user_documents(supabase, user_id, session_id)
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
        return await search_knowledge_base(supabase, user_id, session_id, tool_args.get("query", ""))
    elif tool_name == "list_documents":
        return await list_documents(supabase, user_id, session_id)
    else:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
