import json
import logging
import httpx
import urllib.parse
from bs4 import BeautifulSoup
from datetime import datetime
import zoneinfo
from supabase import Client

logger = logging.getLogger(__name__)

DUCKDUCKGO_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "duckduckgo_search",
        "description": "Search the web for up-to-date information. Use this when the user asks about current events, real-time data, or facts you do not know.",
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
        "description": "Evaluates a basic arithmetic expression.",
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
                    "description": "The IANA timezone string, e.g. 'America/New_York', 'Asia/Tokyo', 'UTC'. Default is 'UTC'."
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
        "description": "Queries the past messages for this chat session. Use this to recall earlier context if the user refers back to something said earlier in the session.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The text to search for in past messages."
                },
                "limit": {
                    "type": "integer",
                    "description": "Max number of messages to return. Default is 5."
                }
            },
            "required": ["query"]
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
        return json.dumps({"error": f"Failed to evaluate expression: {e}"})

async def get_current_time(timezone: str = "UTC") -> str:
    try:
        tz = zoneinfo.ZoneInfo(timezone)
        now = datetime.now(tz)
        formatted_time = now.strftime("%Y-%m-%d %H:%M:%S %Z, %A")
        return json.dumps({"time": formatted_time})
    except Exception as e:
        logger.error(f"Get time failed for timezone '{timezone}': {e}")
        return json.dumps({"error": f"Invalid timezone format or failure: {e}"})

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
        return json.dumps({"error": f"Failed to fetch weather: {e}"})

async def search_chat_history(supabase: Client, session_id: str, query: str, limit: int = 5) -> str:
    try:
        res = supabase.table("messages").select("role, content").eq("session_id", session_id).ilike("content", f"%{query}%").order("created_at", desc=True).limit(limit).execute()
        if not res.data:
            return json.dumps({"result": "No past messages matched your query."})
            
        # Reverse to chronological order among the results returned
        results = [{"role": row["role"], "content": row["content"]} for row in reversed(res.data)]
        return json.dumps(results)
    except Exception as e:
        logger.error(f"Search chat history failed: {e}")
        return json.dumps({"error": f"Database search failed: {e}"})

async def execute_tool(tool_name: str, tool_args: dict, supabase: Client, session_id: str) -> str:
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
    else:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
