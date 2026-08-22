"""
Voice agent system prompt (Phase 2). Kept separate from voice/agent.py so
the prompt text can be edited without touching worker wiring.
"""

from typing import Iterable

VOICE_SYSTEM_PROMPT = (
    "You are the voice assistant for AetherChat. Your responses are spoken "
    "aloud to the user. Keep all answers concise and direct, limited to 3 "
    "to 4 sentences, unless the user explicitly requests more detail. You "
    "have access to tools for retrieval and MCP execution; use them when "
    "needed before answering.\n\n"
    "You are a real-time conversational voice assistant. Speak naturally. "
    "NEVER use Markdown formatting, bullet points, asterisks, backticks, or "
    "pronounce snake_case function names (e.g., say 'the weather tool' "
    "instead of 'get_current_weather'). "
    "ROUTING RULE: If the user asks for current events, news, weather, or "
    "facts outside your immediate training data, YOU MUST default to using "
    "the web search tool. "
    "Never disclose internal API keys, tokens, or backend logic."
)


def build_mcp_tools_prompt_block(mcp_tools: Iterable) -> str:
    """
    Formats the MCP tools currently discovered by mcp_manager (e.g. a
    generic MCP_SERVER_URLS entry such as Brave Search) into a text block appended to VOICE_SYSTEM_PROMPT at session start (see
    voice/agent.py's entrypoint), so the voice LLM actually knows which
    tool_name/arguments are valid to pass to VoiceTools.execute_mcp_tool.

    Without this, the model has no way to know an MCP tool exists at all --
    LiveKit's Toolset gives every @function_tool its own static schema
    rather than a per-session dynamically-merged one the way text chat's
    get_merged_tool_schemas works (see tools_adapter.py's module docstring),
    so execute_mcp_tool is otherwise a blind escape hatch the model can only
    reach by guessing a plausible-sounding tool_name.

    `mcp_tools` is a list of `mcp.types.Tool` (as returned by
    mcp_manager.list_cached_tools()) -- each has `.name` and `.description`.
    Empty/no tools connected is handled explicitly so the model isn't left
    to guess whether silence means "none configured" or "just not listed".
    """
    mcp_tools = list(mcp_tools)
    if not mcp_tools:
        return (
            "\n\nNo external MCP tools (e.g. GitHub) are currently connected. "
            "Do not call execute_mcp_tool -- it will always fail right now."
        )

    lines = [
        "\n\nThe following external MCP tools are currently connected. Call "
        "them via execute_mcp_tool(tool_name, arguments_json), passing the "
        "exact tool_name below and a JSON object string matching its "
        "arguments -- never call execute_mcp_tool for anything not listed "
        "here:"
    ]
    for tool in mcp_tools:
        description = (tool.description or "").strip()
        lines.append(f"- {tool.name}: {description}" if description else f"- {tool.name}")
    return "\n".join(lines)
