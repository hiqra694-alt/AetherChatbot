"""
Dynamic Intent Router -- picks a small, relevant subset of a large MCP
connector's tool schemas for one specific chat turn, instead of always
offering every tool the connector advertises.

Exists because Google's managed Gmail MCP server (mcp_integration/gmail_mcp.py)
exposes 21 full tool schemas (~14k tokens): sending all of them on every turn
blew past Groq's 12,000 TPM limit and got the whole request rejected with an
HTTP 413. Rather than hardcoding a permanent static whitelist (which would
silently make 16 of the 21 tools unreachable forever, regardless of what the
user actually asks for), this scores each tool against the current user
message and keeps only the top `max_tools` -- so a "search my email" turn
gets the search tools, a "draft a reply" turn gets the drafting tools, and
the full 21-tool catalog stays reachable across turns even though no single
turn ever offers more than `max_tools` of them.

Deliberately generic over its `all_tools` input: each entry can be either an
`mcp.types.Tool` (has `.name`/`.description` attributes, e.g. a raw MCP
session's `.tools` list) or this app's own native tool-schema dict shape
(`{"type": "function", "function": {"name": ..., "description": ...}}`, e.g.
api.chat.tools.ALL_TOOLS) -- see _tool_name_and_description. Only ever wired
into the Gmail MCP merge point today (api/chat/services.py), but scoped
generically enough to reuse against another oversized connector later
without changes here.
"""

import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Action-word categories used to boost tools whose *name* signals they match
# the kind of thing the user is asking for -- a plain token-overlap score
# alone under-ranks a tool like "create_draft" against a query like "email
# my team" (no literal word overlap at all, but clearly a drafting intent).
DRAFT_ACTION_KEYWORDS = frozenset({
    "draft", "drafts", "send", "email", "mail", "write", "compose", "reply", "forward",
})
SEARCH_ACTION_KEYWORDS = frozenset({
    "search", "find", "read", "show", "get", "list", "look", "lookup", "check", "view", "see",
})

DRAFT_TOOL_NAME_HINTS = ("draft", "send", "compose", "reply", "forward")
SEARCH_TOOL_NAME_HINTS = ("search", "thread", "message", "list", "get", "read")

# Direct token-overlap and intent-keyword boosts, tuned so a single
# name-hint intent match outweighs a couple of incidental description-word
# overlaps, but never drowns out a tool with strong direct name overlap.
_NAME_TOKEN_WEIGHT = 3.0
_DESCRIPTION_TOKEN_WEIGHT = 1.0
_INTENT_HINT_BOOST = 5.0

# Used only when the query scores every tool at 0 -- i.e. it's ambiguous, or
# shares no vocabulary with any tool at all (e.g. "hello", or an empty
# message). Drafting + search cover the two most common Gmail intents, so
# this keeps the model minimally useful rather than tool-less on a vague turn.
FALLBACK_TOOL_NAMES = ("create_draft", "search_threads")

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set:
    return set(_WORD_RE.findall(text.lower())) if text else set()


def _tool_name_and_description(tool: Any) -> tuple[str, str]:
    """Extracts (name, description) from either an mcp.types.Tool (attribute
    access) or this app's native tool-schema dict (nested under "function") --
    see the module docstring for why both shapes are supported."""
    name = getattr(tool, "name", None)
    description = getattr(tool, "description", None)
    if name is None and isinstance(tool, dict):
        function = tool.get("function", tool)
        name = function.get("name", "")
        description = function.get("description", "")
    return name or "", description or ""


def _score_tool(name: str, description: str, query_tokens: set) -> float:
    name_lower = name.lower()
    name_tokens = _tokenize(name_lower.replace("_", " ").replace("-", " "))
    description_tokens = _tokenize(description)

    score = 0.0
    score += _NAME_TOKEN_WEIGHT * len(query_tokens & name_tokens)
    score += _DESCRIPTION_TOKEN_WEIGHT * len(query_tokens & description_tokens)

    if query_tokens & DRAFT_ACTION_KEYWORDS and any(hint in name_lower for hint in DRAFT_TOOL_NAME_HINTS):
        score += _INTENT_HINT_BOOST
    if query_tokens & SEARCH_ACTION_KEYWORDS and any(hint in name_lower for hint in SEARCH_TOOL_NAME_HINTS):
        score += _INTENT_HINT_BOOST

    return score


def select_relevant_tools(user_query: Optional[str], all_tools: list, max_tools: int = 5) -> list:
    """
    Returns the `max_tools` entries of `all_tools` most relevant to
    `user_query`, preserving each entry's original object/dict as-is (only
    its name/description are inspected, never mutated). Never raises --
    every edge case (empty `all_tools`, empty/None `user_query`, a query
    that matches nothing) degrades to a sensible, logged result rather than
    an exception, since a routing bug here must never take down a chat turn.

    - `all_tools` with `max_tools` or fewer entries is returned unchanged
      (nothing to trim, so no reason to risk dropping a tool the user
      needed) -- this is also what makes the function a safe no-op for any
      connector that was never the problem in the first place.
    - Otherwise, every tool is scored against `user_query` (direct name/
      description token overlap, plus an intent-keyword boost for
      drafting/search-shaped queries -- see _score_tool), and the top
      `max_tools` by score are returned, ties broken by original order
      (Python's sort is stable).
    - If every tool scores 0 (an empty/ambiguous query, or one that shares
      no vocabulary with any tool at all), falls back to whichever of
      FALLBACK_TOOL_NAMES are present in `all_tools`, and only as a last
      resort -- no fallback name present either -- the first `max_tools`
      tools in their original order, so this never returns an empty list
      for a non-empty `all_tools` input.
    """
    all_tools = list(all_tools)
    query = user_query or ""

    if not all_tools:
        logger.info('[IntentRouter] User Query: "%s" -> Selected 0 tools from 0 available tools: []', query)
        return []

    if len(all_tools) <= max_tools:
        _log_selection(query, all_tools, all_tools)
        return all_tools

    query_tokens = _tokenize(query)

    scored = []  # list of (score, name, tool), in original order
    for tool in all_tools:
        name, description = _tool_name_and_description(tool)
        scored.append((_score_tool(name, description, query_tokens), name, tool))

    ranked = sorted(scored, key=lambda item: item[0], reverse=True)
    top_score = ranked[0][0]

    if top_score > 0:
        selected = [tool for _, _, tool in ranked[:max_tools]]
    else:
        by_name = {name: tool for _, name, tool in ranked if name}
        fallback = [by_name[name] for name in FALLBACK_TOOL_NAMES if name in by_name]
        selected = fallback[:max_tools] if fallback else [tool for _, _, tool in ranked[:max_tools]]

    _log_selection(query, selected, all_tools)
    return selected


def _log_selection(query: str, selected: list, all_tools: list) -> None:
    names = [_tool_name_and_description(tool)[0] for tool in selected]
    logger.info(
        '[IntentRouter] User Query: "%s" -> Selected %d tools from %d available tools: [%s]',
        query, len(selected), len(all_tools), ", ".join(names),
    )
