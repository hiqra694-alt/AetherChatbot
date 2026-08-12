import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp import types
from mcp_integration.intent_router import select_relevant_tools, FALLBACK_TOOL_NAMES


def _tool(name: str, description: str) -> types.Tool:
    return types.Tool(name=name, description=description, inputSchema={"type": "object", "properties": {}})


# A stand-in for the Gmail MCP server's 21-tool catalog -- kept intentionally
# larger than the default max_tools=5 so every test here exercises the real
# scoring/trimming path, not the "nothing to trim" short-circuit.
_GMAIL_LIKE_TOOLS = [
    _tool("create_draft", "Creates a new email draft."),
    _tool("list_drafts", "Lists the user's saved email drafts."),
    _tool("send_message", "Sends an email message."),
    _tool("search_threads", "Searches Gmail threads matching a query."),
    _tool("get_thread", "Fetches a single Gmail thread by id."),
    _tool("get_message", "Fetches a single Gmail message by id."),
    _tool("list_labels", "Lists the user's Gmail labels."),
    _tool("create_label", "Creates a new Gmail label."),
    _tool("delete_label", "Deletes a Gmail label."),
    _tool("modify_message_labels", "Adds or removes labels on a message."),
    _tool("trash_message", "Moves a message to trash."),
    _tool("untrash_message", "Restores a message from trash."),
    _tool("get_attachment", "Downloads a message attachment."),
    _tool("list_threads", "Lists Gmail threads."),
    _tool("delete_draft", "Deletes an email draft."),
    _tool("update_draft", "Updates an existing email draft."),
    _tool("send_draft", "Sends an existing draft."),
    _tool("get_profile", "Gets the user's Gmail profile."),
    _tool("watch_mailbox", "Registers a push notification watch on the mailbox."),
    _tool("stop_watch", "Stops a push notification watch."),
    _tool("batch_modify_messages", "Modifies labels on multiple messages at once."),
]


def test_short_circuits_when_at_or_under_max_tools():
    small_set = _GMAIL_LIKE_TOOLS[:5]
    result = select_relevant_tools("search my inbox", small_set, max_tools=5)
    assert result == small_set


def test_empty_all_tools_returns_empty_list():
    assert select_relevant_tools("anything", [], max_tools=5) == []


def test_never_returns_more_than_max_tools():
    result = select_relevant_tools("search my email for invoices", _GMAIL_LIKE_TOOLS, max_tools=4)
    assert len(result) == 4


def test_default_max_tools_is_five():
    result = select_relevant_tools("search my email for invoices", _GMAIL_LIKE_TOOLS)
    assert len(result) == 5


def test_search_intent_prioritizes_search_tools():
    result = select_relevant_tools("search my inbox for the invoice from Acme", _GMAIL_LIKE_TOOLS, max_tools=5)
    names = {t.name for t in result}
    assert "search_threads" in names


def test_draft_intent_prioritizes_drafting_tools():
    result = select_relevant_tools("draft an email to my team about the launch", _GMAIL_LIKE_TOOLS, max_tools=5)
    names = {t.name for t in result}
    assert "create_draft" in names


def test_send_keyword_prioritizes_send_tools_over_unrelated_ones():
    result = select_relevant_tools("send this draft to the client now", _GMAIL_LIKE_TOOLS, max_tools=5)
    names = {t.name for t in result}
    assert names & {"send_draft", "send_message", "create_draft"}
    assert "watch_mailbox" not in names
    assert "get_attachment" not in names


def test_ambiguous_query_falls_back_to_fallback_set():
    result = select_relevant_tools("hello", _GMAIL_LIKE_TOOLS, max_tools=5)
    names = {t.name for t in result}
    assert names == set(FALLBACK_TOOL_NAMES)


def test_empty_query_falls_back_to_fallback_set():
    result = select_relevant_tools("", _GMAIL_LIKE_TOOLS, max_tools=5)
    names = {t.name for t in result}
    assert names == set(FALLBACK_TOOL_NAMES)


def test_none_query_falls_back_without_raising():
    result = select_relevant_tools(None, _GMAIL_LIKE_TOOLS, max_tools=5)
    assert {t.name for t in result} == set(FALLBACK_TOOL_NAMES)


def test_fallback_set_present_in_selection_tools_list():
    """Sanity check that the fallback names this module hardcodes actually
    exist in a realistic Gmail-MCP-shaped tool set, matching the example
    names given in the Phase spec (create_draft, search_threads)."""
    available = {t.name for t in _GMAIL_LIKE_TOOLS}
    assert set(FALLBACK_TOOL_NAMES).issubset(available)


def test_no_fallback_names_present_still_returns_max_tools_without_raising():
    """An ambiguous query against a tool catalog that happens to have
    neither fallback name must still degrade gracefully -- first max_tools
    tools in original order -- rather than returning nothing or raising."""
    tools_without_fallback_names = [
        _tool("alpha_tool", "Does alpha things."),
        _tool("beta_tool", "Does beta things."),
        _tool("gamma_tool", "Does gamma things."),
        _tool("delta_tool", "Does delta things."),
        _tool("epsilon_tool", "Does epsilon things."),
        _tool("zeta_tool", "Does zeta things."),
    ]
    result = select_relevant_tools("hello", tools_without_fallback_names, max_tools=5)
    assert len(result) == 5
    assert result == tools_without_fallback_names[:5]


def test_handles_native_tool_schema_dict_shape():
    """Non-MCP tools (this app's own {"type": "function", "function": {...}}
    schema dicts, e.g. api.chat.tools.ALL_TOOLS) must score/select correctly
    too, not just mcp.types.Tool objects."""
    dict_tools = [
        {"type": "function", "function": {"name": "create_draft", "description": "Creates a draft."}},
        {"type": "function", "function": {"name": "search_threads", "description": "Searches threads."}},
        {"type": "function", "function": {"name": "get_profile", "description": "Gets the profile."}},
        {"type": "function", "function": {"name": "list_labels", "description": "Lists labels."}},
        {"type": "function", "function": {"name": "trash_message", "description": "Trashes a message."}},
        {"type": "function", "function": {"name": "watch_mailbox", "description": "Watches the mailbox."}},
    ]

    result = select_relevant_tools("search for an email", dict_tools, max_tools=3)

    assert len(result) == 3
    names = {t["function"]["name"] for t in result}
    assert "search_threads" in names


def test_selected_tools_are_the_original_objects_not_copies():
    result = select_relevant_tools("search my inbox", _GMAIL_LIKE_TOOLS, max_tools=5)
    for tool in result:
        assert tool in _GMAIL_LIKE_TOOLS


def test_logs_selection_summary(caplog):
    import logging
    with caplog.at_level(logging.INFO, logger="mcp_integration.intent_router"):
        select_relevant_tools("search my inbox for invoices", _GMAIL_LIKE_TOOLS, max_tools=5)

    assert any("[IntentRouter]" in record.message for record in caplog.records)
    assert any("search my inbox for invoices" in record.message for record in caplog.records)
