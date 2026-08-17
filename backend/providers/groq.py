from openai import AsyncOpenAI, BadRequestError # pyright: ignore [reportMissingImports]
from typing import AsyncGenerator, List, Optional, Set, Tuple
import json
import logging
import re
import uuid
from providers.base import BaseProvider
from core.config import get_settings
from api.chat.schemas import Message
from api.chat.tools import ALL_TOOL_NAMES, repair_tool_arguments

logger = logging.getLogger(__name__)

# Every tool name this app has ever defined (api.chat.tools.ALL_TOOL_NAMES),
# not just the ones offered on a given turn. These are snake_case function
# identifiers that never legitimately appear in ordinary prose, so
# redacting an exact match is a precise, low-false-positive safety net
# against the model naming an internal tool out loud — SYSTEM_PROMPT is the
# primary defense (it explicitly forbids narrating tool use), this just
# catches the rare case where a small model does it anyway. Deriving this
# from the registry (rather than a second hardcoded list) means a tool
# added to tools.py can never silently fall out of sync with this filter.
#
# A prior version of this filter tried to also catch free-form narration
# ("I will use the X tool", "no tool applies", etc.) by buffering entire
# sentences and dropping any sentence that matched. That caused a real
# regression: dropping a whole sentence destroys any real answer content
# that happened to share it with the flagged phrase (e.g. a single run-on
# sentence listing real filenames that also mentioned a tool name lost the
# filenames too), and holding output until each sentence boundary made
# streaming visibly chunkier. Redacting only the exact matched substring,
# on the same per-chunk cadence as the tag-stripping below, avoids both
# problems.
def _build_tool_name_pattern(names) -> "re.Pattern[str]":
    """Compiles the same kind of exact-name redaction pattern _TOOL_NAME_PATTERN
    is, but over an arbitrary `names` set -- used to extend redaction/buffering
    to whatever MCP tool names were merged into this turn's `tools`, in
    addition to the always-present native registry."""
    return re.compile(
        r'\b(?:' + '|'.join(re.escape(n) for n in sorted(names, key=len, reverse=True)) + r')\b',
        re.IGNORECASE,
    )


_TOOL_NAME_PATTERN = _build_tool_name_pattern(ALL_TOOL_NAMES)

# Any complete `<tag>{...json...}</tag>` pair, where the opening and
# closing tag names match exactly (via the \1 backreference) and the
# wrapped content is a JSON object. Deliberately NOT scoped to known tool
# names: llama-3.3-70b-versatile can hallucinate a call to a tool name that
# was never registered at all (e.g. one it half-remembers from training
# rather than from this app's schemas), and that's just as much a leaked
# tool-call artifact as a real tool's name would be. Requiring the content
# to look like a JSON object (rather than matching any `<word>...</word>`)
# is what keeps this from false-positiving on ordinary prose that happens
# to mention an HTML/XML tag (e.g. explaining `<div>hello</div>`), since
# real answers essentially never pair open/close tags around a JSON blob.
_GENERIC_TOOL_TAG_PATTERN = re.compile(
    r'<([a-zA-Z_][a-zA-Z0-9_]{2,40})>\s*(\{.*?\})\s*</\1>',
    re.DOTALL | re.IGNORECASE,
)


def _extract_text_tool_calls(buffer: str, executable_tool_names: Set[str]) -> Tuple[str, list]:
    """
    llama-3.3-70b-versatile occasionally emits a tool call as literal text
    -- `<tag>{...json args...}</tag>` -- instead of populating the API's
    native tool_calls field. Every complete match is stripped out of the
    buffer regardless of whether `tag` names a real, registered tool (see
    _GENERIC_TOOL_TAG_PATTERN), but only matches whose tag is in
    `executable_tool_names` (the tools actually offered to the model this
    turn -- e.g. empty on a post-tool synthesis turn) are returned as
    executable. A tag naming a real tool that just wasn't offered this
    turn, or naming nothing this app has ever registered at all, is still
    removed from the visible stream but silently discarded rather than
    executed -- "never invoke a tool outside what was explicitly offered
    this turn" is enforced here, not just trusted to the model's prompt.

    Returns the buffer with every complete matching tag removed, plus a
    list of {"name": ..., "arguments": ...} dicts (arguments as the raw,
    still-unparsed text between the tags) for only the executable matches,
    in the order they appeared.
    """
    matches = []

    def _capture(m: "re.Match[str]") -> str:
        name = m.group(1).lower()
        if name in executable_tool_names:
            matches.append({"name": name, "arguments": m.group(2).strip()})
        return ''

    buffer = _GENERIC_TOOL_TAG_PATTERN.sub(_capture, buffer)
    return buffer, matches


def _has_unclosed_tool_tag(buffer: str, known_tool_names: Set[str]) -> bool:
    """True if `buffer` currently contains an opening `<tool_name>` for one
    of `known_tool_names` with no matching closing tag yet — i.e. streaming
    is mid-tag and must be held until the closing tag arrives (or the
    stream ends). Deliberately checked against the full tool registry, not
    just this turn's offered tools, so a real tool's tag hallucinated on a
    turn where it wasn't offered (e.g. a synthesis pass) still can't be
    split-leaked across chunk boundaries -- it's discarded as a whole once
    complete, never streamed as a partial prefix."""
    lower = buffer.lower()
    for name in known_tool_names:
        if f'<{name}>' in lower and f'</{name}>' not in lower:
            return True
    return False


def _ends_with_tool_tag_prefix(buffer: str, known_tool_names: Set[str]) -> bool:
    """True if `buffer` ends with a partial prefix of `<tool_name>` for one
    of `known_tool_names` — i.e. the opening tag may still be arriving
    token by token and the buffer must be held one more chunk to avoid
    splitting it across two yields. See _has_unclosed_tool_tag for why this
    checks the full registry rather than just this turn's offered tools."""
    lower = buffer.lower()
    for name in known_tool_names:
        full_tag = f'<{name}>'
        for i in range(1, len(full_tag) + 1):
            if lower.endswith(full_tag[:i]):
                return True
    return False


# Matches the model's own attempted tool call at the start of a Groq
# `failed_generation` string, in either the Llama-native `<function=name>`
# form or this app's `<name>` convention (see _GENERIC_TOOL_TAG_PATTERN
# above). Unanchored at the end and without requiring a closing tag --
# `failed_generation` is frequently truncated mid-argument (that's usually
# *why* Groq rejected it as an invalid tool call in the first place), so
# whatever JSON fragment follows the opening tag is handed to
# repair_tool_arguments rather than requiring a clean match.
_FAILED_GENERATION_FUNCTION_EQ_PATTERN = re.compile(
    r'^<\s*function\s*=\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*>(.*)', re.DOTALL | re.IGNORECASE
)
_FAILED_GENERATION_TAG_PATTERN = re.compile(
    r'^<\s*([a-zA-Z_][a-zA-Z0-9_]{2,40})\s*>(.*)', re.DOTALL | re.IGNORECASE
)


def _recover_tool_call_from_failed_generation(
    failed_generation: Optional[str], executable_tool_names: Set[str]
) -> Optional[dict]:
    """
    Groq validates a tool call's structure server-side *before* streaming
    anything back, and outright rejects the request with a 400
    ("Failed to call a function...") when llama-3.3-70b-versatile's
    generated call doesn't parse -- most commonly because the call was
    truncated mid-argument. The model's actual attempt is preserved in the
    error body's `failed_generation` field, so rather than treating this as
    an unrecoverable failure and silently falling back to tool-calling
    disabled (previously the only path -- see
    ChatService._is_transient_tool_formatting_error), this reconstructs the
    intended call from that raw text: same tag-based extraction convention
    already used for text-leaked tool calls, plus repair_tool_arguments to
    tolerate the truncation itself.

    Returns a {"name": ..., "arguments": <json string>} dict ready to hand
    to ChatService the same way a native tool call would be, or None if
    nothing recoverable could be found (name missing, name not offered this
    turn, or zero arguments recoverable) -- in which case the caller should
    fall back to the existing degrade-gracefully behavior.
    """
    if not failed_generation or not failed_generation.strip():
        return None
    text = failed_generation.strip()

    match = _FAILED_GENERATION_FUNCTION_EQ_PATTERN.match(text) or _FAILED_GENERATION_TAG_PATTERN.match(text)
    if match:
        name = match.group(1).lower()
        if name not in executable_tool_names:
            return None
        rest = re.sub(rf'</\s*{re.escape(name)}\s*>\s*$', '', match.group(2), flags=re.IGNORECASE).strip()
        args = repair_tool_arguments(name, rest)
        if not args:
            return None
        return {"name": name, "arguments": json.dumps(args)}

    # No tag at all -- try a bare {"name": ..., "arguments"/"parameters": {...}}
    # envelope, the other shape a tool-use-tuned model sometimes emits.
    try:
        envelope = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(envelope, dict) or not isinstance(envelope.get("name"), str):
        return None
    name = envelope["name"].lower()
    args = envelope.get("arguments", envelope.get("parameters"))
    if name not in executable_tool_names or not isinstance(args, dict) or not args:
        return None
    return {"name": name, "arguments": json.dumps(args)}


class GroqProvider(BaseProvider):
    def __init__(self):
        settings = get_settings()
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY is not configured.")
        self.client = AsyncOpenAI(api_key=settings.groq_api_key, base_url="https://api.groq.com/openai/v1")

    async def stream_response(self, messages: List[Message], tools: list = None) -> AsyncGenerator[str, None]:
        formatted_messages = []
        for m in messages:
            msg = {
                "role": m.role,
                "content": m.content if m.content is not None else ""
            }
            if m.tool_calls:
                msg["tool_calls"] = m.tool_calls
            if m.tool_call_id:
                msg["tool_call_id"] = m.tool_call_id
            if m.name:
                msg["name"] = m.name
            formatted_messages.append(msg)
            
        kwargs = {
            "model": "openai/gpt-oss-120b",
            "messages": formatted_messages,
            "stream": True
        }
        # Explicitly binding tool_choice="auto" alongside the tool schemas
        # is what actually asks the API for native structured tool calls
        # (delta.tool_calls) on this turn, rather than leaving tool
        # invocation to whatever the model infers from the schemas alone.
        # It does not guarantee llama-3.3-70b-versatile never free-texts a
        # call instead -- the text-tag extraction below exists precisely
        # because it sometimes still does -- but it is the correct native
        # binding and reduces how often that happens.
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        # Tools actually offered to the model this turn -- e.g. empty during
        # a post-tool synthesis turn -- as opposed to ALL_TOOL_NAMES (every
        # tool this app has ever registered). Only a tag naming one of
        # these is eligible for execution; see _extract_text_tool_calls.
        executable_tool_names = set()
        for t in (tools or []):
            name = (t.get("function") or {}).get("name") if isinstance(t, dict) else None
            if name:
                executable_tool_names.add(name.lower())

        # ALL_TOOL_NAMES plus whatever MCP tool names were merged into
        # `tools` this turn (see api.chat.services -- get_merged_tool_schemas),
        # so the tag-buffering/redaction safety nets below recognize a
        # hallucinated MCP tool tag exactly the same way they already do a
        # native one, instead of only ever knowing about the static native
        # registry. Reuses the precompiled _TOOL_NAME_PATTERN unchanged
        # whenever no MCP names are present this turn -- a plain native-only
        # request never pays for a fresh regex compile.
        known_tool_names = ALL_TOOL_NAMES | executable_tool_names
        tool_name_pattern = (
            _TOOL_NAME_PATTERN if known_tool_names == ALL_TOOL_NAMES else _build_tool_name_pattern(known_tool_names)
        )

        try:
            stream = await self.client.chat.completions.create(**kwargs)
        except BadRequestError as bad_req:
            # bad_req.body is the decoded `error` object from Groq's response
            # (the openai SDK unwraps {"error": {...}} down to just the inner
            # dict -- see openai._client._make_status_error), which is where
            # `failed_generation` -- the model's own raw, usually-truncated
            # attempt at a tool call -- lives for a rejected tool-use request.
            body = bad_req.body if isinstance(bad_req.body, dict) else {}
            failed_generation = body.get("failed_generation") if isinstance(body, dict) else None
            recovered = (
                _recover_tool_call_from_failed_generation(failed_generation, executable_tool_names)
                if isinstance(failed_generation, str) else None
            )
            if recovered is None:
                raise
            logger.warning(
                f"Groq rejected tool call '{recovered['name']}' as malformed (failed_generation); "
                "recovered arguments from the raw generation instead of disabling tool-calling."
            )
            yield {
                "type": "tool_calls",
                "tool_calls": [{
                    "id": f"repaired_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {"name": recovered["name"], "arguments": recovered["arguments"]},
                }],
            }
            return

        tool_calls = {}
        buffer = ""

        async for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'tool_calls') and delta.tool_calls:
                    for tc in delta.tool_calls:
                        index = tc.index
                        if index not in tool_calls:
                            tool_calls[index] = {
                                "id": tc.id or "",
                                "type": "function",
                                "function": {
                                    "name": tc.function.name if tc.function and tc.function.name else "",
                                    "arguments": tc.function.arguments if tc.function and tc.function.arguments else ""
                                }
                            }
                        else:
                            if tc.id:
                                tool_calls[index]["id"] = tc.id
                            if tc.function:
                                if tc.function.name:
                                    tool_calls[index]["function"]["name"] += tc.function.name
                                if tc.function.arguments:
                                    tool_calls[index]["function"]["arguments"] += tc.function.arguments
                
                content = delta.content
                if content:
                    buffer += content
                    
                    # 1. Strip complete function/tool tags (with or without leading '<')
                    buffer = re.sub(r'<?function=.*?</function>', '', buffer, flags=re.DOTALL | re.IGNORECASE)
                    buffer = re.sub(r'<?tool_call.*?</tool_call>', '', buffer, flags=re.DOTALL | re.IGNORECASE)

                    # 2. Strip newline-terminated function/tool lines
                    buffer = re.sub(r'(?:^|\n)<?function=[^\n]*\n', '\n', buffer, flags=re.DOTALL | re.IGNORECASE)
                    buffer = re.sub(r'(?:^|\n)<?tool_call[^\n]*\n', '\n', buffer, flags=re.DOTALL | re.IGNORECASE)

                    # 2b. Extract complete <tag>{...}</tag> tool-call-shaped
                    # tags -- the dual-mode fallback for the "tag == the
                    # tool's own name" convention, which patterns 1/2 above
                    # don't recognize. Matched tags are removed from the
                    # buffer (so they never reach the frontend) regardless of
                    # whether the name is a real, currently-offered tool;
                    # only ones in executable_tool_names are handed upstream
                    # as `text_tool_call` events for ChatService to execute.
                    buffer, text_tool_calls = _extract_text_tool_calls(buffer, executable_tool_names)
                    for ttc in text_tool_calls:
                        yield {"type": "text_tool_call", "name": ttc["name"], "arguments": ttc["arguments"]}

                    # 3. If buffer contains an active/unclosed function, tool, or
                    # known-tool-name tag, hold until closed or ended
                    if (re.search(r'<?function=', buffer, re.IGNORECASE)
                            or re.search(r'<?tool_call', buffer, re.IGNORECASE)
                            or _has_unclosed_tool_tag(buffer, known_tool_names)):
                        continue

                    # 4. Check if buffer ends with a potential tag prefix
                    potential_prefixes = [
                        "<", "<f", "<fu", "<fun", "<func", "<funct", "<functi", "<functio", "<function", "<function=",
                        "fun", "func", "funct", "functi", "functio", "function", "function=",
                        "<t", "<to", "<too", "<tool", "<tool_", "<tool_c", "<tool_ca", "<tool_cal", "<tool_call",
                        "too", "tool", "tool_", "tool_c", "tool_ca", "tool_cal", "tool_call"
                    ]
                    is_prefix = False
                    for prefix in potential_prefixes:
                        if buffer.endswith(prefix):
                            is_prefix = True
                            break

                    if is_prefix or _ends_with_tool_tag_prefix(buffer, known_tool_names):
                        continue

                    # 5. Redact exact internal tool-name mentions, then release
                    # immediately — no sentence-boundary holding, so streaming
                    # stays near-real-time.
                    buffer = tool_name_pattern.sub('', buffer)

                    yield buffer
                    buffer = ""

        if buffer:
            buffer, trailing_text_tool_calls = _extract_text_tool_calls(buffer, executable_tool_names)
            for ttc in trailing_text_tool_calls:
                yield {"type": "text_tool_call", "name": ttc["name"], "arguments": ttc["arguments"]}
            buffer = re.sub(r'<?function=.*$', '', buffer, flags=re.DOTALL | re.IGNORECASE)
            buffer = re.sub(r'<?tool_call.*$', '', buffer, flags=re.DOTALL | re.IGNORECASE)
            buffer = tool_name_pattern.sub('', buffer)
            if buffer:
                yield buffer

        if tool_calls:
            yield {
                "type": "tool_calls",
                "tool_calls": list(tool_calls.values())
            }
