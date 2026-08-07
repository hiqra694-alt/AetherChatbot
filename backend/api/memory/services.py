import json
import logging
import re
from typing import List, Optional

from supabase import Client

from api.chat.schemas import Message
from api.memory.schemas import MemoryFact
from providers.factory import ProviderFactory

logger = logging.getLogger(__name__)

# Facts are extracted with Groq regardless of which provider is driving the
# actual conversation -- mirrors ChatService.generate_and_store_title: this
# is a cheap, latency-insensitive background side task, not something the
# user should pay a slower/pricier provider's latency for.
EXTRACTION_PROVIDER = "groq"

MEMORY_EXTRACTION_SYSTEM_PROMPT = (
    "You extract durable, cross-conversation facts about a user from a single chat message. "
    "A durable fact is something that stays true across future conversations: the user's name, "
    "employer, job title, location, or an explicit, stable preference (e.g. 'I prefer Python over "
    "Java', 'I'm vegetarian'). It is NOT a durable fact if it's only true for this one turn or "
    "conversation (e.g. 'I'm tired today', 'please summarize this', a question, or small talk).\n\n"
    "Read the user's message below and respond with ONLY a JSON object of the exact shape "
    '{"facts": ["fact 1", "fact 2"]}, with each fact rewritten as a short, self-contained third-person '
    "statement (e.g. \"Works at Zylo as a backend engineer\"). If the message contains no durable fact, "
    'respond with {"facts": []}. Never include anything other than that JSON object in your response.'
)

_JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


def _parse_extracted_facts(raw_text: str) -> List[str]:
    """
    Parses the extraction model's response into a list of fact strings.
    Tolerates markdown code fences and any leading/trailing prose a small
    model might still emit despite the strict-JSON instruction, by falling
    back to the first {...} substring found. Returns an empty list (never
    raises) if nothing parseable is found.
    """
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())

    candidates = [text]
    match = _JSON_OBJECT_PATTERN.search(text)
    if match:
        candidates.append(match.group(0))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict) and isinstance(parsed.get("facts"), list):
            return [str(f).strip() for f in parsed["facts"] if str(f).strip()]

    return []


async def list_user_memory(supabase: Client, user_id: str) -> List[MemoryFact]:
    """
    Lists every fact stored for `user_id`, most recent first. RLS on
    user_memory already restricts rows to this user; the explicit filter is
    kept for consistency with list_user_documents/list_documents.
    """
    res = (
        supabase.table("user_memory")
        .select("id, fact, created_at")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .execute()
    )
    return [
        MemoryFact(id=row["id"], fact=row["fact"], created_at=row["created_at"])
        for row in (res.data or [])
    ]


async def delete_user_memory(supabase: Client, user_id: str, memory_id: str) -> bool:
    """
    Deletes a single fact belonging to `user_id`. Returns True if a row was
    actually removed, False if no matching row existed (e.g. already
    deleted, or belongs to another user and RLS silently excluded it).
    """
    res = (
        supabase.table("user_memory")
        .delete()
        .eq("user_id", user_id)
        .eq("id", memory_id)
        .execute()
    )
    return len(res.data or []) > 0


def format_memory_for_prompt(facts: List[MemoryFact]) -> str:
    """
    Renders stored facts for injection into the chat system prompt, one
    bullet per fact.
    """
    return "\n".join(f"- {fact.fact}" for fact in facts)


async def extract_and_store_memory(supabase: Client, user_id: Optional[str], user_message: str) -> None:
    """
    Best-effort background task (scheduled from the chat router after every
    turn that includes a user message): asks a fast provider whether the
    user's message stated any durable fact or preference about themselves,
    and if so, persists each new one to `user_memory`. Facts already present
    (case-insensitive exact match) are skipped so memory doesn't accumulate
    duplicate rows across repeated turns.

    Mirrors ChatService.generate_and_store_title's contract: this runs after
    the user-facing response has already been handled, so it must never
    raise -- a missing API key, rate limit, malformed JSON from the model, or
    a DB hiccup should all degrade to "no memory update this turn", not an
    error anywhere visible to the user.
    """
    if not user_id or not user_message or not user_message.strip():
        return

    try:
        provider = ProviderFactory.get_provider(EXTRACTION_PROVIDER)
        extraction_prompt = [
            Message(role="system", content=MEMORY_EXTRACTION_SYSTEM_PROMPT),
            Message(role="user", content=user_message[:2000]),
        ]

        raw_text = ""
        async for chunk in provider.stream_response(extraction_prompt, tools=None):
            if isinstance(chunk, str):
                raw_text += chunk

        new_facts = _parse_extracted_facts(raw_text)
        if not new_facts:
            return

        existing = await list_user_memory(supabase, user_id)
        existing_lower = {f.fact.strip().lower() for f in existing}

        rows = [
            {"user_id": user_id, "fact": fact}
            for fact in new_facts
            if fact.lower() not in existing_lower
        ]
        if not rows:
            return

        supabase.table("user_memory").insert(rows).execute()
        logger.info(f"Stored {len(rows)} new memory fact(s) for user {user_id}.")
    except Exception as mem_err:
        logger.warning(f"Failed to extract/store memory for user {user_id}: {mem_err}")
