import logging
import re
from typing import Optional

from supabase import Client

from api.chat.schemas import Message
from api.memory.schemas import UserMemoryProfile
from providers.factory import ProviderFactory

logger = logging.getLogger(__name__)

# The extraction/rewrite step runs with Groq regardless of which provider is
# driving the actual conversation -- mirrors ChatService.generate_and_store_title:
# this is a cheap, latency-insensitive background side task, not something the
# user should pay a slower/pricier provider's latency for.
EXTRACTION_PROVIDER = "groq"

# Upsert/Rewrite prompt: the model is handed the user's current profile
# paragraph plus their latest message and asked to rewrite the whole
# paragraph, rather than being asked to extract isolated facts as separate
# JSON list items. This is what keeps user_memory a single, cohesive,
# evolving narrative instead of an ever-growing set of disconnected rows.
MEMORY_EXTRACTION_SYSTEM_PROMPT = (
    "You maintain a single, cohesive, continuously-evolving narrative paragraph that acts as "
    "a profile of a user across every conversation they have -- their role, employer, active "
    "projects, tech stack, goals, and stable preferences, written in flowing third-person prose "
    "(never bullet points or a list).\n\n"
    "You will be given the CURRENT PROFILE (may be empty, if nothing is known yet) and the "
    "user's LATEST MESSAGE. Rewrite the profile paragraph so it naturally weaves in any new "
    "durable fact the latest message reveals -- especially an active project, a technology or "
    "stack they're working with (e.g. FastAPI, React), a goal, an employer or role, or an "
    "explicit stable preference. Keep everything from the current profile that is still true; "
    "revise a detail only when the latest message clearly updates or contradicts it. Never fold "
    "in a transient, one-turn detail (e.g. 'is tired today', a question, small talk) -- those "
    "are not durable facts.\n\n"
    "Respond with ONLY the rewritten paragraph: plain prose, no labels, no preamble, no "
    "surrounding quotes, no markdown, no JSON. If the latest message contains no durable fact, "
    "respond with the CURRENT PROFILE completely unchanged. If nothing is known yet and the "
    "latest message has no durable fact either, respond with an empty string."
)

_MARKDOWN_FENCE_PATTERN = re.compile(r"^```(?:\w+)?\s*|\s*```$")


def _clean_narrative_response(raw_text: str) -> str:
    """
    Cleans the rewrite model's response into the plain-prose paragraph it
    was asked for, tolerating markdown code fences or surrounding quotes a
    small model might still emit despite the strict "plain prose only"
    instruction. Never raises -- an unparseable/empty response just yields
    an empty string, which the caller treats as "no update this turn".
    """
    text = raw_text.strip()
    text = _MARKDOWN_FENCE_PATTERN.sub("", text).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'"):
        text = text[1:-1].strip()
    return text


async def get_user_memory(supabase: Client, user_id: str) -> Optional[UserMemoryProfile]:
    """
    Fetches the single narrative profile row stored for `user_id`, or None
    if nothing has been learned about them yet. RLS on user_memory already
    restricts rows to this user; the explicit filter is kept for consistency
    with list_user_documents/list_documents.
    """
    res = (
        supabase.table("user_memory")
        .select("narrative, updated_at")
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    rows = res.data or []
    if not rows or not rows[0].get("narrative"):
        return None
    return UserMemoryProfile(narrative=rows[0]["narrative"], updated_at=rows[0]["updated_at"])


async def delete_user_memory(supabase: Client, user_id: str) -> bool:
    """
    Clears `user_id`'s entire memory profile. Returns True if a row actually
    existed and was removed, False if there was nothing stored.
    """
    res = supabase.table("user_memory").delete().eq("user_id", user_id).execute()
    return len(res.data or []) > 0


def format_memory_for_prompt(profile: Optional[UserMemoryProfile]) -> str:
    """
    Renders the stored narrative profile for injection into the chat system
    prompt as a single flowing paragraph -- not a bullet list, since the
    profile itself is already cohesive prose.
    """
    if not profile or not profile.narrative.strip():
        return ""
    return profile.narrative.strip()


async def extract_and_store_memory(supabase: Client, user_id: Optional[str], user_message: str) -> None:
    """
    Best-effort background task (scheduled from the chat router after every
    turn that includes a user message): fetches the user's current master
    memory paragraph, asks a fast provider to rewrite/expand it in light of
    this turn's message, and upserts the single result back into
    `user_memory` for that user_id -- never appends a new row. Mirrors
    ChatService.generate_and_store_title's contract: this runs after the
    user-facing response has already been handled, so it must never raise --
    a missing API key, rate limit, malformed response, or a DB hiccup should
    all degrade to "no memory update this turn", not an error anywhere
    visible to the user.
    """
    if not user_id or not user_message or not user_message.strip():
        return

    try:
        existing_profile = await get_user_memory(supabase, user_id)
        existing_narrative = existing_profile.narrative.strip() if existing_profile else ""

        provider = ProviderFactory.get_provider(EXTRACTION_PROVIDER)
        extraction_prompt = [
            Message(role="system", content=MEMORY_EXTRACTION_SYSTEM_PROMPT),
            Message(
                role="user",
                content=(
                    f"CURRENT PROFILE:\n{existing_narrative or '(empty -- nothing known yet)'}\n\n"
                    f"LATEST MESSAGE:\n{user_message[:2000]}"
                ),
            ),
        ]

        raw_text = ""
        async for chunk in provider.stream_response(extraction_prompt, tools=None):
            if isinstance(chunk, str):
                raw_text += chunk

        new_narrative = _clean_narrative_response(raw_text)
        if not new_narrative or new_narrative == existing_narrative:
            # Either the model found no durable fact this turn, or it echoed
            # the profile back unchanged -- either way, nothing to persist.
            return

        supabase.table("user_memory").upsert(
            {"user_id": user_id, "narrative": new_narrative},
            on_conflict="user_id",
        ).execute()
        logger.info(f"Updated memory profile for user {user_id}.")
    except Exception as mem_err:
        logger.warning(f"Failed to extract/store memory for user {user_id}: {mem_err}")
