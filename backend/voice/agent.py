"""
Voice agent worker (Phase 2-5): the LiveKit worker entrypoint that puts
VOICE_SYSTEM_PROMPT and VoiceTools together into a running AgentSession.

Standalone from the FastAPI app -- this is launched as its own process
(`python -m voice.agent` / `python voice/agent.py`), dialing into whichever
room voice/router.py's POST /api/voice/token issued a matching room-join
token for. The room is named after the chat_session_id (see voice/router.py),
and the joining participant's identity is that user's user_id (also set in
voice/router.py), so both are recovered here from the LiveKit room/participant
rather than re-deriving auth independently.

Every LiveKit-facing credential (WorkerOptions' ws_url/api_key/api_secret,
and each plugin's api_key) is passed explicitly from core.config.get_settings()
rather than left for the SDK to fall back to reading os.environ directly.
That fallback (see e.g. livekit/agents/worker.py's `ws_url or
os.environ.get("LIVEKIT_URL")`) only ever sees real values if something
actually populates os.environ from backend/.env -- and nothing in this app
does that: core.config.Settings uses pydantic-settings to parse .env into
its own model, never into os.environ. Without this, the worker fails fast
with `ValueError: ws_url is required, or set LIVEKIT_URL environment
variable` no matter what's set in .env, regardless of cwd.

Phase 4 (Context Continuity + State Synchronization) and Phase 5 (Latency
Tuning) read/write the exact same `messages` table api/chat/services.py's
ChatService uses for text chat (session_id, role, content columns) -- so a
session that starts in text chat and continues in voice (or vice versa)
sees one continuous history, not two disjoint ones.

All of this module's own blocking Supabase calls are routed through
_ISOLATED_EXECUTOR, a small dedicated thread pool -- never asyncio's default
executor -- so a slow DB round trip can never stall the worker's event loop
(and therefore the live audio pipeline) regardless of what else is sharing
that pool.

=== Why there is no local VAD in this file (Phase 5 revision) ===

An earlier revision used silero.VAD for turn detection. On session start it
produced a severe CPU/latency incident: "inference is slower than realtime"
delays climbing into double-digit seconds, worker CPU pinned near 100%, and
a race warning ("transcript arrives after turn has been committed, consider
raising 'min_delay'") from VAD committing a turn before Deepgram's transcript
for that same audio had actually arrived. Root-causing this (see the
now-superseded diagnosis previously in this docstring, and api/documents/
services.py's get_relevant_context for the other, unrelated blocking-call
bug found and fixed in the same investigation) showed Silero's own
onnxruntime session was already correctly configured (single-threaded,
non-spinning, CPU-only) and fast in isolation -- the CPU storm actually came
from a *different*, uncapped onnxruntime session (fastembed's cross-encoder
reranker, invoked synchronously via VoiceTools.search_knowledge_base) that
happened to run inline on the same event loop. That's fixed at its source
now (see api/documents/services.py), independent of anything here.

Even after that fix, keeping a local VAD in the live audio path means every
one of ~31 inference calls/sec is exposed to whatever else briefly loads
this process (OS scheduling jitter, a GC pause, another thread holding the
GIL) -- and it's the thing computing "is the user still talking", so any of
that jitter shows up directly as audio stutter or a mistimed turn boundary.
Deepgram's STT plugin already does cloud-side endpointing (SpeechStarted /
UtteranceEnd, exposed here via vad_events=True and utterance_end_ms) --
livekit-agents has a first-class, fully-supported turn_detection="stt" mode
built exactly for this (see livekit.agents.voice.turn.TurnDetectionMode's
docstring: '"stt": use speech-to-text result to detect the end of the
user's turn'), with audio_recognition.py's START_OF_SPEECH/END_OF_SPEECH/
transcript handling explicitly branching on vad is None + turn_detection ==
"stt" throughout -- this is a documented mode, not an unsupported workaround.
Dropping silero.VAD entirely and setting vad=None, turn_detection="stt"
removes the local audio-inference workload from this process altogether
(no local model, no per-frame CPU cost, nothing that can develop a
startup backlog) and, as a side effect, removes the min_delay race too:
turn-commit timing is now anchored to the arrival of Deepgram's own
UtteranceEnd/speech_final signal instead of racing an independent local
VAD decision against it.

The env vars set at the very top of this file, before any onnxruntime-
touching import, remain relevant even without Silero: voice.tools_adapter
-> api.documents.services still imports fastembed's cross-encoder for
search_knowledge_base's local rerank step. They're defense-in-depth on top
of that reranker's own threads=1 session option (see _get_reranker in
api/documents/services.py), constraining any lower-level BLAS/OpenMP thread
pool a given onnxruntime build might fall back to underneath individual
ops. They must be set before import time -- these native libraries read
them once, at initialization -- which is why they're the first thing this
module does, ahead of every other import.
"""

import os

# See the module docstring above for why this entire block must run before
# any onnxruntime-touching import (transitively: voice.tools_adapter ->
# api.documents.services -> fastembed, for the local knowledge-base
# reranker). setdefault, not direct assignment, so an operator-supplied
# override in the real process environment (e.g. a deliberately higher
# thread count for a beefier production host) always wins.
for _env_name, _env_value in (
    ("OMP_NUM_THREADS", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("NUMEXPR_NUM_THREADS", "1"),
    ("OMP_WAIT_POLICY", "PASSIVE"),  # idle OpenMP workers sleep instead of busy-spin
):
    os.environ.setdefault(_env_name, _env_value)
del _env_name, _env_value

import asyncio
import collections
import concurrent.futures
import logging
import threading
import time
from typing import Optional

import psutil
from livekit.agents import (
    Agent,
    AgentSession,
    AssignmentTimeoutError,
    ConversationItemAddedEvent,
    JobContext,
    JobRequest,
    TurnHandlingOptions,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.llm import ChatContext
from livekit.plugins import cartesia, deepgram, groq
from supabase import Client, create_client

from core.config import get_settings
from mcp_integration.mcp_manager import mcp_manager
from voice.prompts import VOICE_SYSTEM_PROMPT, build_mcp_tools_prompt_block
from voice.tools_adapter import VoiceTools

logger = logging.getLogger(__name__)

# Phase 4: how many of the most recent persisted turns to seed a new voice
# session's ChatContext with -- enough for the agent to pick up an ongoing
# text conversation without paying for an unbounded history fetch/replay.
# Lowered from 10 -> 5 (Phase 6 latency pass): every seeded turn is replayed
# into the LLM's own prompt on top of the live system prompt + MCP tool
# block on *every* turn thereafter, so this directly sets the token floor
# each request pays against Groq's per-minute token budget (see
# GROQ_LLM_MODEL below) -- 10 turns of history was a meaningful contributor
# to the 8,000 TPM 429s seen on openai/gpt-oss-120b before this pass.
CHAT_HISTORY_LIMIT = 5

# Phase 6 (Latency Tuning): Deepgram's two independent turn-ending signals,
# both wired to livekit-agents' turn_detection="stt" (see the module
# docstring) -- deepgram.stt's _process_stream_event emits END_OF_SPEECH on
# whichever fires first:
#
# - `speech_final` (driven by `endpointing_ms`, Deepgram's actual
#   `endpointing` query param): fires once this many ms of silence follow a
#   final word. This is the primary, fast path -- 300ms is a deliberate
#   balance (Deepgram's own low end) between "commits promptly" and "doesn't
#   cut the user off mid-thought" on a brief pause.
# - `UtteranceEnd` (driven by `utterance_end_ms`): a slower fallback for
#   when speech_final never fires at all (e.g. noisy audio Deepgram can't
#   confidently mark final). Deepgram's API floors this at 1000ms --
#   requesting lower is silently not honored server-side -- so it stays a
#   safety net, not the primary latency lever; STT_ENDPOINTING_MS above is
#   what actually delivers the <500ms turn-commit target.
#
# Both require interim_results=True, which is this plugin's own default.
STT_ENDPOINTING_MS = 300
STT_UTTERANCE_END_MS = 1000

# Phase 6 (Rate Limit Fix): openai/gpt-oss-120b was hitting Groq's 8,000 TPM
# limit on this account/model tier, surfacing as repeated 429 RateLimitError
# retries (visible as exponential-backoff stalls mid-turn -- the opposite of
# the low-latency turn-taking this worker is tuned for elsewhere in this
# file). llama-3.1-8b-instant is Groq's fastest-inference production model
# (see livekit.plugins.groq.models.LLMModels) and comfortably fits within
# an 8,000 TPM budget for a conversational turn -- swapping to it fixes the
# rate limit and cuts LLM time-to-first-token, both at once.
GROQ_LLM_MODEL = "llama-3.1-8b-instant"

# Dedicated thread pool for this module's own blocking, off-loop Supabase
# work (chat-history fetch + turn persistence) -- deliberately NOT asyncio's
# default executor, so a slow DB round trip can never compete with anything
# else (STT/TTS callbacks, the agent's own tool calls) sharing that pool for
# worker-thread time. Small on purpose: this only ever runs a handful of
# quick calls per job, and keeping it small caps how much OS-thread overhead
# one worker process can accrue.
_ISOLATED_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="voice-io")

# Guards mcp_manager.initialize(): unlike main.py's FastAPI process (see its
# lifespan calling mcp_manager.initialize() once at app startup), this voice
# worker is a wholly separate process (see module docstring) that never runs
# that lifespan -- so without this, mcp_manager._connections stays empty
# here forever and VoiceTools.execute_mcp_tool's mcp_manager.call_tool would
# silently find zero connected MCP servers even with e.g. GitHub fully
# configured. Lazily initialized on this process's first job instead of at
# import time, since initialize() makes network connections and belongs in
# an async context; the lock + flag ensure concurrent jobs in the same
# worker process only ever pay the connect cost once.
_mcp_manager_init_lock = asyncio.Lock()
_mcp_manager_initialized = False


async def _ensure_mcp_manager_initialized() -> None:
    global _mcp_manager_initialized
    if _mcp_manager_initialized:
        return
    async with _mcp_manager_init_lock:
        if _mcp_manager_initialized:
            return
        await mcp_manager.initialize()
        _mcp_manager_initialized = True


class _ProcessLoadCalc:
    """
    Replaces livekit-agents' built-in load_fnc (worker.py's
    `_DefaultLoadCalc`), which samples `psutil.cpu_percent()` -- CPU usage
    across *every* process on the machine, not just this worker. That's a
    reasonable proxy on a single-purpose production box where the worker is
    the only thing running, but on any shared host (a dev laptop running a
    browser/IDE/etc. alongside it, or a prod box with neighboring services)
    it measures "how busy is this machine" instead of "can this worker take
    another job". That mismatch is exactly what produced the oscillating
    "worker is at full capacity" / "worker is below capacity" log pairs at
    load ~0.65-0.85 seen at startup and while idling, with no session
    active: the number was always tracking unrelated processes (e.g. the
    browser rendering VoiceModal.tsx's animated VoiceOrb), never this
    worker's own work.

    Scoped instead to this process plus any child processes it spawns for a
    job (relevant when JobExecutorType.PROCESS is used -- see worker.py's
    `_default_job_executor_type`, which is THREAD on Windows and PROCESS
    elsewhere; on THREAD, the job runs in *this* process already, so the
    parent reading alone already captures it). Same moving-average-over-5-
    samples shape as the built-in calculator, so the smoothing behavior
    otherwise matches -- only the underlying signal changes.
    """

    _instance: "Optional[_ProcessLoadCalc]" = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._proc = psutil.Process()
        self._cpu_count = psutil.cpu_count() or 1
        self._samples: collections.deque[float] = collections.deque(maxlen=5)
        self._samples_lock = threading.Lock()
        self._proc.cpu_percent(interval=None)  # primes the internal baseline
        threading.Thread(
            target=self._run, daemon=True, name="voice_process_load_monitor"
        ).start()

    def _run(self) -> None:
        while True:
            time.sleep(0.5)
            try:
                children = self._proc.children(recursive=True)
            except psutil.NoSuchProcess:
                children = []

            total_percent = self._proc.cpu_percent(interval=None)
            for child in children:
                try:
                    total_percent += child.cpu_percent(interval=None)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            with self._samples_lock:
                self._samples.append(min(total_percent / 100.0 / self._cpu_count, 1.0))

    def _get_avg(self) -> float:
        with self._samples_lock:
            if not self._samples:
                return 0.0
            return sum(self._samples) / len(self._samples)

    @classmethod
    def get_load(cls, _worker) -> float:
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = _ProcessLoadCalc()
        return cls._instance._get_avg()


async def _handle_job_request(job_req: JobRequest) -> None:
    """
    Explicit request_fnc, replacing livekit-agents' default (`_default_request_fnc`
    in worker.py, which is just `await ctx.accept()` with no guard of its own).

    `ctx.accept()` tells the LiveKit server this worker will take the job, then
    waits up to ASSIGNMENT_TIMEOUT for the server to confirm the assignment;
    if that confirmation never arrives (e.g. the server picked a different
    worker in the meantime, or a transient network hiccup on the signaling
    connection) it raises AssignmentTimeoutError. livekit-agents' own call
    site around request_fnc already catches broadly and only logs (see
    worker.py's `_job_request_task`), so this isn't closing a hole that
    would otherwise take the whole process down -- but that safety net is an
    internal implementation detail of a third-party library, not a contract
    this file should have to trust silently. Catching explicitly here makes
    the guarantee local and explicit, and gives a clearer, room-scoped log
    line than the SDK's generic one.
    """
    try:
        await job_req.accept()
    except AssignmentTimeoutError:
        logger.warning(
            "Voice: job assignment for room %s timed out waiting for server "
            "confirmation; declining without taking this worker down.",
            job_req.room.name,
        )
    except Exception:
        logger.exception(
            "Voice: unexpected error accepting job request for room %s",
            job_req.room.name,
        )


def _build_service_role_supabase() -> Optional[Client]:
    """
    Service-role Supabase client for this background worker, matching
    core/scheduler.py's convention: the worker has no per-request user JWT
    to authenticate an anon-key client with, so it needs RLS bypassed --
    query-level user_id/session_id filters (see get_relevant_context) are
    what keep retrieval scoped to the right caller, same as the scheduler
    scopes task polling explicitly rather than relying on RLS. Returns None
    -- rather than raising -- when unconfigured, so the worker still starts
    and simply can't use knowledge base retrieval until it's set up.
    """
    settings = get_settings()
    if not (settings.supabase_url and settings.supabase_service_role_key):
        logger.warning(
            "Voice agent: SUPABASE_SERVICE_ROLE_KEY not configured -- "
            "knowledge base retrieval will be unavailable."
        )
        return None
    return create_client(settings.supabase_url, settings.supabase_service_role_key)


def _fetch_recent_chat_context(supabase: Optional[Client], session_id: str) -> ChatContext:
    """
    Phase 4 (Context Continuity): seeds a new ChatContext with the
    CHAT_HISTORY_LIMIT most recent turns already persisted for this
    chat_session_id by the text chat pipeline (see
    api.chat.services.ChatService.persist_message / stream_chat's finally
    block), so a voice session picked up mid-conversation is fully aware of
    what was already said instead of starting cold.

    Best-effort, same contract as VoiceTools.search_knowledge_base: a DB
    error here degrades to an empty ChatContext (with a logged warning)
    rather than blocking the agent from starting at all.
    """
    chat_ctx = ChatContext()
    if not supabase:
        return chat_ctx

    try:
        db_res = (
            supabase.table("messages")
            .select("role, content")
            .eq("session_id", session_id)
            .in_("role", ["user", "assistant"])
            .order("created_at", desc=True)
            .limit(CHAT_HISTORY_LIMIT)
            .execute()
        )
    except Exception as history_err:
        logger.warning(
            "Voice: failed to fetch chat history for session %s, starting with empty context: %s",
            session_id, history_err,
        )
        return chat_ctx

    # Supabase returns newest-first (see .order(desc=True) above); reversed
    # here so messages are appended oldest-first, matching actual
    # conversation order -- ChatContext.add_message appends to the end.
    for row in reversed(db_res.data or []):
        role = row.get("role")
        content = row.get("content")
        if role not in ("user", "assistant") or not content:
            continue
        chat_ctx.add_message(role=role, content=content)

    return chat_ctx


def _persist_voice_message(supabase: Optional[Client], session_id: str, role: str, content: str) -> None:
    """
    Phase 4 (State Synchronization): best-effort insert into `messages`,
    matching ChatService.persist_message's schema exactly (session_id,
    role, content) so voice turns interleave correctly with any text turns
    in the same session's history. Never raises -- a DB hiccup here must
    not crash the LiveKit worker or interrupt the live conversation, same
    contract as ChatService.persist_message.
    """
    if not supabase:
        return
    try:
        supabase.table("messages").insert(
            {"session_id": session_id, "role": role, "content": content}
        ).execute()
    except Exception as save_err:
        logger.error("Voice: failed to save %s message for session %s: %s", role, session_id, save_err)


def _bind_state_sync(session: AgentSession, supabase: Optional[Client], session_id: str) -> None:
    """
    Phase 4 (State Synchronization): persists each finalized conversation
    turn -- both the user's transcribed speech and the agent's spoken
    response -- to the same `messages` table the text chat pipeline writes
    to. `conversation_item_added` fires once per finalized ChatMessage
    (after interruption handling has already settled the final text), so
    this single listener covers both "user finished speaking" and "agent
    finished speaking" without needing separate STT/TTS-completion hooks.

    Isolation: AgentSession's event emitter is synchronous (see
    livekit.agents.voice.agent_session's `self.emit(...)` call sites), so
    the DB write can't be awaited inline here -- it's dispatched onto a
    background task, with the blocking Supabase call itself pushed onto
    _ISOLATED_EXECUTOR (NOT asyncio.to_thread, which would use asyncio's
    default executor -- see module docstring) so it never stalls the
    worker's event loop. _persist_voice_message already catches and logs
    its own DB errors, so a failed insert never surfaces as an unhandled
    task exception either.
    """
    background_tasks: set[asyncio.Task] = set()

    @session.on("conversation_item_added")
    def _on_conversation_item_added(event: ConversationItemAddedEvent) -> None:
        item = event.item
        if not isinstance(item, llm.ChatMessage) or item.role not in ("user", "assistant"):
            return

        logger.info(f"🎙️ [Voice {item.role.upper()}]: {item.content}")

        text = item.text_content
        if not text:
            return

        loop = asyncio.get_running_loop()
        task = asyncio.ensure_future(
            loop.run_in_executor(_ISOLATED_EXECUTOR, _persist_voice_message, supabase, session_id, item.role, text)
        )
        background_tasks.add(task)
        task.add_done_callback(background_tasks.discard)


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    settings = get_settings()

    # The room is named after chat_session_id and the joining participant's
    # identity is that user's user_id -- both minted server-side by
    # voice/router.py's POST /api/voice/token, so recovering them here
    # rather than re-authenticating is safe.
    participant = await ctx.wait_for_participant()
    session_id = ctx.room.name
    user_id = participant.identity

    supabase = _build_service_role_supabase()

    voice_tools = VoiceTools()
    voice_tools.bind_context(supabase, user_id, session_id)

    # Phase 4/6: seed the agent with this session's recent history, and make
    # sure this process's mcp_manager has connected to its configured MCP
    # servers (see _ensure_mcp_manager_initialized above) so the prompt can
    # list what's actually reachable through execute_mcp_tool -- otherwise
    # the model has no way to know an MCP tool exists at all. See
    # build_mcp_tools_prompt_block's docstring for why this is injected
    # into the prompt rather than registered as individual per-tool
    # @function_tool methods.
    #
    # Neither depends on the other's result, and `ctx.connect()` /
    # `wait_for_participant()` above have already put the room's media
    # negotiation in flight -- so both run concurrently via asyncio.gather
    # instead of one after another, and neither is on the media path at
    # all: the Supabase fetch is off-loop on _ISOLATED_EXECUTOR (never
    # asyncio's default executor, so a slow DB round trip can't compete
    # with STT/TTS callbacks for worker-thread time -- see module
    # docstring), and mcp_manager's own connect()s are already async I/O.
    # This only delays when the *agent* starts talking, never the room join
    # itself.
    loop = asyncio.get_running_loop()
    chat_ctx, _ = await asyncio.gather(
        loop.run_in_executor(_ISOLATED_EXECUTOR, _fetch_recent_chat_context, supabase, session_id),
        _ensure_mcp_manager_initialized(),
    )
    instructions = VOICE_SYSTEM_PROMPT + build_mcp_tools_prompt_block(mcp_manager.list_cached_tools())

    agent_instance = Agent(instructions=instructions, tools=[voice_tools], chat_ctx=chat_ctx)

    # api_key is passed explicitly to every plugin below rather than left
    # for them to fall back to reading os.environ -- see the module
    # docstring's note on why relying on that fallback silently breaks
    # regardless of what's set in .env.
    #
    # vad=None, turn_handling={"turn_detection": "stt"}: no local VAD model
    # at all -- turn taking is driven entirely by Deepgram's own cloud-side
    # endpointing (vad_events=True's SpeechStarted, and utterance_end_ms's
    # UtteranceEnd, both already wired to the SDK's START_OF_SPEECH/
    # END_OF_SPEECH events for this mode). See the module docstring for why
    # this replaced silero.VAD here. turn_handling (not the top-level
    # turn_detection= kwarg) is the current, non-deprecated way to set this
    # -- passing turn_detection= directly logs "turn_detection is
    # deprecated and will be removed in v2.0".
    #
    # preemptive_generation.preemptive_tts=True: AgentSession already runs
    # the LLM preemptively against interim (not-yet-finalized) STT results
    # by default (preemptive_generation.enabled defaults to True -- see
    # livekit.agents.voice.turn._PREEMPTIVE_GENERATION_DEFAULTS), but
    # preemptive_tts itself defaults to False, meaning Cartesia synthesis
    # only starts once the turn is confirmed over -- so time-to-first-audio
    # is (STT endpointing delay) + (TTS time-to-first-byte) even though the
    # LLM's response text was often already sitting there ready. Setting
    # this lets Cartesia start streaming audio for that preemptive response
    # immediately too, cutting the dead-air gap at the start of each turn
    # that reads as "stutter" -- the actual low-latency lever available
    # here, as opposed to Cartesia's `speed` (a voice-pacing knob, not a
    # synthesis-latency one: speaking faster does not make the model
    # synthesize faster, it just makes the resulting audio shorter, which
    # if anything makes an actual mid-stream buffer underrun *more* likely,
    # not less, since there's less audio duration in flight to absorb a
    # synthesis stall) or swapping the `sonic-3` model for a faster one
    # (a real quality/latency trade-off, not something to flip silently).
    # The cost of a wrong preemptive guess (the turn wasn't actually over)
    # is bounded by max_retries/max_speech_duration, both left at their
    # defaults.
    session = AgentSession(
        stt=deepgram.STT(
            api_key=settings.deepgram_api_key,
            vad_events=True,
            endpointing_ms=STT_ENDPOINTING_MS,
            utterance_end_ms=STT_UTTERANCE_END_MS,
        ),
        llm=groq.LLM(model=GROQ_LLM_MODEL, api_key=settings.groq_api_key),
        tts=cartesia.TTS(api_key=settings.cartesia_api_key),
        vad=None,
        turn_handling=TurnHandlingOptions(
            turn_detection="stt",
            preemptive_generation={"preemptive_tts": True},
        ),
    )

    _bind_state_sync(session, supabase, session_id)

    await session.start(room=ctx.room, agent=agent_instance)


if __name__ == "__main__":
    settings = get_settings()
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            ws_url=settings.livekit_url,
            api_key=settings.livekit_api_key,
            api_secret=settings.livekit_api_secret,
            # See _ProcessLoadCalc above: measures this worker's own CPU
            # usage instead of livekit-agents' default system-wide reading.
            load_fnc=_ProcessLoadCalc.get_load,
            # See _handle_job_request above: makes AssignmentTimeoutError
            # handling explicit and room-scoped rather than relying on the
            # SDK's own internal catch-and-log.
            request_fnc=_handle_job_request,
        )
    )
