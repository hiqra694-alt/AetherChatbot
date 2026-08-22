import asyncio
import logging
from typing import Optional

from supabase import Client, create_client

from api.tasks.services import get_due_tasks, mark_task_notified
from core.config import get_settings

logger = logging.getLogger(__name__)

# How often the background loop wakes up to check for due tasks. Not
# user-configurable yet -- the scheduler is a fixed-cadence polling loop, not
# a per-task cron, so this single interval is all there is to tune.
POLL_INTERVAL_SECONDS = 60


def _get_scheduler_supabase() -> Optional[Client]:
    """
    Builds the service-role-authenticated client the scheduler needs to see
    every user's due tasks (see core.config.Settings.supabase_service_role_key).
    Returns None -- rather than raising -- when it isn't configured, so a
    deployment that hasn't set it up yet still starts cleanly with the
    scheduler simply idling, matching the MCP manager's "no servers
    configured -> no-op" convention in connector_integrations/connector_manager.py.
    """
    settings = get_settings()
    if not settings.supabase_url or not settings.supabase_service_role_key:
        logger.warning(
            "Scheduler: SUPABASE_SERVICE_ROLE_KEY not configured -- background due-task polling is disabled."
        )
        return None
    return create_client(settings.supabase_url, settings.supabase_service_role_key)


async def _poll_due_tasks_once(supabase: Client) -> int:
    """
    Fetches every pending, not-yet-notified task whose due_at has passed and
    flags each as notified. Never raises -- a transient DB error just skips
    this tick, matching the best-effort background-task convention used
    elsewhere in this app (memory extraction, title generation), since a
    failed poll should never take the loop down.
    """
    try:
        due_tasks = await get_due_tasks(supabase)
    except Exception as poll_err:
        logger.error(f"Scheduler: failed to fetch due tasks: {poll_err}")
        return 0

    for task in due_tasks:
        logger.info(f"Task due: [{task.id}] '{task.title}' for user {task.user_id} (was due {task.due_at}).")
        try:
            await mark_task_notified(supabase, task.id)
        except Exception as notify_err:
            logger.error(f"Scheduler: failed to mark task {task.id} as notified: {notify_err}")

    return len(due_tasks)


async def run_scheduler_loop(stop_event: asyncio.Event, interval_seconds: float = POLL_INTERVAL_SECONDS) -> None:
    """
    Background loop started from FastAPI's lifespan (see main.py): polls
    immediately on startup, then every `interval_seconds` until `stop_event`
    is set, at which point it exits promptly instead of finishing out a full
    sleep -- lifespan shutdown awaits this task directly, so a slow exit here
    would delay app shutdown.
    """
    supabase = _get_scheduler_supabase()

    while not stop_event.is_set():
        if supabase is not None:
            await _poll_due_tasks_once(supabase)

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)
        except asyncio.TimeoutError:
            pass
