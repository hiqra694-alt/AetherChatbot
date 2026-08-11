import logging
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from supabase import Client

from api.tasks.schemas import Task, TaskStatus

logger = logging.getLogger(__name__)


def _is_valid_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
        return True
    except (ValueError, AttributeError, TypeError):
        return False


async def create_task(
    supabase: Client,
    user_id: str,
    title: str,
    due_at: Optional[str] = None,
    description: Optional[str] = None,
) -> Task:
    """Inserts a new pending task row for `user_id` and returns it as stored."""
    row = {
        "user_id": user_id,
        "title": title,
        "description": description,
        "due_at": due_at,
        "status": TaskStatus.PENDING.value,
    }
    res = supabase.table("tasks").insert(row).execute()
    return Task(**res.data[0])


async def list_tasks(supabase: Client, user_id: str, status: Optional[str] = "pending") -> List[Task]:
    """
    Lists `user_id`'s tasks, most-recently-created first. `status` filters to
    'pending' or 'completed'; pass None or 'all' to return every status.
    """
    query = supabase.table("tasks").select("*").eq("user_id", user_id)
    if status and status != "all":
        query = query.eq("status", status)
    res = query.order("created_at", desc=True).execute()
    return [Task(**row) for row in (res.data or [])]


async def complete_task(supabase: Client, user_id: str, task_id: str) -> bool:
    """
    Marks a single task completed. `task_id` is normally the task's UUID
    (e.g. from a prior list_tasks call), but the chat tool also allows the
    LLM to pass a task's title directly when it was only given a name to go
    on -- so a non-UUID (or a UUID that doesn't match any row) falls back to
    a case-insensitive substring match against a pending task's title.
    Returns False if no matching row existed for this user.
    """
    if _is_valid_uuid(task_id):
        res = (
            supabase.table("tasks")
            .update({"status": TaskStatus.COMPLETED.value})
            .eq("id", task_id)
            .eq("user_id", user_id)
            .execute()
        )
        if res.data:
            return True

    res = (
        supabase.table("tasks")
        .update({"status": TaskStatus.COMPLETED.value})
        .eq("user_id", user_id)
        .eq("status", TaskStatus.PENDING.value)
        .ilike("title", f"%{task_id}%")
        .execute()
    )
    return len(res.data or []) > 0


async def delete_task(supabase: Client, user_id: str, task_id: str) -> bool:
    """Deletes a single task. Returns False if no matching row existed for this user."""
    res = (
        supabase.table("tasks")
        .delete()
        .eq("id", task_id)
        .eq("user_id", user_id)
        .execute()
    )
    return len(res.data or []) > 0


async def list_due_reminders(supabase: Client, user_id: str) -> List[Task]:
    """
    `user_id`'s pending tasks the scheduler has already flagged as due
    (notified_at set) -- i.e. reminders still awaiting the user's
    acknowledgement (via complete_task or delete_task), soonest-due first.
    Backs the notification bell; unlike get_due_tasks, this is a
    per-request, user-JWT-scoped read (RLS-restricted to this one user).
    """
    res = (
        supabase.table("tasks")
        .select("*")
        .eq("user_id", user_id)
        .eq("status", TaskStatus.PENDING.value)
        .not_.is_("notified_at", "null")
        .order("due_at", desc=False)
        .execute()
    )
    return [Task(**row) for row in (res.data or [])]


async def get_due_tasks(supabase: Client) -> List[Task]:
    """
    Every pending, not-yet-notified task whose due_at has passed, across all
    users. Intended only for the scheduler's service-role-authenticated
    client (core/scheduler.py) -- a per-request, user-JWT-scoped client would
    have RLS restrict this to a single user, which defeats the point.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    res = (
        supabase.table("tasks")
        .select("*")
        .eq("status", TaskStatus.PENDING.value)
        .is_("notified_at", "null")
        .lte("due_at", now_iso)
        .execute()
    )
    return [Task(**row) for row in (res.data or [])]


async def mark_task_notified(supabase: Client, task_id: str) -> None:
    """Flags a task as notified so the scheduler doesn't re-alert on it every poll."""
    supabase.table("tasks").update(
        {"notified_at": datetime.now(timezone.utc).isoformat()}
    ).eq("id", task_id).execute()
