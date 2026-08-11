import logging
from typing import Optional, Tuple

from fastapi import APIRouter, Depends, Header, HTTPException
from supabase import create_client, Client, ClientOptions

from core.config import get_settings
from api.tasks.schemas import TaskCompleteResponse, TaskDeleteResponse, TaskListResponse
from api.tasks.services import complete_task, delete_task, list_due_reminders, list_tasks

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/tasks", tags=["tasks"])


async def get_authenticated_supabase(authorization: Optional[str] = Header(None)) -> Tuple[Client, str]:
    """
    Dependency mirroring api/memory/routers.py and api/chat/routers.py's
    dependency of the same shape: builds a Supabase client bound to the
    caller's JWT so RLS policies on `tasks` apply, then resolves user_id
    from that verified token instead of trusting a client-supplied value.
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header is required.")

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid token format. Must be Bearer <token>")

    jwt_token = authorization.split(" ")[1]
    settings = get_settings()

    options = ClientOptions(headers={"Authorization": f"Bearer {jwt_token}"})
    supabase = create_client(settings.supabase_url, settings.supabase_anon_key, options=options)

    try:
        user_res = supabase.auth.get_user(jwt_token)
    except Exception as auth_err:
        logger.error(f"Failed to resolve user from token: {auth_err}")
        raise HTTPException(status_code=401, detail="Invalid or expired session token.")

    if not user_res or not user_res.user:
        raise HTTPException(status_code=401, detail="Invalid or expired session token.")

    return supabase, user_res.user.id


@router.get("", response_model=TaskListResponse)
async def get_tasks(
    status: Optional[str] = "pending",
    due_only: bool = False,
    auth: Tuple[Client, str] = Depends(get_authenticated_supabase),
):
    """
    Lists the authenticated user's tasks. `status` filters to 'pending',
    'completed', or 'all' (ignored when `due_only=true`).

    `due_only=true` backs the notification bell: pending tasks the
    background scheduler (core/scheduler.py) has already flagged as due,
    still awaiting the user's acknowledgement.
    """
    supabase, user_id = auth

    try:
        if due_only:
            tasks = await list_due_reminders(supabase, user_id)
        else:
            tasks = await list_tasks(supabase, user_id, status)
    except Exception as db_err:
        logger.error(f"Failed to list tasks for user {user_id}: {db_err}")
        raise HTTPException(status_code=500, detail="Failed to fetch tasks. Please try again later.")

    return TaskListResponse(tasks=tasks)


@router.patch("/{task_id}/complete", response_model=TaskCompleteResponse)
async def complete_task_endpoint(task_id: str, auth: Tuple[Client, str] = Depends(get_authenticated_supabase)):
    """Marks a single task completed."""
    supabase, user_id = auth

    try:
        completed = await complete_task(supabase, user_id, task_id)
    except Exception as db_err:
        logger.error(f"Failed to complete task {task_id} for user {user_id}: {db_err}")
        raise HTTPException(status_code=500, detail="Failed to complete the task. Please try again later.")

    return TaskCompleteResponse(completed=completed)


@router.delete("/{task_id}", response_model=TaskDeleteResponse)
async def delete_task_endpoint(task_id: str, auth: Tuple[Client, str] = Depends(get_authenticated_supabase)):
    """Deletes a single task."""
    supabase, user_id = auth

    try:
        deleted = await delete_task(supabase, user_id, task_id)
    except Exception as db_err:
        logger.error(f"Failed to delete task {task_id} for user {user_id}: {db_err}")
        raise HTTPException(status_code=500, detail="Failed to delete the task. Please try again later.")

    return TaskDeleteResponse(deleted=deleted)
