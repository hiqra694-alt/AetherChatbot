import asyncio
import sys
import os
import pytest
from unittest.mock import AsyncMock, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.tasks.schemas import Task
from core.scheduler import _get_scheduler_supabase, _poll_due_tasks_once, run_scheduler_loop


DUE_TASK = Task(
    id="task-1",
    user_id="user-123",
    title="Call John",
    description=None,
    due_at="2026-08-10T09:00:00+00:00",
    status="pending",
    created_at="2026-08-09T09:00:00+00:00",
    notified_at=None,
)


def test_get_scheduler_supabase_returns_none_when_unconfigured(monkeypatch):
    """
    No SUPABASE_SERVICE_ROLE_KEY set (the default in this test env, and any
    deployment that hasn't opted into the scheduler) must degrade to a no-op
    client rather than raising or attempting a doomed connection.
    """
    fake_settings = MagicMock(supabase_url="https://example.supabase.co", supabase_service_role_key="")
    monkeypatch.setattr("core.scheduler.get_settings", lambda: fake_settings)

    assert _get_scheduler_supabase() is None


def test_get_scheduler_supabase_builds_client_when_configured(monkeypatch):
    fake_settings = MagicMock(supabase_url="https://example.supabase.co", supabase_service_role_key="service-key")
    fake_client = MagicMock()
    monkeypatch.setattr("core.scheduler.get_settings", lambda: fake_settings)
    monkeypatch.setattr("core.scheduler.create_client", lambda url, key: fake_client)

    assert _get_scheduler_supabase() is fake_client


# ==================================================
# _poll_due_tasks_once
# ==================================================

@pytest.mark.asyncio
async def test_poll_due_tasks_once_notifies_each_due_task(monkeypatch):
    fake_get_due_tasks = AsyncMock(return_value=[DUE_TASK])
    fake_mark_notified = AsyncMock()
    monkeypatch.setattr("core.scheduler.get_due_tasks", fake_get_due_tasks)
    monkeypatch.setattr("core.scheduler.mark_task_notified", fake_mark_notified)

    mock_supabase = MagicMock()
    count = await _poll_due_tasks_once(mock_supabase)

    assert count == 1
    fake_mark_notified.assert_awaited_once_with(mock_supabase, "task-1")


@pytest.mark.asyncio
async def test_poll_due_tasks_once_no_due_tasks_is_noop(monkeypatch):
    fake_get_due_tasks = AsyncMock(return_value=[])
    fake_mark_notified = AsyncMock()
    monkeypatch.setattr("core.scheduler.get_due_tasks", fake_get_due_tasks)
    monkeypatch.setattr("core.scheduler.mark_task_notified", fake_mark_notified)

    count = await _poll_due_tasks_once(MagicMock())

    assert count == 0
    fake_mark_notified.assert_not_awaited()


@pytest.mark.asyncio
async def test_poll_due_tasks_once_never_raises_on_query_failure(monkeypatch):
    """A transient DB error on the fetch itself must not propagate -- the
    loop that calls this every tick must be able to keep running."""
    async def raise_error(supabase):
        raise RuntimeError("db unavailable")

    monkeypatch.setattr("core.scheduler.get_due_tasks", raise_error)

    count = await _poll_due_tasks_once(MagicMock())
    assert count == 0


@pytest.mark.asyncio
async def test_poll_due_tasks_once_never_raises_on_notify_failure(monkeypatch):
    """A failure marking one task notified must not stop the others (or the
    loop) -- swallowed and logged, same contract as the fetch failure."""
    fake_get_due_tasks = AsyncMock(return_value=[DUE_TASK])

    async def raise_error(supabase, task_id):
        raise RuntimeError("update failed")

    monkeypatch.setattr("core.scheduler.get_due_tasks", fake_get_due_tasks)
    monkeypatch.setattr("core.scheduler.mark_task_notified", raise_error)

    count = await _poll_due_tasks_once(MagicMock())
    assert count == 1


# ==================================================
# run_scheduler_loop -- clean start/stop
# ==================================================

@pytest.mark.asyncio
async def test_run_scheduler_loop_exits_immediately_when_already_stopped(monkeypatch):
    """If stop_event is already set before the loop starts, it must return
    without ever polling -- lifespan shutdown must not hang."""
    fake_poll = AsyncMock()
    monkeypatch.setattr("core.scheduler._get_scheduler_supabase", lambda: MagicMock())
    monkeypatch.setattr("core.scheduler._poll_due_tasks_once", fake_poll)

    stop_event = asyncio.Event()
    stop_event.set()

    await asyncio.wait_for(run_scheduler_loop(stop_event, interval_seconds=60), timeout=2)

    fake_poll.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_scheduler_loop_polls_then_stops_cleanly(monkeypatch):
    """Simulates FastAPI's lifespan startup/shutdown: the loop is started as
    a background task, polls at least once, and shuts down promptly once
    stop_event is set -- without waiting out the full interval."""
    fake_poll = AsyncMock()
    monkeypatch.setattr("core.scheduler._get_scheduler_supabase", lambda: MagicMock())
    monkeypatch.setattr("core.scheduler._poll_due_tasks_once", fake_poll)

    stop_event = asyncio.Event()
    task = asyncio.create_task(run_scheduler_loop(stop_event, interval_seconds=60))

    # Let the loop run its first poll iteration before stopping it.
    await asyncio.sleep(0.05)
    stop_event.set()

    await asyncio.wait_for(task, timeout=2)

    assert fake_poll.await_count >= 1


@pytest.mark.asyncio
async def test_run_scheduler_loop_skips_polling_when_unconfigured(monkeypatch):
    """When no service-role client is available, the loop must still start
    and stop cleanly -- it just never polls."""
    fake_poll = AsyncMock()
    monkeypatch.setattr("core.scheduler._get_scheduler_supabase", lambda: None)
    monkeypatch.setattr("core.scheduler._poll_due_tasks_once", fake_poll)

    stop_event = asyncio.Event()
    task = asyncio.create_task(run_scheduler_loop(stop_event, interval_seconds=60))

    await asyncio.sleep(0.05)
    stop_event.set()

    await asyncio.wait_for(task, timeout=2)

    fake_poll.assert_not_awaited()
