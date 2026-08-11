import sys
import os
import json
import pytest
from unittest.mock import MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.tasks.schemas import Task, TaskStatus
from api.tasks.services import (
    _is_valid_uuid,
    complete_task,
    create_task,
    get_due_tasks,
    list_tasks,
    mark_task_notified,
)
from api.chat.tools import execute_tool
from api.chat import tools as chat_tools


TASK_ROW = {
    "id": "task-1",
    "user_id": "user-123",
    "title": "Call John",
    "description": None,
    "due_at": "2026-08-15T17:00:00+00:00",
    "status": "pending",
    "created_at": "2026-08-10T10:00:00+00:00",
    "notified_at": None,
}


VALID_UUID = "11111111-1111-1111-1111-111111111111"


def _mock_table(execute_data):
    mock_table = MagicMock()
    for method in ("select", "insert", "update", "delete", "eq", "order", "limit", "is_", "lte", "ilike"):
        getattr(mock_table, method).return_value = mock_table
    mock_res = MagicMock()
    mock_res.data = execute_data
    mock_table.execute.return_value = mock_res
    return mock_table


# ==================================================
# api.tasks.services -- CRUD
# ==================================================

@pytest.mark.asyncio
async def test_create_task_returns_stored_row():
    mock_table = _mock_table([TASK_ROW])
    mock_supabase = MagicMock()
    mock_supabase.table.return_value = mock_table

    task = await create_task(mock_supabase, "user-123", "Call John", due_at="2026-08-15T17:00:00Z")

    assert isinstance(task, Task)
    assert task.id == "task-1"
    assert task.status == TaskStatus.PENDING
    mock_table.insert.assert_called_once_with(
        {
            "user_id": "user-123",
            "title": "Call John",
            "description": None,
            "due_at": "2026-08-15T17:00:00Z",
            "status": "pending",
        }
    )


@pytest.mark.asyncio
async def test_list_tasks_filters_by_status():
    mock_table = _mock_table([TASK_ROW])
    mock_supabase = MagicMock()
    mock_supabase.table.return_value = mock_table

    tasks = await list_tasks(mock_supabase, "user-123", status="pending")

    assert len(tasks) == 1
    assert tasks[0].id == "task-1"
    # eq called twice: once for user_id, once for status
    assert mock_table.eq.call_count == 2
    mock_table.eq.assert_any_call("status", "pending")


@pytest.mark.asyncio
async def test_list_tasks_all_status_skips_status_filter():
    mock_table = _mock_table([TASK_ROW])
    mock_supabase = MagicMock()
    mock_supabase.table.return_value = mock_table

    await list_tasks(mock_supabase, "user-123", status="all")

    # Only the user_id filter should have been applied.
    assert mock_table.eq.call_count == 1
    mock_table.eq.assert_called_once_with("user_id", "user-123")


@pytest.mark.asyncio
async def test_complete_task_returns_true_when_row_updated():
    mock_table = _mock_table([{**TASK_ROW, "status": "completed"}])
    mock_supabase = MagicMock()
    mock_supabase.table.return_value = mock_table

    result = await complete_task(mock_supabase, "user-123", "task-1")

    assert result is True
    mock_table.update.assert_called_once_with({"status": "completed"})


@pytest.mark.asyncio
async def test_complete_task_returns_false_when_no_matching_row():
    mock_table = _mock_table([])
    mock_supabase = MagicMock()
    mock_supabase.table.return_value = mock_table

    result = await complete_task(mock_supabase, "user-123", "nonexistent")

    assert result is False


def test_is_valid_uuid():
    assert _is_valid_uuid(VALID_UUID) is True
    assert _is_valid_uuid("Call John") is False
    assert _is_valid_uuid("") is False


@pytest.mark.asyncio
async def test_complete_task_matches_exact_uuid_without_title_fallback():
    mock_table = _mock_table([{**TASK_ROW, "id": VALID_UUID, "status": "completed"}])
    mock_supabase = MagicMock()
    mock_supabase.table.return_value = mock_table

    result = await complete_task(mock_supabase, "user-123", VALID_UUID)

    assert result is True
    mock_table.eq.assert_any_call("id", VALID_UUID)
    mock_table.ilike.assert_not_called()


@pytest.mark.asyncio
async def test_complete_task_falls_back_to_title_when_uuid_has_no_match():
    mock_table = MagicMock()
    for method in ("select", "insert", "update", "delete", "eq", "order", "limit", "is_", "lte", "ilike"):
        getattr(mock_table, method).return_value = mock_table
    no_match = MagicMock(data=[])
    title_match = MagicMock(data=[{**TASK_ROW, "id": VALID_UUID, "status": "completed"}])
    mock_table.execute.side_effect = [no_match, title_match]

    mock_supabase = MagicMock()
    mock_supabase.table.return_value = mock_table

    result = await complete_task(mock_supabase, "user-123", VALID_UUID)

    assert result is True
    mock_table.ilike.assert_called_once_with("title", f"%{VALID_UUID}%")


@pytest.mark.asyncio
async def test_complete_task_matches_by_title_when_id_is_not_a_uuid():
    mock_table = _mock_table([{**TASK_ROW, "status": "completed"}])
    mock_supabase = MagicMock()
    mock_supabase.table.return_value = mock_table

    result = await complete_task(mock_supabase, "user-123", "call john")

    assert result is True
    mock_table.ilike.assert_called_once_with("title", "%call john%")
    # Only the title-fallback update ran -- the exact-UUID branch is skipped
    # entirely for a non-UUID task_id.
    assert mock_table.update.call_count == 1


@pytest.mark.asyncio
async def test_get_due_tasks_queries_pending_unnotified_past_due():
    mock_table = _mock_table([TASK_ROW])
    mock_supabase = MagicMock()
    mock_supabase.table.return_value = mock_table

    tasks = await get_due_tasks(mock_supabase)

    assert len(tasks) == 1
    mock_table.eq.assert_called_once_with("status", "pending")
    mock_table.is_.assert_called_once_with("notified_at", "null")
    assert mock_table.lte.call_count == 1


@pytest.mark.asyncio
async def test_mark_task_notified_updates_notified_at():
    mock_table = _mock_table([TASK_ROW])
    mock_supabase = MagicMock()
    mock_supabase.table.return_value = mock_table

    await mark_task_notified(mock_supabase, "task-1")

    assert mock_table.update.call_count == 1
    updated_fields = mock_table.update.call_args[0][0]
    assert "notified_at" in updated_fields
    mock_table.eq.assert_called_once_with("id", "task-1")


# ==================================================
# api.chat.tools -- create_task / list_tasks / complete_task wrappers
# ==================================================

@pytest.mark.asyncio
async def test_create_task_tool_requires_user_id():
    res = await chat_tools.create_task(MagicMock(), None, "Call John")
    parsed = json.loads(res)
    assert "error" in parsed


@pytest.mark.asyncio
async def test_create_task_tool_requires_nonempty_title():
    res = await chat_tools.create_task(MagicMock(), "user-123", "   ")
    parsed = json.loads(res)
    assert "error" in parsed


@pytest.mark.asyncio
async def test_create_task_tool_success(monkeypatch):
    fake_task = Task(**TASK_ROW)

    async def fake_create_task_row(supabase, user_id, title, due_at=None, description=None):
        assert user_id == "user-123"
        assert title == "Call John"
        return fake_task

    monkeypatch.setattr("api.chat.tools.create_task_row", fake_create_task_row)

    res = await chat_tools.create_task(MagicMock(), "user-123", "Call John")
    parsed = json.loads(res)

    assert parsed["result"] == "Task created."
    assert parsed["task"]["id"] == "task-1"


@pytest.mark.asyncio
async def test_create_task_tool_swallows_db_error(monkeypatch):
    async def raise_error(*args, **kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr("api.chat.tools.create_task_row", raise_error)

    res = await chat_tools.create_task(MagicMock(), "user-123", "Call John")
    parsed = json.loads(res)
    assert "error" in parsed


@pytest.mark.asyncio
async def test_list_tasks_tool_requires_user_id():
    res = await chat_tools.list_tasks(MagicMock(), None)
    parsed = json.loads(res)
    assert "error" in parsed


@pytest.mark.asyncio
async def test_list_tasks_tool_empty_result(monkeypatch):
    async def fake_list_tasks_row(supabase, user_id, status="pending"):
        return []

    monkeypatch.setattr("api.chat.tools.list_tasks_row", fake_list_tasks_row)

    res = await chat_tools.list_tasks(MagicMock(), "user-123")
    parsed = json.loads(res)
    assert "No tasks found" in parsed["result"]


@pytest.mark.asyncio
async def test_list_tasks_tool_returns_serialized_tasks(monkeypatch):
    fake_task = Task(**TASK_ROW)

    async def fake_list_tasks_row(supabase, user_id, status="pending"):
        return [fake_task]

    monkeypatch.setattr("api.chat.tools.list_tasks_row", fake_list_tasks_row)

    res = await chat_tools.list_tasks(MagicMock(), "user-123")
    parsed = json.loads(res)

    assert len(parsed["tasks"]) == 1
    assert parsed["tasks"][0]["id"] == "task-1"


@pytest.mark.asyncio
async def test_complete_task_tool_requires_user_id():
    res = await chat_tools.complete_task(MagicMock(), None, "task-1")
    parsed = json.loads(res)
    assert "error" in parsed


@pytest.mark.asyncio
async def test_complete_task_tool_requires_task_id():
    res = await chat_tools.complete_task(MagicMock(), "user-123", "")
    parsed = json.loads(res)
    assert "error" in parsed


@pytest.mark.asyncio
async def test_complete_task_tool_not_found(monkeypatch):
    async def fake_complete_task_row(supabase, user_id, task_id):
        return False

    monkeypatch.setattr("api.chat.tools.complete_task_row", fake_complete_task_row)

    res = await chat_tools.complete_task(MagicMock(), "user-123", "nonexistent")
    parsed = json.loads(res)
    assert "error" in parsed


@pytest.mark.asyncio
async def test_complete_task_tool_success(monkeypatch):
    async def fake_complete_task_row(supabase, user_id, task_id):
        return True

    monkeypatch.setattr("api.chat.tools.complete_task_row", fake_complete_task_row)

    res = await chat_tools.complete_task(MagicMock(), "user-123", "task-1")
    parsed = json.loads(res)
    assert parsed["result"] == "Task marked as completed."


# ==================================================
# execute_tool dispatcher wiring
# ==================================================

@pytest.mark.asyncio
async def test_execute_tool_dispatches_create_task(monkeypatch):
    fake_task = Task(**TASK_ROW)

    async def fake_create_task_row(supabase, user_id, title, due_at=None, description=None):
        return fake_task

    monkeypatch.setattr("api.chat.tools.create_task_row", fake_create_task_row)

    res = await execute_tool(
        "create_task", {"title": "Call John", "due_at": "2026-08-15T17:00:00Z"}, MagicMock(), "sess_1", "user-123"
    )
    parsed = json.loads(res)
    assert parsed["result"] == "Task created."


@pytest.mark.asyncio
async def test_execute_tool_dispatches_list_tasks(monkeypatch):
    async def fake_list_tasks_row(supabase, user_id, status="pending"):
        assert status == "completed"
        return []

    monkeypatch.setattr("api.chat.tools.list_tasks_row", fake_list_tasks_row)

    res = await execute_tool("list_tasks", {"status": "completed"}, MagicMock(), "sess_1", "user-123")
    parsed = json.loads(res)
    assert "No tasks found" in parsed["result"]


@pytest.mark.asyncio
async def test_execute_tool_dispatches_complete_task(monkeypatch):
    async def fake_complete_task_row(supabase, user_id, task_id):
        assert task_id == "task-1"
        return True

    monkeypatch.setattr("api.chat.tools.complete_task_row", fake_complete_task_row)

    res = await execute_tool("complete_task", {"task_id": "task-1"}, MagicMock(), "sess_1", "user-123")
    parsed = json.loads(res)
    assert parsed["result"] == "Task marked as completed."
