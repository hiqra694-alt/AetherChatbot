import sys
import os
import pytest
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient  # pyright: ignore [reportMissingImports]

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app  # pyright: ignore [reportMissingImports]

client = TestClient(app)

TASK_ROW = {
    "id": "task-1",
    "user_id": "test-user-id",
    "title": "Call John",
    "description": None,
    "due_at": "2026-08-15T17:00:00+00:00",
    "status": "pending",
    "created_at": "2026-08-10T10:00:00+00:00",
    "notified_at": None,
}


def _make_mock_supabase(auth_user_id="test-user-id"):
    mock_supabase = MagicMock()
    mock_user_res = MagicMock()
    mock_user_res.user.id = auth_user_id
    mock_supabase.auth.get_user.return_value = mock_user_res
    return mock_supabase


def _mock_table(execute_data):
    mock_table = MagicMock()
    for method in ("select", "insert", "update", "delete", "eq", "order", "limit", "is_", "lte", "ilike"):
        getattr(mock_table, method).return_value = mock_table
    mock_table.not_ = mock_table
    mock_res = MagicMock()
    mock_res.data = execute_data
    mock_table.execute.return_value = mock_res
    return mock_table


# ==================================================
# GET /api/tasks
# ==================================================

@patch("api.tasks.routers.create_client")
def test_get_tasks_returns_pending_by_default(mock_create_client):
    mock_supabase = _make_mock_supabase()
    mock_supabase.table.return_value = _mock_table([TASK_ROW])
    mock_create_client.return_value = mock_supabase

    response = client.get("/api/tasks", headers={"Authorization": "Bearer fake_token"})
    assert response.status_code == 200
    body = response.json()
    assert len(body["tasks"]) == 1
    assert body["tasks"][0]["id"] == "task-1"


@patch("api.tasks.routers.create_client")
def test_get_tasks_due_only_uses_reminder_query(mock_create_client):
    mock_supabase = _make_mock_supabase()
    mock_table = _mock_table([{**TASK_ROW, "notified_at": "2026-08-10T09:05:00+00:00"}])
    mock_supabase.table.return_value = mock_table
    mock_create_client.return_value = mock_supabase

    response = client.get("/api/tasks?due_only=true", headers={"Authorization": "Bearer fake_token"})
    assert response.status_code == 200
    body = response.json()
    assert len(body["tasks"]) == 1
    assert body["tasks"][0]["notified_at"] is not None


def test_get_tasks_unauthorized():
    response = client.get("/api/tasks")
    assert response.status_code == 401
    assert "detail" in response.json()


# ==================================================
# PATCH /api/tasks/{task_id}/complete
# ==================================================

@patch("api.tasks.routers.create_client")
def test_complete_task_endpoint_success(mock_create_client):
    mock_supabase = _make_mock_supabase()
    mock_supabase.table.return_value = _mock_table([{**TASK_ROW, "status": "completed"}])
    mock_create_client.return_value = mock_supabase

    response = client.patch("/api/tasks/task-1/complete", headers={"Authorization": "Bearer fake_token"})
    assert response.status_code == 200
    assert response.json()["completed"] is True


@patch("api.tasks.routers.create_client")
def test_complete_task_endpoint_not_found(mock_create_client):
    mock_supabase = _make_mock_supabase()
    mock_supabase.table.return_value = _mock_table([])
    mock_create_client.return_value = mock_supabase

    response = client.patch("/api/tasks/nonexistent/complete", headers={"Authorization": "Bearer fake_token"})
    assert response.status_code == 200
    assert response.json()["completed"] is False


def test_complete_task_endpoint_unauthorized():
    response = client.patch("/api/tasks/task-1/complete")
    assert response.status_code == 401


# ==================================================
# DELETE /api/tasks/{task_id}
# ==================================================

@patch("api.tasks.routers.create_client")
def test_delete_task_endpoint_success(mock_create_client):
    mock_supabase = _make_mock_supabase()
    mock_supabase.table.return_value = _mock_table([TASK_ROW])
    mock_create_client.return_value = mock_supabase

    response = client.delete("/api/tasks/task-1", headers={"Authorization": "Bearer fake_token"})
    assert response.status_code == 200
    assert response.json()["deleted"] is True


@patch("api.tasks.routers.create_client")
def test_delete_task_endpoint_not_found(mock_create_client):
    mock_supabase = _make_mock_supabase()
    mock_supabase.table.return_value = _mock_table([])
    mock_create_client.return_value = mock_supabase

    response = client.delete("/api/tasks/nonexistent", headers={"Authorization": "Bearer fake_token"})
    assert response.status_code == 200
    assert response.json()["deleted"] is False


def test_delete_task_endpoint_unauthorized():
    response = client.delete("/api/tasks/task-1")
    assert response.status_code == 401
