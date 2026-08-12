import sys
import os
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient  # pyright: ignore [reportMissingImports]

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app  # pyright: ignore [reportMissingImports]

client = TestClient(app)


def _make_mock_supabase(auth_user_id="test-user-id"):
    mock_supabase = MagicMock()
    mock_user_res = MagicMock()
    mock_user_res.user.id = auth_user_id
    mock_supabase.auth.get_user.return_value = mock_user_res
    return mock_supabase


def _mock_table():
    mock_table = MagicMock()
    mock_table.upsert.return_value = mock_table
    mock_table.execute.return_value = MagicMock(data=[{"user_id": "test-user-id", "provider": "google"}])
    return mock_table


# ==================================================
# GET /api/connectors/status
# ==================================================

def _mock_select_table(rows):
    mock_table = MagicMock()
    for method in ("select", "eq"):
        getattr(mock_table, method).return_value = mock_table
    mock_table.execute.return_value = MagicMock(data=rows)
    return mock_table


@patch("api.connectors.routers.create_client")
def test_status_reports_stored_providers(mock_create_client):
    mock_supabase = _make_mock_supabase()
    mock_table = _mock_select_table([{"provider": "google"}])
    mock_supabase.table.return_value = mock_table
    mock_create_client.return_value = mock_supabase

    response = client.get("/api/connectors/status", headers={"Authorization": "Bearer fake_token"})

    assert response.status_code == 200
    assert response.json() == {"google": True, "github": False}


@patch("api.connectors.routers.create_client")
def test_status_reports_no_providers_stored(mock_create_client):
    mock_supabase = _make_mock_supabase()
    mock_table = _mock_select_table([])
    mock_supabase.table.return_value = mock_table
    mock_create_client.return_value = mock_supabase

    response = client.get("/api/connectors/status", headers={"Authorization": "Bearer fake_token"})

    assert response.status_code == 200
    assert response.json() == {"google": False, "github": False}


@patch("api.connectors.routers.create_client")
def test_status_both_providers_stored(mock_create_client):
    mock_supabase = _make_mock_supabase()
    mock_table = _mock_select_table([{"provider": "google"}, {"provider": "github"}])
    mock_supabase.table.return_value = mock_table
    mock_create_client.return_value = mock_supabase

    response = client.get("/api/connectors/status", headers={"Authorization": "Bearer fake_token"})

    assert response.status_code == 200
    assert response.json() == {"google": True, "github": True}


def test_status_unauthorized():
    response = client.get("/api/connectors/status")
    assert response.status_code == 401
    assert "detail" in response.json()


@patch("api.connectors.routers.create_client")
def test_status_db_failure_returns_500(mock_create_client):
    mock_supabase = _make_mock_supabase()
    mock_table = MagicMock()
    mock_table.select.side_effect = RuntimeError("db unavailable")
    mock_supabase.table.return_value = mock_table
    mock_create_client.return_value = mock_supabase

    response = client.get("/api/connectors/status", headers={"Authorization": "Bearer fake_token"})

    assert response.status_code == 500


# ==================================================
# POST /api/connectors/store-token
# ==================================================

@patch("api.connectors.routers.create_client")
def test_store_token_success(mock_create_client):
    mock_supabase = _make_mock_supabase()
    mock_table = _mock_table()
    mock_supabase.table.return_value = mock_table
    mock_create_client.return_value = mock_supabase

    response = client.post(
        "/api/connectors/store-token",
        json={"provider": "google", "refresh_token": "1//stored-refresh-token"},
        headers={"Authorization": "Bearer fake_token"},
    )

    assert response.status_code == 200
    assert response.json()["stored"] is True

    mock_table.upsert.assert_called_once_with(
        {"user_id": "test-user-id", "provider": "google", "refresh_token": "1//stored-refresh-token"},
        on_conflict="user_id,provider",
    )


@patch("api.connectors.routers.create_client")
def test_store_token_github_provider_accepted(mock_create_client):
    mock_supabase = _make_mock_supabase()
    mock_table = _mock_table()
    mock_supabase.table.return_value = mock_table
    mock_create_client.return_value = mock_supabase

    response = client.post(
        "/api/connectors/store-token",
        json={"provider": "github", "refresh_token": "gho_stored"},
        headers={"Authorization": "Bearer fake_token"},
    )

    assert response.status_code == 200
    assert response.json()["stored"] is True


@patch("api.connectors.routers.create_client")
def test_store_token_rejects_unknown_provider(mock_create_client):
    # Auth must succeed here so the request actually reaches body
    # validation -- an unmocked auth dependency would 401 first (on a real
    # network call against a bogus JWT) and mask the 422 this test exists
    # to check for.
    mock_create_client.return_value = _make_mock_supabase()

    response = client.post(
        "/api/connectors/store-token",
        json={"provider": "facebook", "refresh_token": "abc"},
        headers={"Authorization": "Bearer fake_token"},
    )
    assert response.status_code == 422


@patch("api.connectors.routers.create_client")
def test_store_token_rejects_empty_refresh_token(mock_create_client):
    mock_create_client.return_value = _make_mock_supabase()

    response = client.post(
        "/api/connectors/store-token",
        json={"provider": "google", "refresh_token": ""},
        headers={"Authorization": "Bearer fake_token"},
    )
    assert response.status_code == 422


def test_store_token_unauthorized():
    response = client.post(
        "/api/connectors/store-token",
        json={"provider": "google", "refresh_token": "abc"},
    )
    assert response.status_code == 401
    assert "detail" in response.json()


@patch("api.connectors.routers.create_client")
def test_store_token_db_failure_returns_500(mock_create_client):
    mock_supabase = _make_mock_supabase()
    mock_table = MagicMock()
    mock_table.upsert.side_effect = RuntimeError("db unavailable")
    mock_supabase.table.return_value = mock_table
    mock_create_client.return_value = mock_supabase

    response = client.post(
        "/api/connectors/store-token",
        json={"provider": "google", "refresh_token": "abc"},
        headers={"Authorization": "Bearer fake_token"},
    )

    assert response.status_code == 500


# ==================================================
# DELETE /api/connectors/disconnect
# ==================================================

def _mock_delete_table(deleted_rows):
    mock_table = MagicMock()
    for method in ("delete", "eq"):
        getattr(mock_table, method).return_value = mock_table
    mock_table.execute.return_value = MagicMock(data=deleted_rows)
    return mock_table


@patch("api.connectors.routers.create_client")
def test_disconnect_success_row_existed(mock_create_client):
    mock_supabase = _make_mock_supabase()
    mock_table = _mock_delete_table([{"user_id": "test-user-id", "provider": "google"}])
    mock_supabase.table.return_value = mock_table
    mock_create_client.return_value = mock_supabase

    response = client.delete(
        "/api/connectors/disconnect?provider=google",
        headers={"Authorization": "Bearer fake_token"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert "Google" in body["message"]

    mock_table.delete.assert_called_once()
    mock_table.eq.assert_any_call("user_id", "test-user-id")
    mock_table.eq.assert_any_call("provider", "google")


@patch("api.connectors.routers.create_client")
def test_disconnect_idempotent_when_no_row_existed(mock_create_client):
    """Disconnecting a provider that never captured an offline refresh_token
    (e.g. only ever used with a live in-browser token) must still succeed --
    this endpoint only cleans up its own side-table, it's not the source of
    truth for whether the provider was ever linked."""
    mock_supabase = _make_mock_supabase()
    mock_table = _mock_delete_table([])
    mock_supabase.table.return_value = mock_table
    mock_create_client.return_value = mock_supabase

    response = client.delete(
        "/api/connectors/disconnect?provider=github",
        headers={"Authorization": "Bearer fake_token"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "success"


@patch("api.connectors.routers.create_client")
def test_disconnect_rejects_unknown_provider(mock_create_client):
    mock_create_client.return_value = _make_mock_supabase()

    response = client.delete(
        "/api/connectors/disconnect?provider=facebook",
        headers={"Authorization": "Bearer fake_token"},
    )
    assert response.status_code == 422


@patch("api.connectors.routers.create_client")
def test_disconnect_requires_provider_query_param(mock_create_client):
    # Auth must succeed here so the request reaches query-param validation
    # rather than 401ing first on an unmocked auth dependency -- same
    # ordering quirk as the store-token validation tests above.
    mock_create_client.return_value = _make_mock_supabase()

    response = client.delete(
        "/api/connectors/disconnect",
        headers={"Authorization": "Bearer fake_token"},
    )
    assert response.status_code == 422


def test_disconnect_unauthorized():
    response = client.delete("/api/connectors/disconnect?provider=google")
    assert response.status_code == 401
    assert "detail" in response.json()


@patch("api.connectors.routers.create_client")
def test_disconnect_db_failure_returns_500(mock_create_client):
    mock_supabase = _make_mock_supabase()
    mock_table = MagicMock()
    mock_table.delete.side_effect = RuntimeError("db unavailable")
    mock_supabase.table.return_value = mock_table
    mock_create_client.return_value = mock_supabase

    response = client.delete(
        "/api/connectors/disconnect?provider=google",
        headers={"Authorization": "Bearer fake_token"},
    )

    assert response.status_code == 500
