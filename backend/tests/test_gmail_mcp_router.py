import sys
import os
from unittest.mock import patch, AsyncMock, MagicMock

from fastapi.testclient import TestClient  # pyright: ignore [reportMissingImports]

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app  # pyright: ignore [reportMissingImports]

client = TestClient(app)


class _FakeSettings:
    def __init__(
        self,
        gmail_mcp_client_id="",
        gmail_mcp_client_secret="",
        gmail_mcp_redirect_uri="",
        gmail_mcp_dev_mode=False,
    ):
        self.gmail_mcp_client_id = gmail_mcp_client_id
        self.gmail_mcp_client_secret = gmail_mcp_client_secret
        self.gmail_mcp_redirect_uri = gmail_mcp_redirect_uri
        self.gmail_mcp_dev_mode = gmail_mcp_dev_mode


_CONFIGURED_SETTINGS = _FakeSettings(
    gmail_mcp_client_id="client-id",
    gmail_mcp_client_secret="client-secret",
    gmail_mcp_redirect_uri="https://app.example.com/api/connectors/gmail-mcp/callback",
)

_CONFIGURED_DEV_MODE_SETTINGS = _FakeSettings(
    gmail_mcp_client_id="client-id",
    gmail_mcp_client_secret="client-secret",
    gmail_mcp_redirect_uri="https://app.example.com/api/connectors/gmail-mcp/callback",
    gmail_mcp_dev_mode=True,
)


# ==================================================
# GET /api/connectors/gmail-mcp/authorize
# ==================================================

@patch("mcp_integration.gmail_mcp_router.get_settings", return_value=_FakeSettings())
def test_authorize_not_configured_returns_503(mock_settings):
    response = client.get("/api/connectors/gmail-mcp/authorize", params={"access_token": "tok"})
    assert response.status_code == 503


@patch("mcp_integration.gmail_mcp_router.get_settings", return_value=_CONFIGURED_SETTINGS)
@patch("mcp_integration.gmail_mcp_router.resolve_user_id_from_access_token", new_callable=AsyncMock)
def test_authorize_invalid_token_returns_401(mock_resolve, mock_settings):
    mock_resolve.side_effect = ValueError("bad token")

    response = client.get("/api/connectors/gmail-mcp/authorize", params={"access_token": "bad"})

    assert response.status_code == 401


@patch("mcp_integration.gmail_mcp_router.get_settings", return_value=_CONFIGURED_SETTINGS)
@patch("mcp_integration.gmail_mcp_router.resolve_user_id_from_access_token", new_callable=AsyncMock)
@patch("mcp_integration.gmail_mcp_router.sign_state", return_value="signed-state-value")
def test_authorize_redirects_to_google_consent_screen(mock_sign_state, mock_resolve, mock_settings):
    mock_resolve.return_value = "user-123"

    response = client.get(
        "/api/connectors/gmail-mcp/authorize",
        params={"access_token": "live-token"},
        follow_redirects=False,
    )

    assert response.status_code == 302
    location = response.headers["location"]
    assert location.startswith("https://accounts.google.com/o/oauth2/v2/auth?")
    assert "access_type=offline" in location
    assert "prompt=consent" in location
    assert "state=signed-state-value" in location
    mock_resolve.assert_awaited_once_with("live-token")


def test_authorize_missing_access_token_returns_422():
    response = client.get("/api/connectors/gmail-mcp/authorize")
    assert response.status_code == 422


@patch("mcp_integration.gmail_mcp_router.get_settings", return_value=_CONFIGURED_SETTINGS)
def test_authorize_dev_user_id_rejected_when_dev_mode_off(mock_settings):
    """The dev_user_id shortcut must be a hard no-op (403, not silently
    accepted) whenever GMAIL_MCP_DEV_MODE isn't explicitly true -- this is
    what keeps the shortcut from ever working by accident in a deployed
    environment that forgot to unset it."""
    response = client.get(
        "/api/connectors/gmail-mcp/authorize",
        params={"dev_user_id": "test-user-123"},
    )
    assert response.status_code == 403


@patch("mcp_integration.gmail_mcp_router.get_settings", return_value=_CONFIGURED_DEV_MODE_SETTINGS)
@patch("mcp_integration.gmail_mcp_router.resolve_user_id_from_access_token", new_callable=AsyncMock)
@patch("mcp_integration.gmail_mcp_router.sign_state", return_value="signed-state-value")
def test_authorize_dev_user_id_redirects_when_dev_mode_on(mock_sign_state, mock_resolve, mock_settings):
    """With GMAIL_MCP_DEV_MODE=true, dev_user_id stands in for a verified
    access_token -- no Supabase verification call is made at all -- and the
    flow otherwise redirects to Google exactly like the real path."""
    response = client.get(
        "/api/connectors/gmail-mcp/authorize",
        params={"dev_user_id": "test-user-123"},
        follow_redirects=False,
    )

    assert response.status_code == 302
    location = response.headers["location"]
    assert location.startswith("https://accounts.google.com/o/oauth2/v2/auth?")
    mock_sign_state.assert_called_once_with("test-user-123")
    mock_resolve.assert_not_called()


@patch("mcp_integration.gmail_mcp_router.get_settings", return_value=_CONFIGURED_DEV_MODE_SETTINGS)
@patch("mcp_integration.gmail_mcp_router.resolve_user_id_from_access_token", new_callable=AsyncMock)
def test_authorize_live_access_token_still_wins_over_dev_user_id(mock_resolve, mock_settings):
    """A real access_token always takes precedence, even with dev mode on
    and a dev_user_id also present -- dev_user_id is only ever a fallback
    for when there's no real session to test with."""
    mock_resolve.return_value = "resolved-from-token"

    response = client.get(
        "/api/connectors/gmail-mcp/authorize",
        params={"access_token": "live-token", "dev_user_id": "test-user-123"},
        follow_redirects=False,
    )

    assert response.status_code == 302
    mock_resolve.assert_awaited_once_with("live-token")


# ==================================================
# GET /api/connectors/gmail-mcp/callback
# ==================================================

def test_callback_google_error_returns_400_html():
    response = client.get(
        "/api/connectors/gmail-mcp/callback",
        params={"error": "access_denied"},
    )
    assert response.status_code == 400
    assert "access_denied" in response.text


def test_callback_missing_code_or_state_returns_400():
    response = client.get("/api/connectors/gmail-mcp/callback", params={"code": "abc"})
    assert response.status_code == 400

    response2 = client.get("/api/connectors/gmail-mcp/callback", params={"state": "abc"})
    assert response2.status_code == 400


@patch("mcp_integration.gmail_mcp_router.verify_state")
def test_callback_invalid_state_returns_400(mock_verify_state):
    mock_verify_state.side_effect = ValueError("bad state")

    response = client.get(
        "/api/connectors/gmail-mcp/callback",
        params={"code": "auth-code", "state": "tampered"},
    )

    assert response.status_code == 400


@patch("mcp_integration.gmail_mcp_router.verify_state", return_value="user-123")
@patch("mcp_integration.gmail_mcp_router.exchange_code_for_tokens", new_callable=AsyncMock)
def test_callback_token_exchange_failure_returns_502(mock_exchange, mock_verify_state):
    mock_exchange.side_effect = RuntimeError("google unreachable")

    response = client.get(
        "/api/connectors/gmail-mcp/callback",
        params={"code": "auth-code", "state": "valid-state"},
    )

    assert response.status_code == 502


@patch("mcp_integration.gmail_mcp_router.verify_state", return_value="user-123")
@patch("mcp_integration.gmail_mcp_router.exchange_code_for_tokens", new_callable=AsyncMock)
def test_callback_no_refresh_token_returns_502(mock_exchange, mock_verify_state):
    mock_exchange.return_value = {"access_token": "at"}  # no refresh_token

    response = client.get(
        "/api/connectors/gmail-mcp/callback",
        params={"code": "auth-code", "state": "valid-state"},
    )

    assert response.status_code == 502


@patch("mcp_integration.gmail_mcp_router.verify_state", return_value="user-123")
@patch("mcp_integration.gmail_mcp_router.exchange_code_for_tokens", new_callable=AsyncMock)
@patch("mcp_integration.gmail_mcp_router.build_service_role_client", return_value=None)
def test_callback_no_service_role_key_returns_503(mock_build_client, mock_exchange, mock_verify_state):
    mock_exchange.return_value = {"access_token": "at", "refresh_token": "rt"}

    response = client.get(
        "/api/connectors/gmail-mcp/callback",
        params={"code": "auth-code", "state": "valid-state"},
    )

    assert response.status_code == 503


@patch("mcp_integration.gmail_mcp_router.verify_state", return_value="user-123")
@patch("mcp_integration.gmail_mcp_router.exchange_code_for_tokens", new_callable=AsyncMock)
@patch("mcp_integration.gmail_mcp_router.build_service_role_client")
@patch("mcp_integration.gmail_mcp_router.store_gmail_mcp_tokens", new_callable=AsyncMock)
def test_callback_success_stores_token_under_distinct_provider(
    mock_store, mock_build_client, mock_exchange, mock_verify_state
):
    mock_exchange.return_value = {"access_token": "at", "refresh_token": "stored-refresh-token"}
    fake_supabase = MagicMock()
    mock_build_client.return_value = fake_supabase

    response = client.get(
        "/api/connectors/gmail-mcp/callback",
        params={"code": "auth-code", "state": "valid-state"},
    )

    assert response.status_code == 200
    assert "connected" in response.text.lower()
    mock_store.assert_awaited_once_with(fake_supabase, "user-123", "stored-refresh-token")


@patch("mcp_integration.gmail_mcp_router.verify_state", return_value="user-123")
@patch("mcp_integration.gmail_mcp_router.exchange_code_for_tokens", new_callable=AsyncMock)
@patch("mcp_integration.gmail_mcp_router.build_service_role_client")
@patch("mcp_integration.gmail_mcp_router.store_gmail_mcp_tokens", new_callable=AsyncMock)
def test_callback_storage_failure_returns_500(mock_store, mock_build_client, mock_exchange, mock_verify_state):
    mock_exchange.return_value = {"access_token": "at", "refresh_token": "rt"}
    mock_build_client.return_value = MagicMock()
    mock_store.side_effect = RuntimeError("db unavailable")

    response = client.get(
        "/api/connectors/gmail-mcp/callback",
        params={"code": "auth-code", "state": "valid-state"},
    )

    assert response.status_code == 500
