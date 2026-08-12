import sys
import os
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_integration.gmail_mcp import (
    GMAIL_MCP_PROVIDER,
    build_google_authorize_url,
    build_service_role_client,
    exchange_code_for_tokens,
    get_gmail_mcp_refresh_token,
    refresh_gmail_mcp_access_token,
    resolve_user_id_from_access_token,
    sign_state,
    store_gmail_mcp_tokens,
    verify_state,
)


class _FakeSettings:
    def __init__(
        self,
        gmail_mcp_client_id="",
        gmail_mcp_client_secret="",
        gmail_mcp_redirect_uri="",
        supabase_url="",
        supabase_anon_key="",
        supabase_service_role_key="",
    ):
        self.gmail_mcp_client_id = gmail_mcp_client_id
        self.gmail_mcp_client_secret = gmail_mcp_client_secret
        self.gmail_mcp_redirect_uri = gmail_mcp_redirect_uri
        self.supabase_url = supabase_url
        self.supabase_anon_key = supabase_anon_key
        self.supabase_service_role_key = supabase_service_role_key


# ==================================================
# sign_state / verify_state -- HMAC-signed OAuth state, standing in for a
# server-side session since the callback route is a plain browser redirect.
# ==================================================

def test_sign_and_verify_state_round_trips(monkeypatch):
    monkeypatch.setattr(
        "mcp_integration.gmail_mcp.get_settings",
        lambda: _FakeSettings(gmail_mcp_client_secret="test-secret"),
    )

    state = sign_state("user-123")
    assert verify_state(state) == "user-123"


def test_verify_state_rejects_tampered_state(monkeypatch):
    monkeypatch.setattr(
        "mcp_integration.gmail_mcp.get_settings",
        lambda: _FakeSettings(gmail_mcp_client_secret="test-secret"),
    )

    state = sign_state("user-123")
    tampered = state[:-1] + ("A" if state[-1] != "A" else "B")

    with pytest.raises(ValueError):
        verify_state(tampered)


def test_verify_state_rejects_expired_state(monkeypatch):
    monkeypatch.setattr(
        "mcp_integration.gmail_mcp.get_settings",
        lambda: _FakeSettings(gmail_mcp_client_secret="test-secret"),
    )
    monkeypatch.setattr("mcp_integration.gmail_mcp.STATE_TTL_SECONDS", -1)

    state = sign_state("user-123")

    with pytest.raises(ValueError):
        verify_state(state)


def test_verify_state_rejects_malformed_state(monkeypatch):
    monkeypatch.setattr(
        "mcp_integration.gmail_mcp.get_settings",
        lambda: _FakeSettings(gmail_mcp_client_secret="test-secret"),
    )

    with pytest.raises(ValueError):
        verify_state("not-a-real-state-value")


def test_verify_state_rejects_state_signed_with_a_different_secret(monkeypatch):
    monkeypatch.setattr(
        "mcp_integration.gmail_mcp.get_settings",
        lambda: _FakeSettings(gmail_mcp_client_secret="secret-a"),
    )
    state = sign_state("user-123")

    monkeypatch.setattr(
        "mcp_integration.gmail_mcp.get_settings",
        lambda: _FakeSettings(gmail_mcp_client_secret="secret-b"),
    )
    with pytest.raises(ValueError):
        verify_state(state)


# ==================================================
# build_google_authorize_url
# ==================================================

def test_build_google_authorize_url_includes_offline_access_and_consent_prompt(monkeypatch):
    monkeypatch.setattr(
        "mcp_integration.gmail_mcp.get_settings",
        lambda: _FakeSettings(
            gmail_mcp_client_id="client-id",
            gmail_mcp_redirect_uri="https://app.example.com/api/connectors/gmail-mcp/callback",
        ),
    )

    url = build_google_authorize_url("signed-state")

    assert url.startswith("https://accounts.google.com/o/oauth2/v2/auth?")
    assert "client_id=client-id" in url
    assert "access_type=offline" in url
    assert "prompt=consent" in url
    assert "state=signed-state" in url
    assert "redirect_uri=https%3A%2F%2Fapp.example.com" in url


# ==================================================
# resolve_user_id_from_access_token -- read-only Supabase Auth verification
# ==================================================

@pytest.mark.asyncio
async def test_resolve_user_id_from_access_token_success(monkeypatch):
    monkeypatch.setattr(
        "mcp_integration.gmail_mcp.get_settings",
        lambda: _FakeSettings(supabase_url="https://proj.supabase.co", supabase_anon_key="anon-key"),
    )

    fake_user = MagicMock()
    fake_user.id = "user-123"
    fake_client = MagicMock()
    fake_client.auth.get_user.return_value = MagicMock(user=fake_user)
    monkeypatch.setattr("mcp_integration.gmail_mcp.create_client", lambda url, key: fake_client)

    user_id = await resolve_user_id_from_access_token("live-token")

    assert user_id == "user-123"
    fake_client.auth.get_user.assert_called_once_with("live-token")


@pytest.mark.asyncio
async def test_resolve_user_id_from_access_token_invalid_token_raises(monkeypatch):
    monkeypatch.setattr(
        "mcp_integration.gmail_mcp.get_settings",
        lambda: _FakeSettings(supabase_url="https://proj.supabase.co", supabase_anon_key="anon-key"),
    )

    fake_client = MagicMock()
    fake_client.auth.get_user.side_effect = Exception("expired")
    monkeypatch.setattr("mcp_integration.gmail_mcp.create_client", lambda url, key: fake_client)

    with pytest.raises(ValueError):
        await resolve_user_id_from_access_token("bad-token")


@pytest.mark.asyncio
async def test_resolve_user_id_from_access_token_no_user_raises(monkeypatch):
    monkeypatch.setattr(
        "mcp_integration.gmail_mcp.get_settings",
        lambda: _FakeSettings(supabase_url="https://proj.supabase.co", supabase_anon_key="anon-key"),
    )

    fake_client = MagicMock()
    fake_client.auth.get_user.return_value = MagicMock(user=None)
    monkeypatch.setattr("mcp_integration.gmail_mcp.create_client", lambda url, key: fake_client)

    with pytest.raises(ValueError):
        await resolve_user_id_from_access_token("bad-token")


# ==================================================
# build_service_role_client
# ==================================================

def test_build_service_role_client_returns_none_when_unconfigured(monkeypatch):
    monkeypatch.setattr("mcp_integration.gmail_mcp.get_settings", lambda: _FakeSettings())
    assert build_service_role_client() is None


def test_build_service_role_client_builds_client_when_configured(monkeypatch):
    monkeypatch.setattr(
        "mcp_integration.gmail_mcp.get_settings",
        lambda: _FakeSettings(supabase_url="https://proj.supabase.co", supabase_service_role_key="service-key"),
    )
    sentinel = object()
    captured = {}

    def fake_create_client(url, key):
        captured["args"] = (url, key)
        return sentinel

    monkeypatch.setattr("mcp_integration.gmail_mcp.create_client", fake_create_client)

    assert build_service_role_client() is sentinel
    assert captured["args"] == ("https://proj.supabase.co", "service-key")


# ==================================================
# store_gmail_mcp_tokens / get_gmail_mcp_refresh_token -- isolated
# provider row, never the plain 'google' one.
# ==================================================

def _mock_oauth_supabase(rows=None):
    table = MagicMock()
    for method in ("select", "eq", "limit", "upsert"):
        getattr(table, method).return_value = table
    table.execute.return_value = MagicMock(data=rows if rows is not None else [])
    supabase = MagicMock()
    supabase.table.return_value = table
    return supabase, table


@pytest.mark.asyncio
async def test_store_gmail_mcp_tokens_upserts_under_distinct_provider():
    supabase, table = _mock_oauth_supabase()

    await store_gmail_mcp_tokens(supabase, "user-123", "refresh-token-value")

    table.upsert.assert_called_once_with(
        {"user_id": "user-123", "provider": GMAIL_MCP_PROVIDER, "refresh_token": "refresh-token-value"},
        on_conflict="user_id,provider",
    )
    assert GMAIL_MCP_PROVIDER == "google_gmail_mcp"
    assert GMAIL_MCP_PROVIDER != "google"


@pytest.mark.asyncio
async def test_get_gmail_mcp_refresh_token_queries_distinct_provider():
    supabase, table = _mock_oauth_supabase(rows=[{"refresh_token": "stored-token"}])

    result = await get_gmail_mcp_refresh_token(supabase, "user-123")

    assert result == "stored-token"
    table.eq.assert_any_call("user_id", "user-123")
    table.eq.assert_any_call("provider", GMAIL_MCP_PROVIDER)


@pytest.mark.asyncio
async def test_get_gmail_mcp_refresh_token_no_row_returns_none():
    supabase, _ = _mock_oauth_supabase(rows=[])
    result = await get_gmail_mcp_refresh_token(supabase, "user-123")
    assert result is None


# ==================================================
# exchange_code_for_tokens / refresh_gmail_mcp_access_token
# ==================================================

class _FakeHttpResponse:
    def __init__(self, json_data, raise_error=False):
        self._json_data = json_data
        self._raise_error = raise_error

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self._raise_error:
            raise Exception("400 Bad Request")


def _fake_async_client(response_or_factory, captured=None):
    class _FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, data=None, timeout=None):
            if captured is not None:
                captured["url"] = url
                captured["data"] = data
            return response_or_factory() if callable(response_or_factory) else response_or_factory

    return _FakeAsyncClient


@pytest.mark.asyncio
async def test_exchange_code_for_tokens_posts_expected_params(monkeypatch):
    monkeypatch.setattr(
        "mcp_integration.gmail_mcp.get_settings",
        lambda: _FakeSettings(
            gmail_mcp_client_id="client-id",
            gmail_mcp_client_secret="client-secret",
            gmail_mcp_redirect_uri="https://app.example.com/callback",
        ),
    )
    captured = {}
    monkeypatch.setattr(
        "mcp_integration.gmail_mcp.httpx.AsyncClient",
        _fake_async_client(_FakeHttpResponse({"access_token": "at", "refresh_token": "rt"}), captured),
    )

    result = await exchange_code_for_tokens("auth-code")

    assert result == {"access_token": "at", "refresh_token": "rt"}
    assert captured["url"] == "https://oauth2.googleapis.com/token"
    assert captured["data"] == {
        "client_id": "client-id",
        "client_secret": "client-secret",
        "redirect_uri": "https://app.example.com/callback",
        "code": "auth-code",
        "grant_type": "authorization_code",
    }


@pytest.mark.asyncio
async def test_exchange_code_for_tokens_raises_on_http_error(monkeypatch):
    monkeypatch.setattr(
        "mcp_integration.gmail_mcp.get_settings",
        lambda: _FakeSettings(gmail_mcp_client_id="id", gmail_mcp_client_secret="secret", gmail_mcp_redirect_uri="uri"),
    )
    monkeypatch.setattr(
        "mcp_integration.gmail_mcp.httpx.AsyncClient",
        _fake_async_client(_FakeHttpResponse({}, raise_error=True)),
    )

    with pytest.raises(Exception):
        await exchange_code_for_tokens("bad-code")


@pytest.mark.asyncio
async def test_refresh_gmail_mcp_access_token_no_credentials_skips_lookup(monkeypatch):
    monkeypatch.setattr("mcp_integration.gmail_mcp.get_settings", lambda: _FakeSettings())
    supabase = MagicMock()

    result = await refresh_gmail_mcp_access_token(supabase, "user-123")

    assert result is None
    supabase.table.assert_not_called()


@pytest.mark.asyncio
async def test_refresh_gmail_mcp_access_token_no_stored_token_returns_none(monkeypatch):
    monkeypatch.setattr(
        "mcp_integration.gmail_mcp.get_settings",
        lambda: _FakeSettings(gmail_mcp_client_id="id", gmail_mcp_client_secret="secret"),
    )
    supabase, _ = _mock_oauth_supabase(rows=[])

    result = await refresh_gmail_mcp_access_token(supabase, "user-123")

    assert result is None


@pytest.mark.asyncio
async def test_refresh_gmail_mcp_access_token_success(monkeypatch):
    monkeypatch.setattr(
        "mcp_integration.gmail_mcp.get_settings",
        lambda: _FakeSettings(gmail_mcp_client_id="id", gmail_mcp_client_secret="secret"),
    )
    supabase, _ = _mock_oauth_supabase(rows=[{"refresh_token": "stored-refresh-token"}])
    monkeypatch.setattr(
        "mcp_integration.gmail_mcp.httpx.AsyncClient",
        _fake_async_client(_FakeHttpResponse({"access_token": "fresh-access-token"})),
    )

    result = await refresh_gmail_mcp_access_token(supabase, "user-123")

    assert result == "fresh-access-token"


@pytest.mark.asyncio
async def test_refresh_gmail_mcp_access_token_google_error_returns_none(monkeypatch):
    monkeypatch.setattr(
        "mcp_integration.gmail_mcp.get_settings",
        lambda: _FakeSettings(gmail_mcp_client_id="id", gmail_mcp_client_secret="secret"),
    )
    supabase, _ = _mock_oauth_supabase(rows=[{"refresh_token": "revoked-token"}])
    monkeypatch.setattr(
        "mcp_integration.gmail_mcp.httpx.AsyncClient",
        _fake_async_client(_FakeHttpResponse({}, raise_error=True)),
    )

    result = await refresh_gmail_mcp_access_token(supabase, "user-123")

    assert result is None
