import sys
import os
import pytest
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient  # pyright: ignore [reportMissingImports]

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app  # pyright: ignore [reportMissingImports]
from api.memory.services import extract_and_store_memory, _clean_narrative_response  # pyright: ignore [reportMissingImports]

client = TestClient(app)


class FakeExtractionProvider:
    """Yields `response_text` word-by-word, mirroring a real streaming provider."""
    def __init__(self, response_text):
        self.response_text = response_text

    async def stream_response(self, messages, tools=None):
        for word in self.response_text.split(" "):
            yield word + " "


def _make_mock_supabase(auth_user_id="test-user-id"):
    mock_supabase = MagicMock()
    mock_user_res = MagicMock()
    mock_user_res.user.id = auth_user_id
    mock_supabase.auth.get_user.return_value = mock_user_res
    return mock_supabase


# ==================================================
# GET /api/memory
# ==================================================

@patch("api.memory.routers.create_client")
def test_get_memory_returns_narrative(mock_create_client):
    mock_supabase = _make_mock_supabase()
    mock_table = MagicMock()
    mock_table.select.return_value = mock_table
    mock_table.eq.return_value = mock_table
    mock_table.limit.return_value = mock_table
    mock_res = MagicMock()
    mock_res.data = [{"narrative": "Works at Zylo as a backend engineer.", "updated_at": "2026-07-23T10:00:00Z"}]
    mock_table.execute.return_value = mock_res
    mock_supabase.table.return_value = mock_table

    mock_create_client.return_value = mock_supabase

    response = client.get("/api/memory", headers={"Authorization": "Bearer fake_token"})
    assert response.status_code == 200
    body = response.json()
    assert body["narrative"] == "Works at Zylo as a backend engineer."
    assert body["updated_at"] is not None


@patch("api.memory.routers.create_client")
def test_get_memory_returns_empty_when_no_profile_yet(mock_create_client):
    mock_supabase = _make_mock_supabase()
    mock_table = MagicMock()
    mock_table.select.return_value = mock_table
    mock_table.eq.return_value = mock_table
    mock_table.limit.return_value = mock_table
    mock_res = MagicMock()
    mock_res.data = []
    mock_table.execute.return_value = mock_res
    mock_supabase.table.return_value = mock_table

    mock_create_client.return_value = mock_supabase

    response = client.get("/api/memory", headers={"Authorization": "Bearer fake_token"})
    assert response.status_code == 200
    body = response.json()
    assert body["narrative"] == ""
    assert body["updated_at"] is None


def test_get_memory_unauthorized():
    response = client.get("/api/memory")
    assert response.status_code == 401
    assert "detail" in response.json()


# ==================================================
# DELETE /api/memory
# ==================================================

def _make_mock_supabase_for_delete(deleted_rows):
    mock_supabase = _make_mock_supabase()
    mock_table = MagicMock()
    mock_table.delete.return_value = mock_table
    mock_table.eq.return_value = mock_table
    mock_res = MagicMock()
    mock_res.data = deleted_rows
    mock_table.execute.return_value = mock_res
    mock_supabase.table.return_value = mock_table
    return mock_supabase


@patch("api.memory.routers.create_client")
def test_delete_memory_clears_profile(mock_create_client):
    mock_create_client.return_value = _make_mock_supabase_for_delete([{"user_id": "test-user-id"}])

    response = client.delete("/api/memory", headers={"Authorization": "Bearer fake_token"})
    assert response.status_code == 200
    body = response.json()
    assert body["deleted"] is True


@patch("api.memory.routers.create_client")
def test_delete_memory_when_nothing_stored(mock_create_client):
    mock_create_client.return_value = _make_mock_supabase_for_delete([])

    response = client.delete("/api/memory", headers={"Authorization": "Bearer fake_token"})
    assert response.status_code == 200
    body = response.json()
    assert body["deleted"] is False


def test_delete_memory_unauthorized():
    response = client.delete("/api/memory")
    assert response.status_code == 401


# ==================================================
# _clean_narrative_response
# ==================================================

def test_clean_narrative_response_plain_prose():
    assert _clean_narrative_response("Works at Zylo as a backend engineer.") == "Works at Zylo as a backend engineer."


def test_clean_narrative_response_empty():
    assert _clean_narrative_response("") == ""
    assert _clean_narrative_response("   ") == ""


def test_clean_narrative_response_strips_markdown_fence():
    text = "```\nWorks at Zylo as a backend engineer.\n```"
    assert _clean_narrative_response(text) == "Works at Zylo as a backend engineer."


def test_clean_narrative_response_strips_surrounding_quotes():
    assert _clean_narrative_response('"Lives in Lahore."') == "Lives in Lahore."


# ==================================================
# extract_and_store_memory
# ==================================================

def _make_mock_supabase_for_extraction(existing_narrative_rows):
    mock_supabase = MagicMock()
    mock_table = MagicMock()
    mock_table.select.return_value = mock_table
    mock_table.eq.return_value = mock_table
    mock_table.limit.return_value = mock_table
    mock_table.upsert.return_value = mock_table
    mock_res = MagicMock()
    mock_res.data = existing_narrative_rows
    mock_table.execute.return_value = mock_res
    mock_supabase.table.return_value = mock_table
    return mock_supabase, mock_table


@pytest.mark.asyncio
async def test_extract_and_store_memory_writes_first_profile(monkeypatch):
    fake_provider = FakeExtractionProvider("Works at Zylo as a backend engineer.")
    monkeypatch.setattr("api.memory.services.ProviderFactory.get_provider", lambda name: fake_provider)

    mock_supabase, mock_table = _make_mock_supabase_for_extraction(existing_narrative_rows=[])

    await extract_and_store_memory(mock_supabase, "user-123", "I work at Zylo as a backend engineer")

    mock_table.upsert.assert_called_once_with(
        {"user_id": "user-123", "narrative": "Works at Zylo as a backend engineer."},
        on_conflict="user_id",
    )


@pytest.mark.asyncio
async def test_extract_and_store_memory_rewrites_existing_profile_via_upsert(monkeypatch):
    """
    The Upsert/Rewrite pattern: an existing profile is passed to the
    extraction prompt and the (expanded) rewrite is upserted back onto the
    same user_id row -- never inserted as an additional row.
    """
    fake_provider = FakeExtractionProvider("Works at Zylo as a backend engineer. Is currently building a FastAPI service.")
    monkeypatch.setattr("api.memory.services.ProviderFactory.get_provider", lambda name: fake_provider)

    mock_supabase, mock_table = _make_mock_supabase_for_extraction(
        existing_narrative_rows=[{"narrative": "Works at Zylo as a backend engineer.", "updated_at": "2026-07-23T10:00:00Z"}]
    )

    await extract_and_store_memory(mock_supabase, "user-123", "I'm building a FastAPI service right now")

    mock_table.insert.assert_not_called()
    mock_table.upsert.assert_called_once_with(
        {"user_id": "user-123", "narrative": "Works at Zylo as a backend engineer. Is currently building a FastAPI service."},
        on_conflict="user_id",
    )


@pytest.mark.asyncio
async def test_extract_and_store_memory_skips_write_when_profile_unchanged(monkeypatch):
    """
    When the rewrite model echoes the current profile back unchanged (no new
    durable fact this turn), no DB write should happen at all.
    """
    fake_provider = FakeExtractionProvider("Works at Zylo as a backend engineer.")
    monkeypatch.setattr("api.memory.services.ProviderFactory.get_provider", lambda name: fake_provider)

    mock_supabase, mock_table = _make_mock_supabase_for_extraction(
        existing_narrative_rows=[{"narrative": "Works at Zylo as a backend engineer.", "updated_at": "2026-07-23T10:00:00Z"}]
    )

    await extract_and_store_memory(mock_supabase, "user-123", "what's the weather like today?")

    mock_table.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_extract_and_store_memory_no_facts_skips_db_write(monkeypatch):
    fake_provider = FakeExtractionProvider("")
    monkeypatch.setattr("api.memory.services.ProviderFactory.get_provider", lambda name: fake_provider)

    mock_supabase, mock_table = _make_mock_supabase_for_extraction(existing_narrative_rows=[])

    await extract_and_store_memory(mock_supabase, "user-123", "what's the weather like today?")

    mock_table.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_extract_and_store_memory_never_raises_on_provider_failure(monkeypatch):
    """
    Best-effort background task -- a missing API key, rate limit, or any
    other provider error must be swallowed, never propagated, since there's
    no request left to fail by the time this runs.
    """
    def raise_missing_key(name):
        raise ValueError("GROQ_API_KEY is not configured.")

    monkeypatch.setattr("api.memory.services.ProviderFactory.get_provider", raise_missing_key)

    mock_supabase = MagicMock()

    await extract_and_store_memory(mock_supabase, "user-123", "I work at Zylo")


@pytest.mark.asyncio
async def test_extract_and_store_memory_ignores_transient_statements(monkeypatch):
    """
    The extraction prompt instructs the model to echo the current (empty)
    profile back unchanged for a transient statement; this test exercises
    that "no durable fact" response shape end-to-end.
    """
    fake_provider = FakeExtractionProvider("")
    monkeypatch.setattr("api.memory.services.ProviderFactory.get_provider", lambda name: fake_provider)

    mock_supabase, mock_table = _make_mock_supabase_for_extraction(existing_narrative_rows=[])

    await extract_and_store_memory(mock_supabase, "user-123", "I'm feeling tired today.")

    mock_table.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_extract_and_store_memory_no_user_id_is_noop():
    mock_supabase = MagicMock()

    await extract_and_store_memory(mock_supabase, None, "I work at Zylo")

    mock_supabase.table.assert_not_called()
