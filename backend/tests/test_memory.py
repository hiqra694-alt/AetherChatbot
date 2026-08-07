import sys
import os
import pytest
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient  # pyright: ignore [reportMissingImports]

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app  # pyright: ignore [reportMissingImports]
from api.memory.services import extract_and_store_memory, _parse_extracted_facts  # pyright: ignore [reportMissingImports]

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
def test_get_memory_lists_facts(mock_create_client):
    mock_supabase = _make_mock_supabase()
    mock_table = MagicMock()
    mock_table.select.return_value = mock_table
    mock_table.eq.return_value = mock_table
    mock_table.order.return_value = mock_table
    mock_res = MagicMock()
    mock_res.data = [{"id": "1", "fact": "Works at Zylo", "created_at": "2026-07-23T10:00:00Z"}]
    mock_table.execute.return_value = mock_res
    mock_supabase.table.return_value = mock_table

    mock_create_client.return_value = mock_supabase

    response = client.get("/api/memory", headers={"Authorization": "Bearer fake_token"})
    assert response.status_code == 200
    body = response.json()
    assert len(body["facts"]) == 1
    assert body["facts"][0]["fact"] == "Works at Zylo"


def test_get_memory_unauthorized():
    response = client.get("/api/memory")
    assert response.status_code == 401
    assert "detail" in response.json()


# ==================================================
# DELETE /api/memory/{memory_id}
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
def test_delete_memory_success(mock_create_client):
    mock_create_client.return_value = _make_mock_supabase_for_delete([{"id": "1"}])

    response = client.delete("/api/memory/1", headers={"Authorization": "Bearer fake_token"})
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == "1"
    assert body["deleted"] is True


@patch("api.memory.routers.create_client")
def test_delete_memory_not_found(mock_create_client):
    mock_create_client.return_value = _make_mock_supabase_for_delete([])

    response = client.delete("/api/memory/missing", headers={"Authorization": "Bearer fake_token"})
    assert response.status_code == 404


def test_delete_memory_unauthorized():
    response = client.delete("/api/memory/1")
    assert response.status_code == 401


# ==================================================
# _parse_extracted_facts
# ==================================================

def test_parse_extracted_facts_plain_json():
    assert _parse_extracted_facts('{"facts": ["Works at Zylo"]}') == ["Works at Zylo"]


def test_parse_extracted_facts_empty_list():
    assert _parse_extracted_facts('{"facts": []}') == []


def test_parse_extracted_facts_garbage_text():
    assert _parse_extracted_facts("I don't know what you mean.") == []


def test_parse_extracted_facts_extracts_embedded_json():
    text = 'Sure! Here you go: {"facts": ["Works at Acme"]} Hope that helps.'
    assert _parse_extracted_facts(text) == ["Works at Acme"]


def test_parse_extracted_facts_strips_markdown_fence():
    text = '```json\n{"facts": ["Lives in Lahore"]}\n```'
    assert _parse_extracted_facts(text) == ["Lives in Lahore"]


# ==================================================
# extract_and_store_memory
# ==================================================

def _make_mock_supabase_for_extraction(existing_rows):
    mock_supabase = MagicMock()
    mock_table = MagicMock()
    mock_table.select.return_value = mock_table
    mock_table.eq.return_value = mock_table
    mock_table.order.return_value = mock_table
    mock_table.insert.return_value = mock_table
    mock_res = MagicMock()
    mock_res.data = existing_rows
    mock_table.execute.return_value = mock_res
    mock_supabase.table.return_value = mock_table
    return mock_supabase, mock_table


@pytest.mark.asyncio
async def test_extract_and_store_memory_stores_new_fact(monkeypatch):
    fake_provider = FakeExtractionProvider('{"facts": ["Works at Zylo as a backend engineer"]}')
    monkeypatch.setattr("api.memory.services.ProviderFactory.get_provider", lambda name: fake_provider)

    mock_supabase, mock_table = _make_mock_supabase_for_extraction(existing_rows=[])

    await extract_and_store_memory(mock_supabase, "user-123", "I work at Zylo as a backend engineer")

    mock_table.insert.assert_called_once_with(
        [{"user_id": "user-123", "fact": "Works at Zylo as a backend engineer"}]
    )


@pytest.mark.asyncio
async def test_extract_and_store_memory_dedupes_case_insensitively(monkeypatch):
    fake_provider = FakeExtractionProvider('{"facts": ["works at zylo"]}')
    monkeypatch.setattr("api.memory.services.ProviderFactory.get_provider", lambda name: fake_provider)

    mock_supabase, mock_table = _make_mock_supabase_for_extraction(
        existing_rows=[{"id": "1", "fact": "Works at Zylo", "created_at": "2026-07-23T10:00:00Z"}]
    )

    await extract_and_store_memory(mock_supabase, "user-123", "I still work at zylo")

    mock_table.insert.assert_not_called()


@pytest.mark.asyncio
async def test_extract_and_store_memory_no_facts_skips_db_entirely(monkeypatch):
    fake_provider = FakeExtractionProvider('{"facts": []}')
    monkeypatch.setattr("api.memory.services.ProviderFactory.get_provider", lambda name: fake_provider)

    mock_supabase = MagicMock()

    await extract_and_store_memory(mock_supabase, "user-123", "what's the weather like today?")

    mock_supabase.table.assert_not_called()


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
    The extraction prompt instructs the model to only report durable facts;
    this test exercises the "no durable fact" response shape end-to-end.
    """
    fake_provider = FakeExtractionProvider('{"facts": []}')
    monkeypatch.setattr("api.memory.services.ProviderFactory.get_provider", lambda name: fake_provider)

    mock_supabase = MagicMock()

    await extract_and_store_memory(mock_supabase, "user-123", "I'm feeling tired today.")

    mock_supabase.table.assert_not_called()


@pytest.mark.asyncio
async def test_extract_and_store_memory_no_user_id_is_noop():
    mock_supabase = MagicMock()

    await extract_and_store_memory(mock_supabase, None, "I work at Zylo")

    mock_supabase.table.assert_not_called()
