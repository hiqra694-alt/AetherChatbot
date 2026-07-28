import sys
import os
import pytest
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient  # pyright: ignore [reportMissingImports]

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app  # pyright: ignore [reportMissingImports]
from api.documents.services import get_relevant_context  # pyright: ignore [reportMissingImports]

client = TestClient(app)


def _make_mock_supabase(deleted_rows):
    """
    Builds a MagicMock Supabase client that satisfies get_authenticated_supabase
    (auth.get_user) and the delete().eq().eq().execute() chain used by
    delete_document, returning `deleted_rows` as the deletion result.
    """
    mock_supabase = MagicMock()
    mock_table = MagicMock()

    mock_table.delete.return_value = mock_table
    mock_table.eq.return_value = mock_table

    mock_res = MagicMock()
    mock_res.data = deleted_rows
    mock_table.execute.return_value = mock_res

    mock_supabase.table.return_value = mock_table

    mock_user_res = MagicMock()
    mock_user_res.user.id = "test-user-id"
    mock_supabase.auth.get_user.return_value = mock_user_res

    return mock_supabase


@patch("api.documents.routers.create_client")
def test_delete_document_success(mock_create_client):
    mock_create_client.return_value = _make_mock_supabase(
        deleted_rows=[{"id": 1}, {"id": 2}]
    )

    response = client.delete(
        "/api/documents/handbook.pdf",
        headers={"Authorization": "Bearer fake_token"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["document_name"] == "handbook.pdf"
    assert body["chunks_deleted"] == 2


@patch("api.documents.routers.create_client")
def test_delete_document_not_found(mock_create_client):
    mock_create_client.return_value = _make_mock_supabase(deleted_rows=[])

    response = client.delete(
        "/api/documents/missing.pdf",
        headers={"Authorization": "Bearer fake_token"},
    )
    assert response.status_code == 404
    assert "detail" in response.json()


def test_delete_document_unauthorized():
    response = client.delete("/api/documents/handbook.pdf")
    assert response.status_code == 401
    assert "detail" in response.json()


def _make_mock_supabase_for_rpc(rows):
    """
    Builds a MagicMock Supabase client whose rpc(...).execute() returns
    `rows`. Scoping by document_name is now enforced inside the
    match_document_chunks SQL function itself (via filter_document_name in
    the WHERE clause), so these tests only need to verify get_relevant_context
    passes the right RPC params through — not re-simulate SQL filtering.
    """
    mock_supabase = MagicMock()
    mock_rpc_builder = MagicMock()

    mock_res = MagicMock()
    mock_res.data = rows
    mock_rpc_builder.execute.return_value = mock_res

    mock_supabase.rpc.return_value = mock_rpc_builder
    return mock_supabase, mock_rpc_builder


@patch("api.documents.services._embed_with_retry")
@pytest.mark.asyncio
async def test_get_relevant_context_passes_filter_document_name_to_rpc(mock_embed):
    mock_embed.return_value = [[0.1, 0.2, 0.3]]
    rows = [
        {"document_name": "IQRA HAMEED_CV.pdf", "chunk_text": "cv chunk", "similarity": 0.8},
    ]
    mock_supabase, mock_rpc_builder = _make_mock_supabase_for_rpc(rows)

    result = await get_relevant_context(
        mock_supabase, "summarize the pdf", "user-123", document_name="IQRA HAMEED_CV.pdf"
    )

    mock_supabase.rpc.assert_called_once()
    rpc_name, rpc_params = mock_supabase.rpc.call_args.args
    assert rpc_name == "match_document_chunks"
    assert rpc_params["filter_user_id"] == "user-123"
    assert rpc_params["filter_document_name"] == "IQRA HAMEED_CV.pdf"

    assert len(result) == 1
    assert result[0].document_name == "IQRA HAMEED_CV.pdf"


@patch("api.documents.services._embed_with_retry")
@pytest.mark.asyncio
async def test_get_relevant_context_no_document_name_passes_none(mock_embed):
    mock_embed.return_value = [[0.1, 0.2, 0.3]]
    rows = [
        {"document_name": "attention_paper.pdf", "chunk_text": "old doc chunk", "similarity": 0.9},
        {"document_name": "IQRA HAMEED_CV.pdf", "chunk_text": "cv chunk", "similarity": 0.8},
    ]
    mock_supabase, mock_rpc_builder = _make_mock_supabase_for_rpc(rows)

    result = await get_relevant_context(mock_supabase, "what did we discuss?", "user-123")

    _, rpc_params = mock_supabase.rpc.call_args.args
    assert rpc_params["filter_document_name"] is None
    assert len(result) == 2
