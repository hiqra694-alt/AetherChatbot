import pytest # pyright: ignore [reportMissingImports]
from fastapi.testclient import TestClient # pyright: ignore [reportMissingImports]
import sys
import os
from unittest.mock import patch, MagicMock

# Adjust import path to find main
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app # pyright: ignore [reportMissingImports]
from providers.factory import ProviderFactory # pyright: ignore [reportMissingImports]
from providers.mock import MockProvider # pyright: ignore [reportMissingImports]

client = TestClient(app)

def _make_mock_supabase():
    """
    Builds a MagicMock Supabase client that satisfies get_authenticated_supabase
    (auth.get_user) and ChatService.get_history/persist_message (table chain).
    """
    mock_supabase = MagicMock()
    mock_table = MagicMock()

    # Chain methods: select().eq().order().execute() / insert().execute()
    mock_table.select.return_value = mock_table
    mock_table.eq.return_value = mock_table
    mock_table.order.return_value = mock_table
    mock_table.insert.return_value = mock_table

    mock_res = MagicMock()
    mock_res.data = [{"role": "user", "content": "hello"}]
    mock_table.execute.return_value = mock_res

    mock_supabase.table.return_value = mock_table

    mock_user_res = MagicMock()
    mock_user_res.user.id = "test-user-id"
    mock_supabase.auth.get_user.return_value = mock_user_res

    return mock_supabase

@patch("api.chat.services.get_relevant_context")
@patch("api.chat.routers.create_client")
def test_chat_endpoint_mock_provider(mock_create_client, mock_get_relevant_context):
    # No file attached in this request (agentic Branch B), and MockProvider
    # never emits a tool call, so RAG retrieval must never fire — it's only
    # triggered if the model itself calls the search_knowledge_base tool.
    mock_create_client.return_value = _make_mock_supabase()
    mock_get_relevant_context.return_value = []

    response = client.post(
        "/api/chat",
        data={"provider": "mock", "sessionId": "12345-abcde", "message": "Hello there"},
        headers={"Authorization": "Bearer fake_token"}
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    # Read streamed events
    body = response.text
    assert "data:" in body
    assert "[DONE]" in body
    mock_get_relevant_context.assert_not_called()

def test_chat_endpoint_unauthorized():
    response = client.post(
        "/api/chat",
        data={"provider": "mock", "sessionId": "12345-abcde", "message": "Hello there"}
    )
    assert response.status_code == 401
    assert "detail" in response.json()

@patch("api.chat.routers.create_client")
def test_chat_endpoint_missing_file_and_message(mock_create_client):
    mock_create_client.return_value = _make_mock_supabase()

    response = client.post(
        "/api/chat",
        data={"provider": "mock", "sessionId": "12345-abcde"},
        headers={"Authorization": "Bearer fake_token"}
    )
    assert response.status_code == 400
    assert "detail" in response.json()

@patch("api.chat.routers.process_and_store_pdf")
@patch("api.chat.routers.create_client")
def test_chat_endpoint_file_only_upload_acknowledges_without_llm_call(mock_create_client, mock_process_and_store_pdf):
    mock_create_client.return_value = _make_mock_supabase()

    async def fake_process_and_store_pdf(supabase, file_bytes, filename, user_id):
        return 3

    mock_process_and_store_pdf.side_effect = fake_process_and_store_pdf

    response = client.post(
        "/api/chat",
        data={"provider": "mock", "sessionId": "12345-abcde"},
        files={"file": ("handbook.pdf", b"%PDF-1.4 fake pdf bytes", "application/pdf")},
        headers={"Authorization": "Bearer fake_token"}
    )
    assert response.status_code == 200
    body = response.text
    assert "successfully processed your document" in body
    assert "[DONE]" in body
    mock_process_and_store_pdf.assert_called_once()

def test_provider_factory_mock():
    provider = ProviderFactory.get_provider("mock")
    assert isinstance(provider, MockProvider)

def test_provider_factory_invalid():
    with pytest.raises(ValueError):
        ProviderFactory.get_provider("invalid_provider")
