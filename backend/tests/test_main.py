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

@patch("api.chat.routers.create_client")
def test_chat_endpoint_mock_provider(mock_create_client):
    # Setup mock supabase client response
    mock_supabase = MagicMock()
    mock_table = MagicMock()
    
    # Chain methods: select().eq().order().execute()
    mock_table.select.return_value = mock_table
    mock_table.eq.return_value = mock_table
    mock_table.order.return_value = mock_table
    
    mock_res = MagicMock()
    mock_res.data = [{"role": "user", "content": "hello"}]
    mock_table.execute.return_value = mock_res
    
    mock_supabase.table.return_value = mock_table
    mock_create_client.return_value = mock_supabase

    response = client.post(
        "/api/chat",
        json={"provider": "mock", "sessionId": "12345-abcde"},
        headers={"Authorization": "Bearer fake_token"}
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    
    # Read streamed events
    body = response.text
    assert "data:" in body
    assert "[DONE]" in body

def test_chat_endpoint_unauthorized():
    response = client.post(
        "/api/chat",
        json={"provider": "mock", "sessionId": "12345-abcde"}
    )
    assert response.status_code == 401
    assert "detail" in response.json()

def test_provider_factory_mock():
    provider = ProviderFactory.get_provider("mock")
    assert isinstance(provider, MockProvider)

def test_provider_factory_invalid():
    with pytest.raises(ValueError):
        ProviderFactory.get_provider("invalid_provider")
