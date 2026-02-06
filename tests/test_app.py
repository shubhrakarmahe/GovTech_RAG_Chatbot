import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from app import app

client = TestClient(app)

# --- SYSTEM ENDPOINTS ---

def test_read_health():
    """Verifies system health check logic."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "components" in data

def test_get_stats_error_handling():
    """Tests statistics endpoint when the vector store is unavailable."""
    with patch("app.ingestor.vectorstore._collection.get", side_effect=Exception("DB Down")):
        response = client.get("/stats")
        assert response.status_code == 500
        assert "Could not retrieve system statistics" in response.json()["detail"]

# --- LIBRARY ENDPOINTS ---

def test_list_documents_empty():
    """Verifies list documents returns an empty list when no files are indexed."""
    mock_get = MagicMock(return_value={"metadatas": []})
    with patch("app.ingestor.vectorstore._collection.get", mock_get):
        response = client.get("/documents")
        assert response.status_code == 200
        assert response.json()["documents"] == []
        assert response.json()["count"] == 0

def test_delete_document_success():
    """Verifies delete request processing."""
    with patch("app.ingestor.vectorstore._collection.delete") as mock_delete:
        response = client.post("/delete", json={"filename": "test.pdf"})
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        mock_delete.assert_called_once()

# --- INGESTION ENDPOINTS ---

def test_upload_file_queuing():
    """Verifies that file uploads are correctly queued as background tasks."""
    # Create a mock file
    file_content = b"fake pdf content"
    files = {"file": ("test.pdf", file_content, "application/pdf")}
    
    with patch("shutil.copyfileobj"), patch("app.run_ingestion"):
        response = client.post("/upload", files=files)
        assert response.status_code == 200
        assert response.json()["status"] == "queued"
        assert response.json()["filename"] == "test.pdf"

# --- CHAT ENDPOINTS ---

def test_chat_workflow():
    """Verifies that the chat endpoint correctly invokes the LangGraph workflow."""
    mock_result = {
        "answer": '{"synthesized_answer": "Hello"}',
        "metadata_manifest": ["doc1.pdf"],
        "confidence_score": 0.95
    }
    
    with patch("app.workflow.invoke", return_value=mock_result):
        payload = {
            "question": "What is the policy?",
            "thread_id": "test-session"
        }
        response = client.post("/chat", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["confidence_score"] == 0.95
        assert "doc1.pdf" in data["sources"]
        assert "latency_ms" in data