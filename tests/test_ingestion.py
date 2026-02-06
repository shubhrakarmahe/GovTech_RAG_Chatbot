import pytest
import os
import io
import pandas as pd
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from src.ingestion import UnifiedIngestor

# --- FIXTURES ---

@pytest.fixture
def ingestor():
    """
    Initializes the ingestor with mocked components.
    Mocks external services (HuggingFace, Groq, Chroma) to ensure 
    tests are fast, deterministic, and cost-free.
    """
    with patch('src.ingestion.HuggingFaceEmbeddings'), \
         patch('src.ingestion.ChatGroq'), \
         patch('src.ingestion.Chroma'):
        return UnifiedIngestor()

# --- UNIT TESTS: PARSING & LOGIC ---

def test_split_by_headings(ingestor):
    """
    Tests the regex-based heading detection.
    Verifies that the system correctly identifies segments like 'Section 1' 
    and 'APPENDIX A'.
    """
    content = (
        "Section 1: Executive Summary\nThis is the summary.\n"
        "APPENDIX A: Technical Details\nThis is the appendix."
    )
    sections = ingestor._split_by_headings(content)
    
    assert len(sections) == 2
    assert sections[0]["header"] == "Section 1: Executive Summary"
    assert sections[1]["header"] == "APPENDIX A: Technical Details"

def test_process_csv_metadata(ingestor):
    """
    Validates CSV processing, ensuring row-level metadata and 
    column summaries are correctly handled.
    """
    csv_bytes = b"id,policy_name\n101,Standard Health\n102,Basic Dental"
    filename = "policies.csv"
    
    # Mock LLM response for column descriptions and summaries
    ingestor.llm.invoke = MagicMock(return_value=MagicMock(content="Mocked LLM Response"))
    
    docs = ingestor.process_csv(csv_bytes, filename)
    
    assert len(docs) == 2
    assert docs[0].metadata["source"] == filename
    assert docs[0].metadata["row"] == 1
    assert "Standard Health" in docs[0].page_content
    assert docs[0].metadata["column_content"] == "Mocked LLM Response"

def test_get_summary_fallback(ingestor):
    """
    Ensures that if the LLM fails, the system provides a 
    safe fallback summary.
    """
    ingestor.llm.invoke = MagicMock(side_effect=Exception("API Timeout"))
    summary = ingestor._get_summary("Some text", "context")
    
    assert summary == "General content."

# --- INTEGRATION TESTS: INGESTION WORKFLOW ---

def test_ingest_unsupported_extension(ingestor):
    """
    Verifies that unsupported file types return 0 chunks 
    processed.
    """
    with patch('os.path.exists', return_value=True):
        count = ingestor.ingest("data.exe")
        assert count == 0

@patch('src.ingestion.pdfplumber.open')
def test_full_pdf_ingestion_path(mock_pdf_open, ingestor):
    """
    Simulates the full workflow for a PDF file from 
    extraction to vector storage.
    """
    # Mock PDF structure
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "Section 1: Scope\nContent of the scope."
    mock_pdf = MagicMock()
    mock_pdf.pages = [mock_page]
    mock_pdf_open.return_value.__enter__.return_value = mock_pdf

    # Mock Chroma and utility methods
    ingestor._get_summary = MagicMock(return_value="Short Summary")
    
    with patch('os.path.exists', return_value=True), \
         patch('src.ingestion.Chroma') as mock_chroma:
        
        count = ingestor.ingest("scope_doc.pdf")
        
        # Verify the ingestor returned a positive count and called storage
        assert count > 0
        mock_chroma.return_value.add_documents.assert_called_once()