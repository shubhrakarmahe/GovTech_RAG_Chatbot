import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from src.retriever import SmartRetriever
from langchain_core.documents import Document

# --- FIXTURES ---

@pytest.fixture
def retriever():
    """Initializes the retriever with mocked external dependencies."""
    with patch('src.retriever.HuggingFaceEmbeddings'), \
         patch('src.retriever.ChatGroq'), \
         patch('src.retriever.Chroma'), \
         patch('src.retriever.Ranker'), \
         patch('src.retriever.Cache'):
        return SmartRetriever()

# --- UNIT TESTS: FILTER LOGIC ---

def test_format_chroma_filter_none(retriever):
    """Ensures empty or None filters return None."""
    assert retriever._format_chroma_filter({}) is None
    assert retriever._format_chroma_filter(None) is None
    assert retriever._format_chroma_filter({"key": None}) is None

def test_format_chroma_filter_single(retriever):
    """Ensures a single filter remains a flat dictionary."""
    filter_dict = {"source": "test.pdf"}
    formatted = retriever._format_chroma_filter(filter_dict)
    assert formatted == {"source": "test.pdf"}

def test_format_chroma_filter_multiple(retriever):
    """Ensures multiple filters are wrapped in a $and operator for ChromaDB."""
    filter_dict = {"source": "test.pdf", "type": "pdf_text"}
    formatted = retriever._format_chroma_filter(filter_dict)
    assert "$and" in formatted
    assert len(formatted["$and"]) == 2

# --- UNIT TESTS: QUERY ANALYSIS ---

@pytest.mark.asyncio
async def test_analyze_query_json_extraction(retriever):
    """Verifies that the LLM output is correctly parsed into clean_query and filters."""
    mock_res = MagicMock()
    mock_res.content = '{"clean_query": "budget report", "filter": {"header": "Financials"}}'
    retriever.llm.ainvoke = AsyncMock(return_value=mock_res)
    
    # Mock schema peek to avoid vectorstore calls
    retriever._get_schema_peek = MagicMock(return_value="Mock Schema")
    
    analysis = await retriever._analyze_query("Show me the Financials section of the budget report")
    
    assert analysis["clean_query"] == "budget report"
    assert analysis["filter"]["header"] == "Financials"

# --- INTEGRATION TESTS: INVOKE WORKFLOW ---

def test_invoke_cache_hit(retriever):
    """Ensures that the retriever returns cached results immediately if available."""
    retriever.cache.__contains__ = MagicMock(return_value=True)
    retriever.cache.__getitem__ = MagicMock(return_value=[
        {"text": "cached content", "meta": {"source": "cache.pdf"}}
    ])
    
    results = retriever.invoke("any query")
    
    assert len(results) == 1
    assert results[0].page_content == "cached content"
    assert results[0].metadata["source"] == "cache.pdf"

def test_invoke_full_retrieval_flow(retriever):
    """
    Tests the full flow: Analysis -> Similarity Search -> Reranking.
    """
    # 1. Mock Analysis
    retriever._analyze_query = AsyncMock(return_value={"clean_query": "test", "filter": None})
    
    # 2. Mock Vector Search Candidates
    mock_docs = [Document(page_content="raw text", metadata={"header": "Section 1"})]
    retriever.vectorstore.similarity_search = MagicMock(return_value=mock_docs)
    
    # 3. Mock FlashRank Reranker
    # FlashRank returns a list of dicts with 'text' and 'meta' keys
    retriever.ranker.rerank = MagicMock(return_value=[
        {"text": "Section: Section 1\nraw text", "meta": {"header": "Section 1"}}
    ])
    
    results = retriever.invoke("test query", top_k=1)
    
    assert len(results) == 1
    assert "Section: Section 1" in results[0].page_content
    retriever.ranker.rerank.assert_called_once()