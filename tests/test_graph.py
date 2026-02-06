import pytest
import json
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from src.graph import build_rag_graph, GraphState

# --- FIXTURES ---

@pytest.fixture
def graph():
    """Compiles the LangGraph workflow for testing."""
    with patch('src.graph.ChatGroq'), patch('src.graph.get_retriever'):
        return build_rag_graph()

@pytest.fixture
def mock_state():
    """Provides a base state for testing graph nodes."""
    return {
        "question": "What is the budget?",
        "documents": [],
        "answer": "",
        "metadata_manifest": [],
        "confidence_score": 0.0,
        "latency_ms": 0.0
    }

# --- UNIT TESTS: NODES & LOGIC ---

def test_retrieve_node(mock_state):
    """Verifies that the retrieve node populates documents and manifest."""
    from src.graph import retrieve_node
    
    # Mock documents returned by retriever
    mock_docs = [Document(page_content="Budget is $1M", metadata={"source": "budget.pdf"})]
    
    with patch('src.graph.get_retriever') as mock_retriever:
        mock_retriever.return_value.invoke.return_value = mock_docs
        
        result = retrieve_node(mock_state)
        
        assert len(result["documents"]) == 1
        assert "budget.pdf" in result["metadata_manifest"]
        assert result["latency_ms"] >= 0

def test_synthesize_node_success(mock_state):
    """Verifies that synthesis correctly parses structured LLM output."""
    from src.graph import synthesize_node
    
    # Set up state with context
    mock_state["documents"] = [Document(page_content="Answer here", metadata={"source": "test.pdf"})]
    
    # Mock LLM response
    mock_res = {
        "detailed_analysis": "Found in test.pdf",
        "synthesized_answer": "The answer is X.",
        "confidence_score": 0.9,
        "sources_cited": ["test.pdf"]
    }
    
    with patch('src.graph.ChatGroq') as mock_llm:
        # Mock the chain invocation
        mock_llm.return_value.__or__.return_value.invoke.return_value = mock_res
        
        result = synthesize_node(mock_state)
        
        # Verify the answer is a JSON string as expected by the UI
        parsed_answer = json.loads(result["answer"])
        assert parsed_answer["confidence_score"] == 0.9
        assert result["confidence_score"] == 0.9

def test_no_data_node(mock_state):
    """Verifies the fallback response when no documents are found."""
    from src.graph import no_data_node
    
    result = no_data_node(mock_state)
    parsed_answer = json.loads(result["answer"])
    
    assert parsed_answer["confidence_score"] == 0.0
    assert "No relevant documents" in parsed_answer["detailed_analysis"]

# --- UNIT TESTS: CONDITIONAL EDGES ---

def test_check_docs_routing():
    """Tests the logic that decides to synthesize or refuse."""
    from src.graph import check_docs
    
    # Case 1: Documents found
    state_with_docs = {"documents": [Document(page_content="test")]}
    assert check_docs(state_with_docs) == "synthesize"
    
    # Case 2: No documents
    state_empty = {"documents": []}
    assert check_docs(state_empty) == "no_data"

# --- INTEGRATION TEST: FULL GRAPH EXECUTION ---

def test_graph_integration(graph, mock_state):
    """Tests a full pass through the compiled graph."""
    # Mock the components within the graph
    with patch('src.graph.get_retriever') as mock_ret:
        mock_ret.return_value.invoke.return_value = [] # Force 'no_data' path
        
        # Run graph
        final_state = graph.invoke(mock_state)
        
        # Verify it went through no_data_node
        assert json.loads(final_state["answer"])["confidence_score"] == 0.0