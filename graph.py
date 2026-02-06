import time
import logging
import json
from typing import TypedDict, List, Any
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from src.retriever import get_retriever

logger = logging.getLogger("RAG_Graph")

class GraphState(TypedDict):
    question: str
    documents: List[Any]
    answer: str  # Stores the raw JSON string for the UI
    metadata_manifest: List[str]
    confidence_score: float  # Tracked for UI and logic
    latency_ms: float

def retrieve_node(state: GraphState):
    logger.info("--- NODE: RETRIEVE ---")
    start = time.time()
    # Uses the heading-aware SmartRetriever
    docs = get_retriever().invoke(state["question"], top_k=6)
    manifest = list(set([d.metadata.get("source", "Unknown") for d in docs]))
    return {
        "documents": docs, 
        "metadata_manifest": manifest,
        "latency_ms": (time.time() - start) * 1000
    }

def synthesize_node(state: GraphState):
    logger.info("--- NODE: SYNTHESIZE ---")
    # Temperature 0 is mandatory for deterministic confidence scoring
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    parser = JsonOutputParser()

    # Build context with headers and chunk summaries from ingestion
    context_blocks = []
    for i, doc in enumerate(state["documents"]):
        block = (
            f"--- SOURCE {i+1} ({doc.metadata.get('source')}) ---\n"
            f"SECTION/HEADER: {doc.metadata.get('header', 'General')}\n"
            f"CHUNK SUMMARY: {doc.metadata.get('chunk_summary', 'N/A')}\n"
            f"CONTENT: {doc.page_content}\n"
        )
        context_blocks.append(block)

    context_str = "\n".join(context_blocks)

    # Strict JSON Prompt with Refusal and Confidence Scoring
    prompt = f"""
    SYSTEM: You are a professional RAG assistant. You must provide answers based ONLY on the provided context.
    
    STRICT RULES:
    1. If the context does not contain the answer, set "confidence_score" to 0.0 and refuse to answer.
    2. Do NOT use any external knowledge. 
    3. Use the following rubric for "confidence_score":
       - 0.9-1.0: Answer is explicitly stated in the context.
       - 0.6-0.8: Answer is inferred from multiple sections of the context.
       - 0.1-0.5: Context is tangentially related but insufficient.
       - 0.0: Information is completely missing.

    USER QUESTION: {state['question']}
    
    CONTEXT DOCUMENTS:
    {context_str}

    RESPONSE FORMAT (JSON ONLY):
    {{
        "detailed_analysis": "string (explain which headers were relevant and how you validated the answer)",
        "synthesized_answer": "string (the answer OR a polite refusal if context is missing)",
        "confidence_score": float,
        "sources_cited": ["filename1", "filename2"]
    }}
    """
    
    try:
        chain = llm | parser
        res = chain.invoke(prompt)
    except Exception as e:
        logger.error(f"Synthesis/Parsing Error: {e}")
        res = {
            "detailed_analysis": "Parsing failed or model timed out.",
            "synthesized_answer": "I'm sorry, I encountered an error processing the response.",
            "confidence_score": 0.0,
            "sources_cited": []
        }
    
    return {
        "answer": json.dumps(res),
        "confidence_score": res.get("confidence_score", 0.0)
    }

def no_data_node(state: GraphState):
    error_res = {
        "detailed_analysis": "No relevant documents found in the database for this query.",
        "synthesized_answer": "I am sorry, but I don't have any documents in my context that can answer that question.",
        "confidence_score": 0.0,
        "sources_cited": []
    }
    return {"answer": json.dumps(error_res), "confidence_score": 0.0}

def check_docs(state: GraphState):
    """Conditional edge to route between synthesis and refusal."""
    return "synthesize" if state["documents"] else "no_data"

def build_rag_graph():
    workflow = StateGraph(GraphState)
    
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("synthesize", synthesize_node)
    workflow.add_node("no_data", no_data_node)
    
    workflow.add_edge(START, "retrieve")
    workflow.add_conditional_edges(
        "retrieve", 
        check_docs, 
        {"synthesize": "synthesize", "no_data": "no_data"}
    )
    workflow.add_edge("synthesize", END)
    workflow.add_edge("no_data", END)
    
    return workflow.compile()