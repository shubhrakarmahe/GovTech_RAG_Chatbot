import os
import logging
import json
import asyncio
import re
from typing import List, Dict, Any, Tuple
from diskcache import Cache

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from flashrank import Ranker, RerankRequest

# --- LOGGING ---
logger = logging.getLogger("SmartRetriever")
logging.basicConfig(level=logging.INFO)

PERSIST_DIR = os.getenv("PERSIST_DIRECTORY", "./chroma_db")
CACHE_DIR = "./query_cache"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

class SmartRetriever:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        self.vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=self.embeddings)
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        self.ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="./opt")
        self.cache = Cache(CACHE_DIR)

    def _format_chroma_filter(self, filter_dict: Dict[str, Any]) -> Dict[str, Any] or None:
        """
        Formats a flat dictionary into a Chroma-compatible $and filter.
        Ensures no 'None' values are passed.
        """
        if not filter_dict:
            return None
            
        # 1. Clean out any None values or empty strings
        clean_filters = {k: v for k, v in filter_dict.items() if v is not None and v != ""}
        
        if not clean_filters:
            return None

        # 2. If there's only one filter, Chroma accepts it as a flat dict
        if len(clean_filters) == 1:
            return clean_filters

        # 3. If there are multiple, wrap them in an '$and' operator
        # This fixes the "Expected where to have exactly one operator" error
        return {"$and": [{k: v} for k, v in clean_filters.items()]}

    def _get_schema_peek(self, query: str) -> str:
        try:
            peek_docs = self.vectorstore.similarity_search(query, k=3)
            peek_context = []
            for d in peek_docs:
                src = d.metadata.get("source", "unknown")
                header = d.metadata.get("header", "N/A")
                chunk_sum = d.metadata.get("chunk_summary", "No chunk summary")
                file_sum = d.metadata.get("file_summary", "N/A") 
                
                peek_context.append(
                    f"File: {src} | Section: {header} | Context: {chunk_sum or file_sum}"
                )
            return "\n".join(peek_context)
        except Exception as e:
            logger.warning(f"Peek failed: {e}")
            return "No structural metadata available."

    async def _analyze_query(self, query: str) -> Dict[str, Any]:
        schema_info = self._get_schema_peek(query)
        
        prompt = f"""
        Analyze the USER_QUERY and identify specific keywords or metadata values to filter by.
        
        STRUCTURAL CONTEXT (Sections & Summaries found):
        {schema_info}

        INSTRUCTIONS:
        1. Extract 'clean_query' (the core search terms).
        2. Identify 'filter' criteria based on:
           - 'source' (filename)
           - 'header' (The specific section title like 'Introduction')
           - 'type' (pdf_text, csv_row, markdown)
        3. If the user asks for a specific section, use the 'header' filter.
        
        USER_QUERY: "{query}"

        Output JSON only:
        {{
            "clean_query": "string",
            "filter": {{"metadata_key": "extracted_value"}} or null
        }}
        """
        try:
            res = await self.llm.ainvoke(prompt)
            match = re.search(r'\{.*\}', res.content, re.DOTALL)
            if match:
                return json.loads(match.group())
            return {"clean_query": query, "filter": None}
        except Exception as e:
            logger.error(f"Analysis Error: {e}")
            return {"clean_query": query, "filter": None}

    def invoke(self, query: str, top_k: int = 5) -> List[Document]:
        # 1. Cache Check
        if query in self.cache:
            logger.info(f"Cache hit: {query}")
            data = self.cache[query]
            return [Document(page_content=d['text'], metadata=d['meta']) for d in data]

        # 2. Extract Intent
        try:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
            
            analysis = loop.run_until_complete(self._analyze_query(query))
        except Exception as e:
            logger.error(f"Query Analysis failed: {e}")
            analysis = {"clean_query": query, "filter": None}

        clean_q = analysis.get("clean_query", query)
        raw_filters = analysis.get("filter")
        
        # FIX: Format the filters for ChromaDB compatibility
        formatted_filters = self._format_chroma_filter(raw_filters)

        # 3. Vector Search
        logger.info(f"Retrieving Intent: '{clean_q}' | Filters: {formatted_filters}")
        candidates = self.vectorstore.similarity_search(clean_q, k=25, filter=formatted_filters)
        
        if not candidates:
            return []

        # 4. Reranking
        passages = []
        for i, d in enumerate(candidates):
            header_prefix = f"Section: {d.metadata.get('header', 'General')}\n"
            passages.append({
                "id": i, 
                "text": f"{header_prefix}{d.page_content}", 
                "meta": d.metadata
            })
        
        rerank_request = RerankRequest(query=query, passages=passages)
        results = self.ranker.rerank(rerank_request)

        # 5. Build Final Docs
        final_docs = []
        for r in results[:top_k]:
            final_docs.append(Document(page_content=r["text"], metadata=r["meta"]))

        # 6. Cache and Return
        self.cache.set(query, [{"text": d.page_content, "meta": d.metadata} for d in final_docs], expire=3600)
        return final_docs

# Global singleton
_retriever = None
def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = SmartRetriever()
    return _retriever