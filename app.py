import os
import uuid
import shutil
import logging
import time
from typing import List
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel
from src.ingestion import UnifiedIngestor
from src.graph import build_rag_graph

# --- CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RAG_API")

app = FastAPI(title="Granicus AI Backend")

# Initialize core components
ingestor = UnifiedIngestor()
workflow = build_rag_graph()

UPLOAD_DIR = "./temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- MODELS ---
class ChatRequest(BaseModel):
    question: str
    thread_id: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence_score: float
    latency_ms: float

class DeleteRequest(BaseModel):
    filename: str

# --- MIDDLEWARE ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = (time.time() - start) * 1000
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {duration:.2f}ms")
    return response

# --- HELPERS ---
def run_ingestion(path: str, original_filename: str):
    try:
        logger.info(f"🚀 Processing background ingestion for: {original_filename}")
        final_path = os.path.join(UPLOAD_DIR, original_filename)
        os.rename(path, final_path)
        
        count = ingestor.ingest(final_path)
        
        if count > 0:
            logger.info(f"✅ Successfully indexed {count} chunks from {original_filename}")
        else:
            logger.warning(f"⚠️ Ingestion complete for {original_filename} but 0 chunks were added.")
            
        if os.path.exists(final_path):
            os.remove(final_path)
    except Exception as e:
        logger.error(f"❌ Critical Ingestion Error for {original_filename}: {e}", exc_info=True)

# --- ENDPOINTS ---

@app.get("/documents")
async def list_documents():
    """Returns a list of unique filenames currently in the vector store."""
    try:
        # Access Chroma collection through the LangChain wrapper
        collection = ingestor.vectorstore._collection
        results = collection.get(include=["metadatas"])
        
        sources = set()
        for meta in results.get("metadatas", []):
            if meta and "source" in meta:
                # Store only the filename, not the full path
                sources.add(os.path.basename(meta["source"]))
        
        return {"documents": sorted(list(sources)), "count": len(sources)}
    except Exception as e:
        logger.error(f"List error: {e}")
        raise HTTPException(status_code=500, detail="Could not retrieve documents.")

@app.post("/delete")
async def delete_document(req: DeleteRequest):
    """Deletes all chunks belonging to a specific filename."""
    try:
        # Chroma handles 'where' filters for deletion
        # Note: Ensure metadata 'source' exactly matches req.filename or path
        collection = ingestor.vectorstore._collection
        collection.delete(where={"source": req.filename})
        
        logger.info(f"🗑️ Deleted: {req.filename}")
        return {"status": "success", "message": f"Deleted {req.filename}"}
    except Exception as e:
        logger.error(f"Delete error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    try:
        unique_id = uuid.uuid4().hex[:8]
        temp_filename = f"{unique_id}_{file.filename}"
        file_path = os.path.join(UPLOAD_DIR, temp_filename)
        
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
            
        background_tasks.add_task(run_ingestion, file_path, file.filename)
        
        return {"status": "queued", "filename": file.filename}
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        start_time = time.time()
        initial_state = {
            "question": req.question, 
            "documents": [], 
            "metadata_manifest": [],
            "confidence_score": 0.0
        }
        
        result = workflow.invoke(initial_state)
        latency = (time.time() - start_time) * 1000
        
        return {
            "answer": result.get("answer", "No response."),
            "sources": result.get("metadata_manifest", []),
            "confidence_score": result.get("confidence_score", 0.0),
            "latency_ms": latency
        }
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail="Internal processing error.")

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)