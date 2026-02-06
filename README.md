# GovTech_RAG_Chatbot

## Project Overview

GovTech AI is a specialized Retrieval-Augmented Generation (RAG) system designed for government technology applications. It allows users to upload official documents—such as PDFs, CSVs, TXT, and Markdown files—and interact with them through a grounded conversational interface. The system is built to ensure high accuracy, transparency, and data security, making it suitable for public sector use cases where factual grounding is critical.

### Key Features

* **Factual Grounding & Confidence Scoring:** Uses a LangGraph-based orchestration that enforces strict rules: the AI only answers based on provided documents and provides a confidence score (0.0–1.0) for every response.
* **Smart Retrieval & Reranking:** Employs semantic vector search combined with a `FlashRank` reranking step to ensure the most relevant policy or data sections are prioritized.
* **Heading-Aware Ingestion:** Automatically detects document headers and sections during indexing to preserve context and enable precise citations.
* **Background Processing:** Handles document ingestion as asynchronous background tasks, ensuring the UI remains responsive during large file uploads.
* **Government-Ready UI:** Features a Streamlit dashboard with confidence meters, reasoning transparency (analysis of why an answer was chosen), and session export capabilities.

---

## System Architecture

The system follows a modular 5-part structure to ensure scalability and maintainability:

1. **Ingestion Pipeline:** Processes and indexes multi-format government documents.
2. **Vector Database:** Stores document embeddings in ChromaDB for fast semantic search.
3. **Retrieval Engine:** Identifies and ranks relevant information based on user intent and metadata filters.
4. **Orchestration Layer (Graph-Based):** Manages the workflow between retrieval, synthesis, and error handling.
5. **User Interface:** A conversational front-end for user interaction and document management.

---

## Technology Stack

* **LLM:** Llama-3.3-70b-versatile (via Groq).
* **Orchestration:** LangGraph and LangChain.
* **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`.
* **Databases:** ChromaDB (Vector) and DiskCache (Query Cache).
* **Backend:** FastAPI.
* **Frontend:** Streamlit.

---

## Setup Instructions

### 1. Prerequisites

* Python 3.9+
* Groq API Key (for LLM access)

### 2. Environment Configuration

Create a `.env` file in the root directory:

```bash
GROQ_API_KEY=groq_api_key
LANGCHAIN_API_KEY=langchain_api_key

# Configuration
LLM_MODEL_NAME=llama-3.3-70b-versatile
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
PERSIST_DIRECTORY=./chroma_db
DATA_DIRECTORY=./data

```

### 3. Installation

Install the required dependencies:

```bash
pip install -r requirements.txt

```

### 4. Running the Application

The system requires both the backend API and the frontend UI to be running.

**Step A: Start the Backend API**

```bash
python app.py

```

*The API will start on `http://localhost:8000`.*

**Step B: Start the Frontend UI**

```bash
streamlit run streamlit_app.py

```

*Access the dashboard via your browser at the provided local URL.*
<img width="1466" height="647" alt="image" src="https://github.com/user-attachments/assets/26350503-fa21-4f62-80f0-652df10885c3" />

---

## API Documentation Summary

* `POST /chat`: Submit a question and receive a grounded answer with citations.
* `POST /upload`: Upload files for background indexing.
* `GET /documents`: List all currently indexed documents.
* `POST /delete`: Remove a specific document from the vector store.
* `GET /health`: Provides a simple status check to confirm the backend API is running and reachable.
