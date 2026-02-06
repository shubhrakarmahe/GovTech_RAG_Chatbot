import streamlit as st
import requests
import uuid
import time
import logging
import json
from datetime import datetime

# --- LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GranicusUI")

# --- CONFIGURATION ---
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Granicus AI Assistant",
    page_icon="🏛️",
    layout="wide"
)

# --- SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "indexed_docs" not in st.session_state:
    st.session_state.indexed_docs = []

# --- HELPER FUNCTIONS ---
def parse_json_response(raw_json_str: str):
    """Parses the JSON string returned by the Graph synthesize node."""
    try:
        data = json.loads(raw_json_str)
        return (
            data.get("synthesized_answer", "No answer generated."),
            data.get("detailed_analysis", "No analysis available."),
            data.get("sources_cited", []),
            data.get("confidence_score", 0.0)
        )
    except Exception as e:
        logger.error(f"JSON Parsing failed: {e}")
        return raw_json_str, "Error: Response was not in expected JSON format.", [], 0.0

def get_library():
    """Fetches the list of indexed documents from the backend."""
    try:
        response = requests.get(f"{API_URL}/documents", timeout=5)
        if response.status_code == 200:
            st.session_state.indexed_docs = response.json().get("documents", [])
    except Exception as e:
        logger.error(f"Failed to fetch library: {e}")

def format_chat_for_download():
    """Converts session messages into a formatted Markdown string."""
    chat_text = f"# Granicus AI Chat Export\n"
    chat_text += f"**Session ID:** {st.session_state.thread_id}\n"
    chat_text += f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n---\n\n"
    
    for msg in st.session_state.messages:
        role = msg["role"].upper()
        content = msg["content"]
        chat_text += f"### {role}\n{content}\n\n"
        
        if role == "ASSISTANT":
            chat_text += f"**Confidence:** {msg.get('confidence', 0.0)*100:.0f}%\n"
            chat_text += f"**Analysis:** {msg.get('analysis', 'N/A')}\n"
            if msg.get("sources"):
                chat_text += f"**Sources:** {', '.join(msg['sources'])}\n"
        chat_text += "\n---\n\n"
    return chat_text

# --- SIDEBAR: DOCUMENT & CHAT MANAGEMENT ---
with st.sidebar:
    st.header("📂 Data Management")
    st.caption(f"Session ID: {st.session_state.thread_id[:8]}")
    
    # 1. Upload Section
    uploaded_files = st.file_uploader(
        "Upload Documents", 
        type=["pdf", "csv", "txt", "md"], 
        accept_multiple_files=True
    )
    
    if st.button("🚀 Process & Index", use_container_width=True):
        if uploaded_files:
            for file in uploaded_files:
                with st.spinner(f"Uploading {file.name}..."):
                    files = {"file": (file.name, file.getvalue(), file.type)}
                    requests.post(f"{API_URL}/upload", files=files)
            st.success("Ingestion queued!")
            time.sleep(1)
            get_library()

    st.divider()

    # 2. Indexed Library List
    st.subheader("🗄️ Indexed Library")
    if st.button("🔄 Refresh List", use_container_width=True):
        get_library()

    if st.session_state.indexed_docs:
        for doc in st.session_state.indexed_docs:
            cols = st.columns([0.8, 0.2])
            cols[0].caption(f"📄 {doc}")
            if cols[1].button("🗑️", key=f"del_{doc}"):
                with st.spinner("Deleting..."):
                    res = requests.post(f"{API_URL}/delete", json={"filename": doc})
                    if res.status_code == 200:
                        st.toast(f"Deleted {doc}")
                        get_library()
                        st.rerun()
    else:
        st.info("No documents found.")

    st.divider()

    # 3. Export & Reset
    st.subheader("⚙️ Actions")
    
    # Download Chat Button
    if st.session_state.messages:
        chat_md = format_chat_for_download()
        st.download_button(
            label="📥 Download Chat History",
            data=chat_md,
            file_name=f"chat_export_{st.session_state.thread_id[:8]}.md",
            mime="text/markdown",
            use_container_width=True
        )

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- MAIN CHAT INTERFACE ---
st.title("🏛️ Granicus AI")
st.caption("Structured RAG | Grounded Retrieval | Confidence Scoring")

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            # Confidence Meter
            conf = msg.get("confidence", 0.0)
            st.progress(conf)
            st.caption(f"Confidence Score: {conf*100:.0f}%")
            
            with st.expander("🔍 Reasoning & Sources"):
                st.info(msg.get("analysis", "No analysis available."))
                for s in msg.get("sources", []):
                    st.write(f"📄 `{s}`")

# Handle User Input
if prompt := st.chat_input("Ask a question based on your indexed documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing context..."):
            try:
                payload = {"question": prompt, "thread_id": st.session_state.thread_id}
                response = requests.post(f"{API_URL}/chat", json=payload, timeout=90)
                
                if response.status_code == 200:
                    res_data = response.json()
                    
                    # Parse the nested JSON response from the LLM
                    answer, analysis, sources, confidence = parse_json_response(res_data.get("answer", "{}"))
                    
                    # 1. Main Answer
                    st.markdown(answer)
                    
                    # 2. Confidence Visualization
                    st.progress(confidence)
                    st.caption(f"Confidence Score: {confidence*100:.0f}%")
                    
                    # 3. Reasoning and Sources
                    with st.expander("🔍 Reasoning & Sources"):
                        st.info(analysis)
                        if sources:
                            for s in sources: st.write(f"📄 `{s}`")
                        st.caption(f"Latency: {res_data.get('latency_ms', 0):.2f}ms")

                    # Save to state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "analysis": analysis,
                        "sources": sources,
                        "confidence": confidence
                    })
                    # Rerun to update the download button in sidebar
                    st.rerun()
                else:
                    st.error(f"Error: {response.status_code}")
            except Exception as e:
                st.error(f"Connection error: {e}")