import os
import io
import re
import logging
import pandas as pd
import pdfplumber
from typing import List
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# --- CONFIGURATION ---
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("UnifiedIngestor")

PERSIST_DIR = os.getenv("PERSIST_DIRECTORY", "./chroma_db")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

class UnifiedIngestor:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        self.llm = ChatGroq(model=LLM_MODEL, temperature=0)
        self.persist_directory = "./chroma_db"
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )

    # --- NEW HELPER METHODS ---

    def _get_summary(self, text: str, context: str) -> str:
        """Generates a brief summary for metadata."""
        try:
            prompt = f"Summarize this {context} in one concise sentence for metadata: {text[:1500]}"
            res = self.llm.invoke(prompt)
            return res.content.strip()
        except Exception as e:
            logger.warning(f"Summary generation failed: {e}")
            return "General content."

    def _split_by_headings(self, text: str) -> List[dict]:
        """Regex-based splitter to identify common heading patterns in PDF/TXT."""
        # Matches patterns like "1.0 Title", "Section 1:", or "APPENDIX A"
        heading_pattern = r"(\n(?:[0-9]+\.[0-9.]*|[A-Z]{2,}|Section\s\d+)[^\n]+)"
        parts = re.split(heading_pattern, text)
        
        sections = []
        current_heading = "General Introduction"
        
        for part in parts:
            if not part.strip(): continue
            if re.match(heading_pattern, part):
                current_heading = part.strip()
            else:
                sections.append({"header": current_heading, "content": part.strip()})
        return sections

    # --- CSV LOGIC (RETAINED AS REQUESTED) ---

    def _get_column_descriptions(self, df: pd.DataFrame) -> str:
        """Generates a mapping of column names to their likely meaning."""
        try:
            cols = list(df.columns)
            sample = df.head(3).to_string()
            prompt = f"""
            Analyze these CSV columns: {cols}
            Sample Data: {sample}
            Provide a concise 'Data Dictionary' string where you explain what each column represents.
            Format: col_name: description; col_name: description;
            """
            res = self.llm.invoke(prompt)
            return res.content.strip()
        except Exception as e:
            logger.warning(f"Column summary generation failed: {e}")
            return "Column mapping unavailable."

    def process_csv(self, file_content: bytes, filename: str) -> List[Document]:
        logger.info(f"Processing CSV: {filename}")
        try:
            df = pd.read_csv(io.BytesIO(file_content))
            column_summary = self._get_column_descriptions(df)
            file_summary = self._get_summary(df.head(10).to_string(), "CSV Data")
            
            docs = []
            for i, row in df.iterrows():
                content = " | ".join([f"{k}: {v}" for k, v in row.to_dict().items() if pd.notna(v)])
                docs.append(Document(
                    page_content=content,
                    metadata={
                        "source": filename, 
                        "type": "csv_row", 
                        "row": i + 1, 
                        "file_summary": file_summary,
                        "column_content": column_summary
                    }
                ))
            return docs
        except Exception as e:
            logger.error(f"Error reading CSV: {e}")
            return []

    # --- UPDATED PDF/TXT/MD LOGIC WITH HEADING AWARENESS ---

    def process_pdf(self, file_path: str) -> List[Document]:
        logger.info(f"Processing PDF: {os.path.basename(file_path)}")
        full_text = ""
        filename = os.path.basename(file_path)
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    full_text += (page.extract_text() or "") + "\n"
            
            sections = self._split_by_headings(full_text)
            docs = []
            for sec in sections:
                chunk_sum = self._get_summary(sec["content"], "Paragraph Section")
                docs.append(Document(
                    page_content=f"## {sec['header']}\n{sec['content']}",
                    metadata={
                        "source": filename, 
                        "type": "pdf_text", 
                        "header": sec["header"],
                        "chunk_summary": chunk_sum,
                        "column_content": "N/A"
                    }
                ))
            return self.text_splitter.split_documents(docs)
        except Exception as e:
            logger.error(f"Error reading PDF: {e}")
            return []

    def process_txt(self, file_content: bytes, filename: str) -> List[Document]:
        logger.info(f"Processing TXT: {filename}")
        text = file_content.decode("utf-8", errors="ignore")
        sections = self._split_by_headings(text)
        docs = []
        for sec in sections:
            chunk_sum = self._get_summary(sec["content"], "Text Section")
            docs.append(Document(
                page_content=sec["content"], 
                metadata={
                    "source": filename, 
                    "type": "text", 
                    "header": sec["header"],
                    "chunk_summary": chunk_sum,
                    "column_content": "N/A"
                }
            ))
        return self.text_splitter.split_documents(docs)

    def process_md(self, file_content: bytes, filename: str) -> List[Document]:
        logger.info(f"Processing Markdown: {filename}")
        text = file_content.decode("utf-8", errors="ignore")
        
        # Split by actual Markdown headers
        headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
        md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        sections = md_splitter.split_text(text)
        
        final_docs = []
        for sec in sections:
            chunk_sum = self._get_summary(sec.page_content, "Markdown Section")
            sec.metadata.update({
                "source": filename,
                "type": "markdown",
                "chunk_summary": chunk_sum,
                "column_content": "N/A"
            })
            final_docs.append(sec)
        return final_docs

    # --- UNIFIED INGESTION ENTRY POINT ---

    def ingest(self, file_path: str):
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return 0
            
        filename = os.path.basename(file_path)
        ext = filename.split(".")[-1].lower()
        docs = []

        if ext == "pdf":
            docs = self.process_pdf(file_path)
        elif ext == "csv":
            with open(file_path, "rb") as f:
                docs = self.process_csv(f.read(), filename)
        elif ext == "txt":
            with open(file_path, "rb") as f:
                docs = self.process_txt(f.read(), filename)
        elif ext == "md":
            pdf_sibling = os.path.splitext(file_path)[0] + ".pdf"
            if os.path.exists(pdf_sibling):
                logger.info(f"Skipping {filename}: PDF version present.")
                return 0
            with open(file_path, "rb") as f:
                docs = self.process_md(f.read(), filename)
        
        if docs:
            vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=self.embeddings)
            vectorstore.add_documents(docs)
            logger.info(f"✅ Indexed {len(docs)} chunks from {filename}")
            return len(docs)

        return 0
