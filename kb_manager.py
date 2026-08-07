

from pathlib import Path
import tempfile

import streamlit as st

from rag import initialize_rag


def init_knowledge_base_state() -> None:
    """Create the shared RAG system once for the current Streamlit session."""
    if "kb_initialized" not in st.session_state:
        st.session_state.kb_initialized = True
        st.session_state.rag_system = initialize_rag()


def add_uploaded_file_to_knowledge_base(uploaded_file) -> str:
    """Embed an uploaded TXT or PDF file and add its chunks to the FAISS index."""
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_path = Path(temp_file.name)

    try:
        rag = st.session_state.rag_system
        if suffix == ".pdf":
            rag.add_pdf_file(str(temp_path), source_name=uploaded_file.name)
        else:
            rag.add_text_file(str(temp_path), source_name=uploaded_file.name)
        return uploaded_file.name
    finally:
        temp_path.unlink(missing_ok=True)
