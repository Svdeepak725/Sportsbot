"""
Knowledge Base Manager for the Sports Chatbot.
Handles UI components and operations for managing the RAG knowledge base.
"""

import streamlit as st
from pathlib import Path
import tempfile
from rag import RAGSystem, initialize_rag, SportsDataIntegrator


def init_knowledge_base_state():
    """Initialize session state for knowledge base management."""
    if "kb_initialized" not in st.session_state:
        st.session_state.kb_initialized = False
        st.session_state.rag_system = initialize_rag()


def add_uploaded_file_to_knowledge_base(uploaded_file) -> str:
    """Add a Streamlit-uploaded TXT or PDF file to the RAG knowledge base."""
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = Path(tmp_file.name)

    try:
        rag = st.session_state.rag_system
        if suffix == ".pdf":
            rag.add_pdf_file(str(tmp_path), source_name=uploaded_file.name)
        else:
            rag.add_text_file(str(tmp_path), source_name=uploaded_file.name)
        return uploaded_file.name
    finally:
        tmp_path.unlink(missing_ok=True)


def display_kb_stats():
    """Display statistics about the knowledge base."""
    rag = st.session_state.rag_system
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Vector Store Status", "✅ Active")
    with col2:
        try:
            # Try to get document count (FAISS doesn't have built-in count)
            st.metric("Embedding Model", "all-MiniLM-L6-v2")
        except:
            st.metric("Embedding Model", "Ready")
    with col3:
        st.metric("Vector Store", "FAISS (Local)")


def handle_file_upload():
    """Handle file uploads for knowledge base enrichment."""
    st.subheader("📤 Upload Documents")
    
    uploaded_file = st.file_uploader(
        "Upload a file (TXT or PDF)",
        type=["txt", "pdf"],
        help="Upload sports-related documents to enrich your knowledge base"
    )
    
    if uploaded_file:
        rag = st.session_state.rag_system
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = tmp_file.name
        
        try:
            with st.spinner("Processing document..."):
                if uploaded_file.name.endswith('.pdf'):
                    rag.add_pdf_file(tmp_path, source_name=uploaded_file.name)
                else:
                    rag.add_text_file(tmp_path, source_name=uploaded_file.name)
            
            st.success(f"✅ Successfully added '{uploaded_file.name}' to knowledge base!")
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
        finally:
            Path(tmp_path).unlink()  # Clean up temp file


def handle_text_input():
    """Handle direct text input for knowledge base enrichment."""
    st.subheader("✏️ Add Text Content")
    
    text_content = st.text_area(
        "Enter sports-related content",
        height=150,
        placeholder="E.g., Facts about sports, player statistics, team information, etc.",
        help="Add custom sports knowledge to your RAG system",
        key="kb_text_area"
    )
    
    if st.button("Add to Knowledge Base", key="add_text_btn"):
        if text_content.strip():
            rag = st.session_state.rag_system
            
            try:
                with st.spinner("Adding content..."):
                    rag.add_text(text_content, source="manual_input")
                
                st.success("✅ Content added successfully!")
            except Exception as e:
                st.error(f"❌ Error adding content: {str(e)}")
        else:
            st.warning("⚠️ Please enter some content first")


def handle_api_integration():
    """Handle sports API integration."""
    st.subheader("🔗 API Integration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        sport_type = st.selectbox(
            "Select Sport",
            ["Cricket", "Football", "Basketball", "Tennis", "Formula 1"]
        )
    
    with col2:
        if st.button("Fetch & Add Data", key="api_btn"):
            rag = st.session_state.rag_system
            
            try:
                with st.spinner(f"Fetching {sport_type} data..."):
                    SportsDataIntegrator.add_api_data_to_knowledge_base(
                        rag,
                        sport_type.lower()
                    )
                
                st.success(f"✅ {sport_type} data added successfully!")
            except Exception as e:
                st.error(f"❌ Error fetching data: {str(e)}")
    
    st.info("💡 Note: Integrate with APIs like ESPN, TheSportsDB, or Cricket-Data for real-time sports information")


def handle_knowledge_base_reset():
    """Handle knowledge base reset."""
    st.subheader("🔄 Reset Knowledge Base")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.warning("⚠️ This will clear all custom documents and reset to default sports knowledge.")
    
    with col2:
        if st.button("Reset KB", key="reset_kb_btn"):
            rag = st.session_state.rag_system
            
            try:
                with st.spinner("Resetting knowledge base..."):
                    rag.clear_knowledge_base()
                
                st.success("✅ Knowledge base reset successfully!")
            except Exception as e:
                st.error(f"❌ Error resetting knowledge base: {str(e)}")


def display_knowledge_base_panel():
    """Display the complete knowledge base management panel."""
    st.sidebar.markdown("---")
    st.sidebar.title("📚 Knowledge Base Manager")
    
    # Initialize KB state
    init_knowledge_base_state()
    
    # Tab-based interface
    tab1, tab2, tab3, tab4 = st.sidebar.tabs(["Stats", "Upload", "Add Text", "Reset"])
    
    with tab1:
        display_kb_stats()
    
    with tab2:
        handle_file_upload()
    
    with tab3:
        handle_text_input()
    
    with tab4:
        handle_knowledge_base_reset()


def display_retrieval_debug():
    """Display RAG retrieval debugging information."""
    with st.expander("🔍 RAG Debug Info"):
        debug_query = st.text_input("Query for debugging:")
        
        if debug_query:
            rag = st.session_state.rag_system
            results = rag.retrieve_with_scores(debug_query, k=3)
            
            if results:
                st.write("**Retrieved Documents:**")
                for i, (doc, score) in enumerate(results, 1):
                    with st.container(border=True):
                        st.write(f"**Match {i}** (Score: {score:.3f})")
                        content = doc.get('content', '')[:200]
                        st.write(content + "...")
                        source = doc.get('source', 'Unknown')
                        st.caption(f"Source: {source}")
            else:
                st.warning("No relevant documents found")
