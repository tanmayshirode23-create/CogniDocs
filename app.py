
from dotenv import load_dotenv
import os
import tempfile

import streamlit as st
load_dotenv()
from rag_pipeline import (
    load_and_split_pdf,
    build_vector_store,
    build_llm,
    answer_question,
    ConversationMemory,
)


# --------------------------------------------------------------------------------------
# Page config & lightweight visual identity
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="PDF Q&A Assistant", page_icon="📄", layout="wide")

CUSTOM_CSS = """
<style>
    html, body, [class*="css"] { font-family: 'Georgia', 'Iowan Old Style', serif; }
    .app-header { border-bottom: 2px solid #1F3A5F; padding-bottom: 0.6rem; margin-bottom: 1rem; }
    .app-header h1 { color: #1F3A5F; font-size: 2rem; margin-bottom: 0.1rem; }
    .app-header p { color: #5c6470; font-size: 0.98rem; margin-top: 0; }
    .stButton>button {
        background-color: #1F3A5F; color: #F3F6FA; border-radius: 4px; border: none;
        padding: 0.5rem 1.2rem; font-weight: 600;
    }
    .stButton>button:hover { background-color: #2c517f; color: #F3F6FA; }
    section[data-testid="stSidebar"] { background-color: #F3F6FA; }
    .source-chunk {
        background-color: #F3F6FA; border-left: 3px solid #1F3A5F; padding: 0.6rem 0.9rem;
        margin-bottom: 0.5rem; font-size: 0.88rem; border-radius: 2px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="app-header">
        <h1>📄 CogniDocs</h1>
        <p>Upload a PDF, then ask questions about it in plain language. Powered by
        Retrieval-Augmented Generation (RAG) with Google Gemini.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------------------
# Session state initialization
# --------------------------------------------------------------------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory()
if "chat_display" not in st.session_state:
    st.session_state.chat_display = []  # list of dicts: {role, content, sources}
if "indexed_filename" not in st.session_state:
    st.session_state.indexed_filename = None


# --------------------------------------------------------------------------------------
# Sidebar: configuration
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        api_key = os.getenv("GOOGLE_API_KEY")

    model_name = st.selectbox(
        "Chat model",
        options=["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"],
        index=0,
        help="gemini-2.5-flash is the recommended free-tier default.",
    )

    st.divider()
    st.subheader("📄 Upload a PDF")
    uploaded_pdf = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_pdf is not None and uploaded_pdf.name != st.session_state.indexed_filename:
        if not api_key:
            st.warning("Enter your Gemini API key above before indexing a PDF.")
        else:
            with st.spinner("Reading PDF, splitting into chunks, and building the search index..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_pdf.read())
                        tmp_path = tmp.name

                    chunks = load_and_split_pdf(tmp_path)
                    st.session_state.vector_store = build_vector_store(chunks, api_key)
                    st.session_state.indexed_filename = uploaded_pdf.name
                    st.session_state.memory.clear()
                    st.session_state.chat_display = []

                    os.unlink(tmp_path)
                    st.success(f"Indexed **{uploaded_pdf.name}** — {len(chunks)} chunks ready for Q&A.")
                except Exception as e:
                    st.error(f"Failed to index PDF: {e}")

    if st.session_state.indexed_filename:
        st.caption(f"✅ Currently indexed: **{st.session_state.indexed_filename}**")
        if st.button("🗑️ Clear document & conversation", use_container_width=True):
            st.session_state.vector_store = None
            st.session_state.indexed_filename = None
            st.session_state.memory.clear()
            st.session_state.chat_display = []
            st.rerun()

    st.divider()
    st.caption(
        "Your PDF content and questions are sent to Google's Gemini API for processing and are "
        "not stored by this app. Avoid uploading documents with sensitive data you don't want "
        "processed by a third-party API."
    )


# --------------------------------------------------------------------------------------
# Main chat area
# --------------------------------------------------------------------------------------
if not st.session_state.vector_store:
    st.info("👈 Upload a PDF in the sidebar and enter your Gemini API key to get started.")
else:
    # Render existing chat history
    for turn in st.session_state.chat_display:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("sources"):
                with st.expander(f"📚 Sources ({len(turn['sources'])} chunks used)"):
                    for doc in turn["sources"]:
                        page = doc.metadata.get("page", "?")
                        st.markdown(
                            f'<div class="source-chunk"><b>Page {page}</b><br>{doc.page_content}</div>',
                            unsafe_allow_html=True,
                        )

    question = st.chat_input("Ask a question about the PDF...")

    if question:
        if not api_key:
            st.error("Please enter your Gemini API key in the sidebar.")
        else:
            st.session_state.chat_display.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner(f"Thinking with {model_name}..."):
                    try:
                        llm = build_llm(api_key, model_name)
                        answer, sources = answer_question(
                            llm, st.session_state.vector_store, question, st.session_state.memory
                        )
                        st.markdown(answer)
                        if sources:
                            with st.expander(f"📚 Sources ({len(sources)} chunks used)"):
                                for doc in sources:
                                    page = doc.metadata.get("page", "?")
                                    st.markdown(
                                        f'<div class="source-chunk"><b>Page {page}</b><br>{doc.page_content}</div>',
                                        unsafe_allow_html=True,
                                    )
                        st.session_state.chat_display.append(
                            {"role": "assistant", "content": answer, "sources": sources}
                        )
                    except Exception as e:
                        error_msg = f"Something went wrong while calling the Gemini API: {e}"
                        st.error(error_msg)
                        st.info(
                            "Common causes: an invalid API key, hitting the free-tier rate limit "
                            "(wait a minute and retry), or a network issue."
                        )

st.divider()
st.caption(
    "Built for Epochs '26 Assignment 11 · RAG pipeline: PyPDFLoader → RecursiveCharacterTextSplitter "
    "→ Gemini Embeddings → FAISS → Gemini Chat · This tool provides AI-generated answers grounded "
    "in the uploaded document — always verify against the original source for important decisions."
)
