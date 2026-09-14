import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from rag_engine import (
    ingest_files,
    build_retriever,
    get_llm,
    answer_question,
    has_existing_vector_store,
    load_existing_vector_store,
)

load_dotenv()

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

if "history" not in st.session_state:
    st.session_state.history = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "auto_load_tried" not in st.session_state:
    st.session_state.auto_load_tried = False

# Auto-reattach to a knowledge base already indexed in a previous run, so a
# restart doesn't force the user through upload/process again.
if (
    not st.session_state.auto_load_tried
    and st.session_state.retriever is None
    and os.environ.get("GOOGLE_API_KEY")
    and has_existing_vector_store()
):
    vector_store = load_existing_vector_store()
    if vector_store is not None:
        st.session_state.retriever = build_retriever(vector_store, k=4)
st.session_state.auto_load_tried = True

ready = bool(os.environ.get("GOOGLE_API_KEY")) and st.session_state.retriever is not None

with st.sidebar:
    with st.expander("⚙️ Setup", expanded=not ready):
        api_key_input = st.text_input(
            "Gemini API Key",
            type="password",
            value=os.environ.get("GOOGLE_API_KEY", ""),
            help="Get one free at https://aistudio.google.com/app/apikey",
        )
        if api_key_input:
            os.environ["GOOGLE_API_KEY"] = api_key_input

        st.divider()
        st.subheader("📄 Knowledge Base")

        uploaded_files = st.file_uploader(
            "Upload documents (PDF, DOCX, TXT, MD)",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True,
        )

        k = st.slider("Chunks retrieved per question (k)", 1, 8, 4)

        if st.button("Process documents", type="primary", disabled=not uploaded_files):
            if not os.environ.get("GOOGLE_API_KEY"):
                st.error("Enter your Gemini API key first.")
            else:
                with st.spinner("Loading → Chunking → Embedding → Storing..."):
                    tmp_dir = tempfile.mkdtemp()
                    paths = []
                    for f in uploaded_files:
                        path = os.path.join(tmp_dir, f.name)
                        with open(path, "wb") as out:
                            out.write(f.getbuffer())
                        paths.append(path)

                    vector_store, chunks = ingest_files(paths)
                    st.session_state.retriever = build_retriever(vector_store, k=k)
                    st.session_state.chunks = chunks
                    st.session_state.history = []

                st.success(f"Ready! Indexed {len(chunks)} chunks from {len(uploaded_files)} file(s).")
                st.rerun()

    if ready:
        st.success("✅ Knowledge base loaded — just ask your question below.")
    if st.session_state.chunks:
        st.caption(f"📚 {len(st.session_state.chunks)} chunks currently indexed")
        with st.expander("Preview chunks"):
            for i, c in enumerate(st.session_state.chunks[:10]):
                st.text(f"--- Chunk {i + 1} ---\n{c.page_content[:200]}")

    st.divider()
    if st.button("Clear conversation"):
        st.session_state.history = []
        st.rerun()
    if st.button("Reset knowledge base"):
        import shutil
        from rag_engine import CHROMA_DIR

        shutil.rmtree(CHROMA_DIR, ignore_errors=True)
        st.session_state.retriever = None
        st.session_state.chunks = []
        st.session_state.history = []
        st.session_state.auto_load_tried = False
        st.rerun()

st.title("🤖 RAG-Based Chatbot")
st.caption("Document Upload → Parsing → Chunking → Embedding → Vector Storage → Retrieval → Response Generation")

for msg in st.session_state.history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

question = st.chat_input("Ask a question about your uploaded document(s)...")

if question:
    if not os.environ.get("GOOGLE_API_KEY"):
        st.error("Enter your Gemini API key in the sidebar first.")
    elif st.session_state.retriever is None:
        st.error("Upload and process at least one document first.")
    else:
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant chunks and generating answer..."):
                llm = get_llm()
                answer, sources = answer_question(
                    st.session_state.retriever, llm, question, st.session_state.history
                )
                st.markdown(answer)
                with st.expander("🔍 Retrieved sources"):
                    for i, s in enumerate(sources):
                        label = s["source"]
                        if s["page"] is not None:
                            label += f" (page {s['page'] + 1})"
                        st.markdown(f"**{i + 1}. {label}**")
                        st.text(s["content"][:400])

        st.session_state.history.append(HumanMessage(content=question))
        st.session_state.history.append(AIMessage(content=answer))
