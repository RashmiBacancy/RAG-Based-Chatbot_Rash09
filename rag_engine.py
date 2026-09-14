"""
Core RAG pipeline: Load -> Chunk -> Embed -> Store -> Retrieve -> Generate.
Each stage is one small function so the flow stays easy to follow end to end.
"""

import os
import shutil
import tempfile

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

CHROMA_DIR = os.path.join(tempfile.gettempdir(), "rag_chatbot_chroma")
COLLECTION_NAME = "rag_chatbot_docs"

RAG_PROMPT = """You are a helpful assistant answering questions using ONLY the context below.
If the answer is not contained in the context, say exactly:
"I don't have that information in the uploaded document(s)."
Do not use outside knowledge and do not guess.

Context:
{context}

Question: {question}

Answer:"""

LOADER_BY_EXTENSION = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
}


def get_text(response):
    """Normalize an LLM response's .content into plain text.

    Some Gemini models return a list of content blocks (text + an internal
    thought-signature block) instead of a plain string.
    """
    content = response.content
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def load_document(file_path):
    """Step 1: Document Upload / Parsing -> LangChain Document objects."""
    ext = os.path.splitext(file_path)[1].lower()
    loader_cls = LOADER_BY_EXTENSION.get(ext)
    if loader_cls is None:
        raise ValueError(f"Unsupported file type: {ext}")
    loader = loader_cls(file_path, encoding="utf-8") if loader_cls is TextLoader else loader_cls(file_path)
    return loader.load()


def chunk_documents(docs, chunk_size=800, chunk_overlap=100):
    """Step 2: Chunking -> smaller, semantically coherent pieces."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)


def get_embeddings_model():
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


def build_vector_store(chunks, reset=True):
    """Step 3 + 4: Embedding Generation -> Vector Storage (persisted with Chroma)."""
    embeddings = get_embeddings_model()

    if reset and has_existing_vector_store():
        # Clear via Chroma's own API first -- on Windows a raw shutil.rmtree can
        # silently fail to delete files still locked by the sqlite connection,
        # which left old and new chunks stacked together in earlier runs.
        try:
            Chroma(
                persist_directory=CHROMA_DIR,
                embedding_function=embeddings,
                collection_name=COLLECTION_NAME,
            ).delete_collection()
        except Exception:
            pass
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
    )
    return vector_store


def has_existing_vector_store():
    """True only if the named collection actually contains documents.

    Checking just for a non-empty CHROMA_DIR isn't enough: an older store
    (or a differently-named collection left over after a schema change) can
    make the folder non-empty while the collection this app expects is
    empty, which silently produced a "ready" chatbot with zero retrievable
    chunks.
    """
    if not os.path.exists(CHROMA_DIR) or not os.listdir(CHROMA_DIR):
        return False
    try:
        embeddings = get_embeddings_model()
        store = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        )
        return store._collection.count() > 0
    except Exception:
        return False


def load_existing_vector_store():
    """Reattach to the Chroma store persisted by a previous run/session."""
    if not has_existing_vector_store():
        return None
    embeddings = get_embeddings_model()
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )


def get_llm(temperature=0.3):
    return ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", temperature=temperature)


def build_retriever(vector_store, k=4):
    """Step 5: Retrieval -> top-k most relevant chunks for a query."""
    return vector_store.as_retriever(search_kwargs={"k": k})


def answer_question(retriever, llm, question, history=None):
    """Step 6: Response Generation, grounded in retrieved chunks + chat memory."""
    history = history or []

    relevant_docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in relevant_docs)

    system_message = SystemMessage(
        content=(
            "Use the CONTEXT below if relevant. If the user asks about a specific fact "
            "that would need to come from a document (e.g. a policy, a number, a name) "
            "and it isn't in the context or earlier in this conversation, say you don't "
            "have that information rather than inventing one. "
            "Math, calculations, and general-knowledge or reasoning questions are not "
            "document facts -- answer those directly and correctly using your own "
            "knowledge, whether or not the context is relevant to them.\n\nCONTEXT:\n"
            + context
        )
    )
    messages = [system_message] + history + [HumanMessage(content=question)]
    response = llm.invoke(messages)
    answer = get_text(response)

    sources = [
        {
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page"),
        }
        for doc in relevant_docs
    ]
    return answer, sources


def ingest_files(file_paths):
    """Convenience wrapper: run steps 1-4 for a batch of uploaded files."""
    all_docs = []
    for path in file_paths:
        all_docs.extend(load_document(path))
    chunks = chunk_documents(all_docs)
    vector_store = build_vector_store(chunks)
    return vector_store, chunks
