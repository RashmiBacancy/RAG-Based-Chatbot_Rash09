# RAG-Based Chatbot — Assignment Submission

A working, end-to-end RAG chatbot built with **Python + LangChain + Gemini + Streamlit**,
directly following the pipeline from the training session:

```
Document Upload → Parsing → Chunking → Embedding → Vector Storage → Retrieval → Response Generation
```

It also adds conversation memory, multi-file support (PDF/DOCX/TXT/MD), a persisted
vector store, and a source-citation panel — the "beyond basic" pieces the assignment asks for.

## 1. Quick Start (5 minutes)

```bash
pip install -r requirements.txt
streamlit run app.py
```

1. Paste your free Gemini API key in the sidebar (get one at https://aistudio.google.com/app/apikey).
2. Upload a document — or use the included [`sample_docs/acme_handbook.txt`](sample_docs/acme_handbook.txt) to test immediately.
3. Click **Process documents**.
4. Ask a question in the chat box, e.g. *"How many vacation days do I get?"*

## 2. How the pieces map to the pipeline

| Pipeline stage | Where it happens | Notes |
|---|---|---|
| Upload / Parsing | [`app.py`](app.py) sidebar uploader → [`rag_engine.load_document`](rag_engine.py) | PDF, DOCX, TXT, MD supported via LangChain loaders |
| Chunking | [`rag_engine.chunk_documents`](rag_engine.py) | `RecursiveCharacterTextSplitter`, 800 chars / 100 overlap |
| Embedding | [`rag_engine.get_embeddings_model`](rag_engine.py) | Gemini `gemini-embedding-001` |
| Vector Storage | [`rag_engine.build_vector_store`](rag_engine.py) | Chroma, persisted to disk (survives restarts, unlike the workshop's in-memory store) |
| Retrieval | [`rag_engine.build_retriever`](rag_engine.py) | Top-k similarity search, k adjustable in UI |
| Response Generation | [`rag_engine.answer_question`](rag_engine.py) | Grounded prompt + conversation history, refuses to answer outside the document |

## 3. Roadmap — build it yourself, step by step

If you want to build this from scratch rather than run the finished app, follow these
steps in order. Each one is checkable on its own before moving to the next.

**Step 1 — Get a working LLM call**
- Install `langchain-google-genai`, get a Gemini API key.
- Call `llm.invoke("hello")` and print the response. ✅ Checkpoint: you get text back.

**Step 2 — Load a document**
- Pick one real file you care about (policy doc, README, PDF).
- Use `PyPDFLoader` / `TextLoader` / `Docx2txtLoader` to load it into `Document` objects.
- ✅ Checkpoint: `print(len(docs), docs[0].page_content[:200])` looks right.

**Step 3 — Chunk it**
- Run it through `RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)`.
- ✅ Checkpoint: print each chunk — no chunk should span unrelated topics.

**Step 4 — Embed + store**
- Embed one chunk with `GoogleGenerativeAIEmbeddings`, confirm you get a vector back.
- Push all chunks into a vector store (`Chroma` for persistence, or `InMemoryVectorStore`
  for a quick test).
- ✅ Checkpoint: vector store reports the right chunk count.

**Step 5 — Retrieve**
- `retriever = vector_store.as_retriever(search_kwargs={"k": 4})`
- Run `retriever.invoke("some question")` and read what comes back.
- ✅ Checkpoint: the most relevant chunk is actually on topic, even with different wording.

**Step 6 — Generate a grounded answer**
- Build a prompt that says "answer ONLY from this context, say so if it's not there."
- Feed it `context + question`, call the LLM, print the answer.
- ✅ Checkpoint: ask something in the doc (correct answer) and something not in the doc
  (should say "I don't have that information").

**Step 7 — Add memory**
- Keep a running list of `HumanMessage`/`AIMessage` and prepend it to every call.
- ✅ Checkpoint: "What's my name?" works after you've told it your name earlier in the chat.

**Step 8 — Wrap it in a UI**
- `streamlit run app.py` — file upload widget → process button → chat box.
- This repo's [`app.py`](app.py) already does this; use it as the reference.

**Step 9 — Make it production-minded (stretch goals)**
- Persist the vector store to disk (done here via Chroma) instead of losing it on restart.
- Show *which* chunks were used to answer (source citations) — builds trust, catches
  hallucinations.
- Support multiple file types, not just `.txt`.
- Add basic evaluation: a small set of Q&A pairs you run after every change to catch regressions.
- Handle rate limits / API errors gracefully instead of crashing the whole app.

## 4. What's intentionally left out (mentioned, not required)

Per the workshop's "natural next steps": tool-calling/agents, long-term memory across
sessions, streaming token-by-token responses, and formal evaluation harnesses. These are
good v2 ideas but not needed to satisfy the assignment's core pipeline.

## 5. File map

```
app.py              Streamlit UI (upload, process, chat)
rag_engine.py        Core pipeline: load → chunk → embed → store → retrieve → answer
requirements.txt     Dependencies
sample_docs/         A ready-to-use test document
.env.example         Copy to .env and fill in GOOGLE_API_KEY (optional — UI also accepts it)
```
