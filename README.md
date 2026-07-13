# Multi-PDF AI RAG System

A privacy-focused **Retrieval-Augmented Generation (RAG)** application that lets you upload multiple PDF documents, index them locally, and ask questions with cited answers. Everything runs on your machine using Ollama for inference, ChromaDB for vector search, and BM25 for keyword retrieval fused with **Reciprocal Rank Fusion (RRF)**.

No cloud APIs required — your documents never leave your computer.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Interfaces                             │
│  ┌──────────────────────┐       ┌──────────────────────────────┐   │
│  │  app.py (Streamlit)  │       │  main.py (CLI + ReAct Agent) │   │
│  └──────────┬───────────┘       └──────────────┬───────────────┘   │
└─────────────┼──────────────────────────────────┼───────────────────┘
              │                                  │
              ▼                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    LangChain ReAct Agent (llama3.2)                 │
│  Tools: query_pdfs | summarize_pdf | extract_tables | JSON RAG      │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌──────────────────────────┐    ┌──────────────────────────────────┐
│  hybrid_search() (RRF)   │    │  pdfplumber (text + tables)      │
│  ┌────────┐ ┌─────────┐  │    └──────────────────────────────────┘
│  │ BM25   │ │ Chroma  │  │
│  │ rank   │ │ vector  │  │
│  └────────┘ └─────────┘  │
└──────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  config.py — models, chunking, paths, TOP_K / BM25_TOP_K / RRF_K    │
└─────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
rag_project/
├── app.py
├── main.py
├── config.py
├── config.yaml
├── requirements.txt
├── .env
├── .gitignore
├── src/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── loader.py
│   ├── chunking/
│   │   ├── __init__.py
│   │   └── chunker.py
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── embedder.py
│   ├── vectordb/
│   │   ├── __init__.py
│   │   └── vector_store.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── retriever.py
│   ├── prompts/
│   │   ├── __init__.py
│   │   └── prompt_templates.py
│   ├── llm/
│   │   ├── __init__.py
│   │   └── llm_client.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
├── tests/
│   └── test_app.py
├── logs/
│   └── app.log
├── pdfs/
├── chroma_db/
└── chroma_langchain_db/
```

The top-level `app.py`, `main.py`, `config.py`, and `vector.py` remain as compatibility entry points while the implementation lives under `src/`.

## Setup

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai/) installed and running
- Ollama model: `llama3.2`

### Install

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
ollama pull llama3.2
```

On first run, `sentence-transformers` downloads the `all-MiniLM-L6-v2` embedding model automatically.

## Usage

### Streamlit UI (recommended)

```powershell
streamlit run app.py
```

1. Upload one or more PDFs in the sidebar.
2. Click **Index uploaded PDFs**.
3. Ask questions in the chat input.
4. Expand **Sources** to see filename and page citations.
5. Use sidebar buttons to **Summarize PDF** or **Extract table** from a page.

### CLI

```powershell
python main.py
```

Place PDF files in `./pdfs/` before starting, or use the Streamlit uploader. Type `q` to quit.

## Example Queries

| Query | Tool used |
|-------|-----------|
| "What are the main findings in the report?" | `query_pdfs` |
| "Summarize annual_report.pdf" | `summarize_pdf` |
| "Show tables on page 3 of invoice.pdf" | `extract_tables` |
| "List all dates mentioned in the contract" | `extract_structured_data` (JSON) |

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Ollama `llama3.2` |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) |
| Vector store | ChromaDB |
| Keyword search | BM25 (`rank-bm25`) |
| Fusion | Reciprocal Rank Fusion (RRF) |
| PDF parsing | pdfplumber, PyMuPDF |
| Agent framework | LangChain ReAct AgentExecutor |
| Web UI | Streamlit |

## Configuration

All tunable values live in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `OLLAMA_MODEL` | `llama3.2` | Local LLM |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence embedding model |
| `CHROMA_DB_PATH` | `./chroma_db` | Vector DB persistence path |
| `CHUNK_SIZE` | `500` | Characters per text chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `TOP_K` | `5` | Final results after RRF |
| `BM25_TOP_K` | `10` | Candidates from each retriever |
| `RRF_K` | `60` | RRF smoothing constant |

## Troubleshooting

**No PDFs indexed** — Upload and index PDFs in Streamlit, or copy files into `./pdfs/` and restart the CLI.

**Ollama connection error** — Ensure Ollama is running: `ollama serve`

**Slow first run** — Embedding model and Chroma index are built on first ingestion; later runs reuse the persisted store.

**Empty table extraction** — Not all PDFs store tables as extractable text; scanned/image PDFs may need OCR (not included).

## License

Open source — available for personal use.
