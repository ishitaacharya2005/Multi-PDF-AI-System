import os
import re
from typing import List, Optional

import pdfplumber
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from config import (
    BM25_TOP_K,
    CHROMA_DB_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    RRF_K,
    TOP_K,
)

COLLECTION_NAME = "pdf_documents"
PDF_DIR = os.path.join(os.path.dirname(CHROMA_DB_PATH) or ".", "pdfs")

vector_store: Optional[Chroma] = None
bm25_index: Optional[BM25Okapi] = None
indexed_documents: List[Document] = []
last_retrieval_sources: List[dict] = []


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def _chunk_text(text: str) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def _doc_key(doc: Document) -> str:
    meta = doc.metadata or {}
    return f"{meta.get('source', '')}|{meta.get('page', '')}|{meta.get('chunk', '')}|{doc.page_content[:80]}"


def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def _init_vector_store() -> Chroma:
    global vector_store
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DB_PATH,
        embedding_function=_get_embeddings(),
    )
    return vector_store


def load_pdfs(pdf_paths: List[str]) -> List[Document]:
    """Load PDF files with pdfplumber and return chunked Document objects."""
    documents: List[Document] = []
    for pdf_path in pdf_paths:
        filename = os.path.basename(pdf_path)
        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                if not text.strip():
                    continue
                for chunk_idx, chunk in enumerate(_chunk_text(text)):
                    documents.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                "source": filename,
                                "page": page_number,
                                "chunk": chunk_idx,
                            },
                        )
                    )
    return documents


def build_bm25_index(documents: List[Document]) -> BM25Okapi:
    """Build a BM25 index from the provided documents."""
    global bm25_index, indexed_documents
    indexed_documents = list(documents)
    tokenized_corpus = [_tokenize(doc.page_content) for doc in indexed_documents]
    bm25_index = BM25Okapi(tokenized_corpus)
    return bm25_index


def _sync_documents_from_chroma() -> None:
    """Reload in-memory document list from the persisted Chroma collection."""
    global indexed_documents
    if vector_store is None:
        indexed_documents = []
        return
    result = vector_store._collection.get(include=["documents", "metadatas"])
    docs = result.get("documents") or []
    metas = result.get("metadatas") or []
    indexed_documents = [
        Document(page_content=content, metadata=meta or {})
        for content, meta in zip(docs, metas)
    ]


def hybrid_search(query: str, top_k: int = TOP_K) -> List[Document]:
    """Merge BM25 and ChromaDB results using Reciprocal Rank Fusion."""
    global last_retrieval_sources

    if not indexed_documents or bm25_index is None:
        last_retrieval_sources = []
        return []

    tokenized_query = _tokenize(query)
    bm25_scores = bm25_index.get_scores(tokenized_query)
    bm25_ranked_indices = sorted(
        range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
    )[:BM25_TOP_K]

    chroma_results = vector_store.similarity_search(query, k=BM25_TOP_K)

    rrf_scores: dict[str, float] = {}
    doc_by_key: dict[str, Document] = {}

    for rank, idx in enumerate(bm25_ranked_indices):
        doc = indexed_documents[idx]
        key = _doc_key(doc)
        doc_by_key[key] = doc
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)

    for rank, doc in enumerate(chroma_results):
        key = _doc_key(doc)
        doc_by_key[key] = doc
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)

    ranked_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)
    results = [doc_by_key[key] for key in ranked_keys[:top_k]]

    last_retrieval_sources = [
        {"source": doc.metadata.get("source", "unknown"), "page": doc.metadata.get("page", "?")}
        for doc in results
    ]
    return results


def ingest_pdfs(pdf_paths: List[str]) -> int:
    """Load PDFs, add to ChromaDB, and rebuild the BM25 index."""
    global vector_store

    if not pdf_paths:
        return count_indexed_pdfs()

    if vector_store is None:
        _init_vector_store()

    new_docs = load_pdfs(pdf_paths)
    if not new_docs:
        return count_indexed_pdfs()

    ids = [
        f"{doc.metadata['source']}_{doc.metadata['page']}_{doc.metadata['chunk']}"
        for doc in new_docs
    ]
    vector_store.add_documents(documents=new_docs, ids=ids)
    _sync_documents_from_chroma()
    build_bm25_index(indexed_documents)
    return count_indexed_pdfs()


def count_indexed_pdfs() -> int:
    """Return the number of unique PDF filenames in the index."""
    sources = {doc.metadata.get("source") for doc in indexed_documents if doc.metadata.get("source")}
    return len(sources)


def get_documents_by_source(filename: str) -> List[Document]:
    """Return all indexed chunks belonging to a PDF filename."""
    return [doc for doc in indexed_documents if doc.metadata.get("source") == filename]


def resolve_pdf_path(filename: str) -> Optional[str]:
    """Resolve a PDF filename to its on-disk path."""
    path = os.path.join(PDF_DIR, filename)
    return path if os.path.isfile(path) else None


def extract_tables(filename: str, page: int) -> str:
    """Extract tables from a PDF page as markdown using pdfplumber."""
    pdf_path = resolve_pdf_path(filename)
    if not pdf_path:
        return f"PDF file '{filename}' not found in {PDF_DIR}."

    with pdfplumber.open(pdf_path) as pdf:
        if page < 1 or page > len(pdf.pages):
            return f"Page {page} is out of range for '{filename}' (1-{len(pdf.pages)})."

        tables = pdf.pages[page - 1].extract_tables()
        if not tables:
            return f"No tables found on page {page} of '{filename}'."

        markdown_parts = []
        for i, table in enumerate(tables, start=1):
            if not table:
                continue
            header = table[0]
            rows = table[1:] if len(table) > 1 else []
            md = f"### Table {i}\n\n| " + " | ".join(str(c or "") for c in header) + " |\n"
            md += "| " + " | ".join("---" for _ in header) + " |\n"
            for row in rows:
                md += "| " + " | ".join(str(c or "") for c in row) + " |\n"
            markdown_parts.append(md)

        return "\n".join(markdown_parts) if markdown_parts else f"No tables found on page {page} of '{filename}'."


def ensure_pdf_dir() -> str:
    os.makedirs(PDF_DIR, exist_ok=True)
    return PDF_DIR


def initialize_index() -> None:
    """Initialize ChromaDB and rebuild BM25 from any persisted documents."""
    global vector_store, bm25_index
    _init_vector_store()
    _sync_documents_from_chroma()
    if indexed_documents:
        build_bm25_index(indexed_documents)
    else:
        bm25_index = None


initialize_index()
