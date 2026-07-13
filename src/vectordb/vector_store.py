from __future__ import annotations

import os
import re
from typing import Optional

import pdfplumber
from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from config import BM25_TOP_K, CHROMA_DB_PATH, RRF_K, TOP_K
from src.chunking.chunker import chunk_text
from src.embeddings.embedder import get_embeddings
from src.utils.helpers import ensure_directory

COLLECTION_NAME = "pdf_documents"
PDF_DIR = os.path.join(os.path.dirname(CHROMA_DB_PATH) or ".", "pdfs")

vector_store: Optional[Chroma] = None
bm25_index: Optional[BM25Okapi] = None
indexed_documents: list[Document] = []
last_retrieval_sources: list[dict] = []


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def doc_key(doc: Document) -> str:
    meta = doc.metadata or {}
    return f"{meta.get('source', '')}|{meta.get('page', '')}|{meta.get('chunk', '')}|{doc.page_content[:80]}"


def init_vector_store() -> Chroma:
    global vector_store
    ensure_directory(CHROMA_DB_PATH)
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DB_PATH,
        embedding_function=get_embeddings(),
    )
    return vector_store


def ensure_pdf_dir() -> str:
    return ensure_directory(PDF_DIR)


def load_pdfs(pdf_paths: list[str]) -> list[Document]:
    documents: list[Document] = []
    for pdf_path in pdf_paths:
        filename = os.path.basename(pdf_path)
        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                if not text.strip():
                    continue
                for chunk_idx, chunk in enumerate(chunk_text(text)):
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


def build_bm25_index(documents: list[Document]) -> BM25Okapi:
    global bm25_index, indexed_documents
    indexed_documents = list(documents)
    tokenized_corpus = [tokenize(doc.page_content) for doc in indexed_documents]
    bm25_index = BM25Okapi(tokenized_corpus)
    return bm25_index


def sync_documents_from_chroma() -> None:
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


def refresh_indexes() -> None:
    global bm25_index
    if indexed_documents:
        build_bm25_index(indexed_documents)
    else:
        bm25_index = None


def store_documents(new_docs: list[Document]) -> int:
    global vector_store

    if not new_docs:
        return count_indexed_pdfs()

    if vector_store is None:
        init_vector_store()

    ids = [
        f"{doc.metadata['source']}_{doc.metadata['page']}_{doc.metadata['chunk']}"
        for doc in new_docs
    ]
    vector_store.add_documents(documents=new_docs, ids=ids)
    sync_documents_from_chroma()
    refresh_indexes()
    return count_indexed_pdfs()


def count_indexed_pdfs() -> int:
    sources = {doc.metadata.get("source") for doc in indexed_documents if doc.metadata.get("source")}
    return len(sources)


def get_documents_by_source(filename: str) -> list[Document]:
    return [doc for doc in indexed_documents if doc.metadata.get("source") == filename]


def resolve_pdf_path(filename: str) -> Optional[str]:
    path = os.path.join(PDF_DIR, filename)
    return path if os.path.isfile(path) else None


def extract_tables(filename: str, page: int) -> str:
    pdf_path = resolve_pdf_path(filename)
    if not pdf_path:
        return f"PDF file '{filename}' not found in {PDF_DIR}."

    with pdfplumber.open(pdf_path) as pdf:
        if page < 1 or page > len(pdf.pages):
            return f"Page {page} is out of range for '{filename}' (1-{len(pdf.pages)})."

        tables = pdf.pages[page - 1].extract_tables()
        if not tables:
            return f"No tables found on page {page} of '{filename}'."

        markdown_parts: list[str] = []
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


def ingest_pdfs(pdf_paths: list[str]) -> int:
    ensure_pdf_dir()
    return store_documents(load_pdfs(pdf_paths))


def hybrid_search(query: str, top_k: int = TOP_K) -> list[Document]:
    global last_retrieval_sources

    if not indexed_documents or bm25_index is None or vector_store is None:
        last_retrieval_sources = []
        return []

    tokenized_query = tokenize(query)
    bm25_scores = bm25_index.get_scores(tokenized_query)
    bm25_ranked_indices = sorted(
        range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
    )[:BM25_TOP_K]

    chroma_results = vector_store.similarity_search(query, k=BM25_TOP_K)

    rrf_scores: dict[str, float] = {}
    doc_by_key: dict[str, Document] = {}

    for rank, idx in enumerate(bm25_ranked_indices):
        doc = indexed_documents[idx]
        key = doc_key(doc)
        doc_by_key[key] = doc
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)

    for rank, doc in enumerate(chroma_results):
        key = doc_key(doc)
        doc_by_key[key] = doc
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)

    ranked_keys = sorted(rrf_scores.keys(), key=lambda key: rrf_scores[key], reverse=True)
    results = [doc_by_key[key] for key in ranked_keys[:top_k]]

    last_retrieval_sources = [
        {"source": doc.metadata.get("source", "unknown"), "page": doc.metadata.get("page", "?")}
        for doc in results
    ]
    return results


def initialize_index() -> None:
    global bm25_index
    init_vector_store()
    sync_documents_from_chroma()
    if indexed_documents:
        build_bm25_index(indexed_documents)
    else:
        bm25_index = None


initialize_index()