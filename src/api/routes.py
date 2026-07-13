from __future__ import annotations

import json
import os

try:
    from langchain.agents import AgentExecutor, create_react_agent
except ImportError:
    from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.tools import tool

from src.llm.llm_client import llm
from src.prompts.prompt_templates import REACT_PROMPT, format_citations, format_docs
from src.retrieval.retriever import get_documents_by_source, hybrid_search
from src.vectordb.vector_store import (
    PDF_DIR,
    count_indexed_pdfs,
    ensure_pdf_dir,
    extract_tables as extract_pdf_tables,
    ingest_pdfs,
)


def _answer_from_docs(question: str, docs) -> str:
    context = format_docs(docs)
    prompt = f"""Answer the question using only the context below. Be concise and accurate.

Context:
{context}

Question: {question}

Answer:"""
    return llm.invoke(prompt).content


@tool
def query_pdfs(query: str) -> str:
    """Search indexed PDF documents, answer the query, and include source citations."""
    docs = hybrid_search(query)
    if not docs:
        return "No indexed PDF content matches that query. Upload PDFs first."
    answer = _answer_from_docs(query, docs)
    return f"{answer}\n\n{format_citations(docs)}"


@tool
def summarize_pdf(filename: str) -> str:
    """Summarize all indexed chunks from a specific PDF filename."""
    docs = get_documents_by_source(filename)
    if not docs:
        return f"No indexed content found for '{filename}'. Check the filename and ensure the PDF was uploaded."
    context = format_docs(docs)
    prompt = f"""Provide a comprehensive summary of the following document content.

{context}

Summary:"""
    return llm.invoke(prompt).content


@tool
def extract_tables(filename: str, page: int) -> str:
    """Extract tables from a specific page of a PDF as markdown. Args: filename, page number."""
    return extract_pdf_tables(filename, page)


@tool
def extract_structured_data(query: str) -> str:
    """Answer a query using RAG and return the result as JSON with answer and sources fields."""
    docs = hybrid_search(query)
    if not docs:
        payload = {"answer": "No relevant documents found.", "sources": []}
        return json.dumps(payload, indent=2)

    answer = _answer_from_docs(query, docs)
    sources = [
        {"filename": doc.metadata.get("source", "unknown"), "page": doc.metadata.get("page")}
        for doc in docs
    ]
    seen = set()
    unique_sources = []
    for src in sources:
        key = (src["filename"], src["page"])
        if key not in seen:
            seen.add(key)
            unique_sources.append(src)

    payload = {"answer": answer, "sources": unique_sources}
    return json.dumps(payload, indent=2)


TOOLS = [query_pdfs, summarize_pdf, extract_tables, extract_structured_data]


def create_agent_executor() -> AgentExecutor:
    agent = create_react_agent(llm, TOOLS, REACT_PROMPT)
    return AgentExecutor(
        agent=agent,
        tools=TOOLS,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=6,
    )


def ingest_pdfs_from_directory() -> int:
    ensure_pdf_dir()
    pdf_paths = [
        os.path.join(PDF_DIR, name)
        for name in os.listdir(PDF_DIR)
        if name.lower().endswith(".pdf")
    ]
    return ingest_pdfs(pdf_paths)


def run_cli() -> None:
    agent_executor = create_agent_executor()
    ingest_pdfs_from_directory()
    print(f"Multi-PDF RAG ready — {count_indexed_pdfs()} PDF(s) indexed.")
    print("Place PDFs in ./pdfs/ or use Streamlit (python -m streamlit run app.py).\n")

    while True:
        print("\n" + "-" * 42)
        question = input("Ask your question (q to quit): ").strip()
        if question.lower() == "q":
            break
        if not question:
            continue

        result = agent_executor.invoke({"input": question})
        print("\n" + result["output"])