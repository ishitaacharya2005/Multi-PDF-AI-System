import os

import streamlit as st

from config import OLLAMA_MODEL
from src.api.routes import create_agent_executor
from src.llm.llm_client import llm
from src.prompts.prompt_templates import format_docs
from src.vectordb.vector_store import (
    PDF_DIR,
    count_indexed_pdfs,
    ensure_pdf_dir,
    extract_tables,
    get_documents_by_source,
    indexed_documents,
    ingest_pdfs,
    last_retrieval_sources,
)

st.set_page_config(page_title="Multi-PDF AI RAG", page_icon="📄", layout="wide")

st.title("Multi-PDF AI RAG System")
st.caption(f"Powered by Ollama ({OLLAMA_MODEL}) + hybrid BM25/Chroma search")


def _save_uploaded_pdfs(uploaded_files) -> list[str]:
    ensure_pdf_dir()
    saved_paths = []
    for uploaded in uploaded_files:
        path = os.path.join(PDF_DIR, uploaded.name)
        with open(path, "wb") as f:
            f.write(uploaded.getbuffer())
        saved_paths.append(path)
    return saved_paths


def _unique_sources(sources: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for src in sources:
        key = (src.get("source"), src.get("page"))
        if key not in seen:
            seen.add(key)
            unique.append(src)
    return unique


def _render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander("Sources", expanded=False):
        for src in _unique_sources(sources):
            st.markdown(f"- **{src.get('source', 'unknown')}** — page {src.get('page', '?')}")


if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = create_agent_executor()


with st.sidebar:
    st.header("PDF Library")
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button("Index uploaded PDFs", type="primary"):
            with st.spinner("Extracting text and building indexes..."):
                paths = _save_uploaded_pdfs(uploaded_files)
                ingest_pdfs(paths)
            st.success(f"Indexed {len(uploaded_files)} file(s).")

    st.metric("PDFs indexed", count_indexed_pdfs())

    st.divider()
    st.subheader("Quick actions")

    pdf_names = sorted(
        {doc.metadata.get("source") for doc in indexed_documents if doc.metadata.get("source")}
    )

    summarize_target = st.selectbox(
        "Summarize PDF",
        options=[""] + pdf_names,
        format_func=lambda x: "Select a PDF..." if x == "" else x,
    )
    if st.button("Run summary") and summarize_target:
        with st.spinner(f"Summarizing {summarize_target}..."):
            docs = get_documents_by_source(summarize_target)
            context = format_docs(docs)
            summary = llm.invoke(
                f"Provide a comprehensive summary of this document:\n\n{context}"
            ).content
        st.session_state.messages.append({"role": "user", "content": f"Summarize {summarize_target}"})
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": summary,
                "sources": [
                    {"source": summarize_target, "page": doc.metadata.get("page")}
                    for doc in docs[:5]
                ],
            }
        )
        st.rerun()

    table_pdf = st.selectbox(
        "Extract table from",
        options=[""] + pdf_names,
        format_func=lambda x: "Select a PDF..." if x == "" else x,
        key="table_pdf",
    )
    table_page = st.number_input("Page number", min_value=1, value=1, step=1)
    if st.button("Extract table") and table_pdf:
        table_md = extract_tables(table_pdf, int(table_page))
        st.session_state.messages.append(
            {"role": "user", "content": f"Extract tables from {table_pdf}, page {table_page}"}
        )
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": table_md,
                "sources": [{"source": table_pdf, "page": int(table_page)}],
            }
        )
        st.rerun()


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            _render_sources(message.get("sources", []))

if prompt := st.chat_input("Ask a question about your PDFs..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = st.session_state.agent_executor.invoke({"input": prompt})
            response = result["output"]
        st.markdown(response)
        sources = list(last_retrieval_sources)
        _render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": response, "sources": sources}
    )
