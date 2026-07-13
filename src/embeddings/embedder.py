from __future__ import annotations

from langchain_community.embeddings import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)