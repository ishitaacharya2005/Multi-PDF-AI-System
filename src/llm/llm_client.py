from __future__ import annotations

from langchain_ollama import ChatOllama

from config import OLLAMA_MODEL

llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)