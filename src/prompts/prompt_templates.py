from __future__ import annotations

from langchain_core.prompts import PromptTemplate


def format_docs(docs) -> str:
    if not docs:
        return "No relevant documents found."
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        parts.append(f"[{source}, page {page}]\n{doc.page_content}")
    return "\n\n".join(parts)


def format_citations(docs) -> str:
    if not docs:
        return ""
    seen = set()
    lines = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        key = (source, page)
        if key not in seen:
            seen.add(key)
            lines.append(f"- {source} (page {page})")
    return "Sources:\n" + "\n".join(lines)


REACT_PROMPT = PromptTemplate.from_template(
    """You are a helpful assistant that answers questions about indexed PDF documents.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""
)