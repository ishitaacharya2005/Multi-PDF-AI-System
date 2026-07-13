from __future__ import annotations

import unittest

from langchain_core.documents import Document

from config import OLLAMA_MODEL
from src.prompts.prompt_templates import format_citations, format_docs


class AppStructureTests(unittest.TestCase):
    def test_config_loads(self) -> None:
        self.assertTrue(OLLAMA_MODEL)

    def test_prompt_formatters_work(self) -> None:
        docs = [Document(page_content="hello world", metadata={"source": "file.pdf", "page": 1})]
        self.assertIn("file.pdf", format_docs(docs))
        self.assertIn("Sources:", format_citations(docs))


if __name__ == "__main__":
    unittest.main()