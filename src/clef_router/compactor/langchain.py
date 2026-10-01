"""LangChain adapter: a document compressor backed by Clef relevance scores.

``ClefDocumentCompressor`` follows the ``BaseDocumentCompressor`` contract
(``compress_documents`` returning ``Document`` objects) so it plugs into
``ContextualCompressionRetriever``. LangChain is an optional dependency:
without it installed, the class still exists but subclasses a small local
base, and constructing the real retriever is the caller's business.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..client import ClefRouter
from .core import compact, default_token_counter

__all__ = ["ClefDocumentCompressor"]

try:  # pragma: no cover - exercised only when langchain-core is installed
    from langchain_core.documents import BaseDocumentCompressor, Document
except ImportError:  # pragma: no cover

    class BaseDocumentCompressor:  # type: ignore[no-redef]
        """Minimal local stand-in mirroring the LangChain contract."""

        def compress_documents(
            self, documents: Sequence[Any], query: str, **kwargs: Any
        ) -> Sequence[Any]:
            raise NotImplementedError

    class Document:  # type: ignore[no-redef]
        """Minimal local stand-in mirroring ``langchain_core.documents.Document``."""

        def __init__(self, page_content: str, metadata: dict[str, Any] | None = None):
            self.page_content = page_content
            self.metadata = metadata or {}


class ClefDocumentCompressor(BaseDocumentCompressor):
    """Drop retrieved documents that do not fit the relevance budget.

    Attributes:
        budget: Maximum tokens across kept documents.
        min_score: Minimum relevance score (0..3) to keep a document.
        token_counter: Token estimate; defaults to a word count.
    """

    budget: int = 2048
    min_score: float = 1.0
    token_counter: Any = default_token_counter

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        router: ClefRouter | None = None,
        **kwargs: Any,
    ) -> Sequence[Document]:
        """Score *documents* against *query* and return the kept ones."""
        texts = [
            doc.page_content if hasattr(doc, "page_content") else str(doc)
            for doc in documents
        ]
        result = compact(
            query,
            texts,
            self.budget,
            router=router,
            min_score=self.min_score,
            token_counter=self.token_counter,
        )
        kept_indexes = {kept.index for kept in result.kept}
        return [
            doc
            for position, doc in enumerate(documents)
            if position in kept_indexes
        ]

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        router: ClefRouter | None = None,
        **kwargs: Any,
    ) -> Sequence[Document]:
        """Async mirror of :meth:`compress_documents` (sync execution)."""
        return self.compress_documents(documents, query, router=router)
