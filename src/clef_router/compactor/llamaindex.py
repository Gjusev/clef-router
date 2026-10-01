"""LlamaIndex adapter: a node postprocessor backed by Clef relevance scores.

``ClefNodePostprocessor`` follows the ``BaseNodePostprocessor`` contract
(``_postprocess_nodes`` returning the surviving nodes) so it slots into a
``QueryBundle`` pipeline. llama-index-core is optional: without it the
class subclasses a small local base and works with any object exposing
``get_content()``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..client import ClefRouter
from .core import compact, default_token_counter

__all__ = ["ClefNodePostprocessor"]

try:  # pragma: no cover - exercised only when llama-index-core is installed
    from llama_index.core.postprocessor.types import BaseNodePostprocessor
    from llama_index.core.schema import NodeWithScore, QueryBundle
except ImportError:  # pragma: no cover

    class QueryBundle:  # type: ignore[no-redef]
        """Minimal local stand-in mirroring ``llama_index.core.schema.QueryBundle``."""

        def __init__(self, query_str: str):
            self.query_str = query_str

    class NodeWithScore:  # type: ignore[no-redef]
        """Minimal local stand-in mirroring the LlamaIndex container."""

        def __init__(self, node: Any, score: float | None = None):
            self.node = node
            self.score = score

    class BaseNodePostprocessor:  # type: ignore[no-redef]
        """Minimal local stand-in mirroring the LlamaIndex contract."""

        def _postprocess_nodes(
            self, nodes: Sequence[NodeWithScore], query_bundle: QueryBundle | None
        ) -> Sequence[NodeWithScore]:
            raise NotImplementedError


class ClefNodePostprocessor(BaseNodePostprocessor):
    """Filter retrieved nodes to the relevance budget decided by Clef.

    Attributes:
        budget: Maximum tokens across kept nodes.
        min_score: Minimum relevance score (0..3) to keep a node.
        token_counter: Token estimate; defaults to a word count.
    """

    def __init__(
        self,
        budget: int = 2048,
        min_score: float = 1.0,
        token_counter: Any = default_token_counter,
    ) -> None:
        self.budget = budget
        self.min_score = min_score
        self.token_counter = token_counter

    def _postprocess_nodes(
        self,
        nodes: Sequence[NodeWithScore],
        query_bundle: QueryBundle | None,
        router: ClefRouter | None = None,
    ) -> Sequence[NodeWithScore]:
        """Keep the nodes Clef scores relevant enough to fit the budget."""
        if query_bundle is None:
            raise ValueError("ClefNodePostprocessor needs a query_bundle")
        texts = [
            (
                node.node.get_content()
                if hasattr(node.node, "get_content")
                else str(node.node)
            )
            for node in nodes
        ]
        result = compact(
            query_bundle.query_str,
            texts,
            self.budget,
            router=router,
            min_score=self.min_score,
            token_counter=self.token_counter,
        )
        kept_indexes = {kept.index for kept in result.kept}
        return [node for position, node in enumerate(nodes) if position in kept_indexes]

    async def apostprocess_nodes(
        self,
        nodes: Sequence[NodeWithScore],
        query_bundle: QueryBundle | None,
        router: ClefRouter | None = None,
    ) -> Sequence[NodeWithScore]:
        """Async mirror of :meth:`_postprocess_nodes` (sync execution)."""
        return self._postprocess_nodes(nodes, query_bundle, router=router)
