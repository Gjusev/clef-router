"""clef_router.compactor: context compaction scored by Clef.

Example::

    from clef_router import ClefRouter
    from clef_router.compactor import compact

    result = compact("invoice processing rules", documents, budget=2048)
    print(result.stats.savings_pct)

The optional ``clef_router.compactor.langchain`` and
``clef_router.compactor.llamaindex`` modules expose framework adapters;
their framework packages are optional extras (``pip install
clef-router[compactor]``).
"""

from .core import (
    RELEVANCE_CRITERIA,
    CompactResult,
    CompactStats,
    CutDoc,
    ScoredDoc,
    compact,
    default_token_counter,
    relevance_questions,
)

__all__ = [
    "RELEVANCE_CRITERIA",
    "CompactResult",
    "CompactStats",
    "CutDoc",
    "ScoredDoc",
    "compact",
    "default_token_counter",
    "relevance_questions",
]
