"""
Retrieval Engine
=================
Combines semantic search with metadata filtering and re-ranking.
Implements hybrid search for improved precision.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from app.retrieval.vector_store import BaseVectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievalFilter:
    """Filters to apply during retrieval."""
    remote_testing: Optional[bool] = None
    adaptive: Optional[bool] = None
    test_types: list[str] = field(default_factory=list)
    max_duration_minutes: Optional[int] = None
    min_duration_minutes: Optional[int] = None
    keywords: list[str] = field(default_factory=list)


class RetrievalEngine:
    """
    Advanced retrieval engine with:
    - Semantic similarity search
    - Metadata filtering
    - Keyword re-ranking (hybrid)
    - Confidence scoring
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        top_k: int = 10,
        similarity_threshold: float = 0.3,
    ):
        self.vector_store = vector_store
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

    def retrieve(
        self,
        query: str,
        filters: Optional[RetrievalFilter] = None,
        top_k: Optional[int] = None,
    ) -> list[dict]:
        """
        Retrieve relevant assessments for a query.
        
        Args:
            query: Natural language query
            filters: Optional metadata filters
            top_k: Override default top_k
            
        Returns:
            List of result dicts with assessment + score
        """
        k = top_k or self.top_k
        # Fetch more than needed to allow for filtering
        fetch_k = min(k * 3, 50)

        # Step 1: Semantic search
        raw_results = self.vector_store.search(query, top_k=fetch_k)

        # Step 2: Apply filters
        if filters:
            raw_results = self._apply_filters(raw_results, filters)

        # Step 3: Hybrid re-ranking with keyword boost
        raw_results = self._rerank(raw_results, query)

        # Step 4: Apply similarity threshold
        raw_results = [
            r for r in raw_results if r["score"] >= self.similarity_threshold
        ]

        # Step 5: Trim to top_k
        raw_results = raw_results[:k]

        # Update ranks
        for i, r in enumerate(raw_results, start=1):
            r["rank"] = i

        logger.info(
            f"Retrieved {len(raw_results)} results for query: '{query[:60]}...'"
            if len(query) > 60
            else f"Retrieved {len(raw_results)} results for query: '{query}'"
        )

        return raw_results

    def _apply_filters(self, results: list[dict], filters: RetrievalFilter) -> list[dict]:
        """Apply metadata filters to results."""
        filtered = []
        for r in results:
            a = r["assessment"]

            # Remote testing filter
            if filters.remote_testing is not None:
                if bool(a.get("remote_testing")) != filters.remote_testing:
                    continue

            # Adaptive filter
            if filters.adaptive is not None:
                if bool(a.get("adaptive")) != filters.adaptive:
                    continue

            # Test type filter
            if filters.test_types:
                a_type = a.get("test_type", "").lower()
                if not any(t.lower() in a_type or a_type in t.lower() for t in filters.test_types):
                    continue

            # Duration filter
            duration_mins = a.get("duration_minutes")
            if filters.max_duration_minutes and duration_mins:
                if duration_mins > filters.max_duration_minutes:
                    continue
            if filters.min_duration_minutes and duration_mins:
                if duration_mins < filters.min_duration_minutes:
                    continue

            filtered.append(r)

        logger.debug(f"Filter reduced results from {len(results)} to {len(filtered)}")
        return filtered

    def _rerank(self, results: list[dict], query: str) -> list[dict]:
        """
        Hybrid re-ranking: boost results where query terms appear in assessment text.
        Implements a simple BM25-inspired keyword boost.
        """
        query_words = set(re.findall(r"\b[a-z]{3,}\b", query.lower()))
        stop_words = {"the", "and", "for", "are", "with", "that", "this", "from", "have"}
        query_words -= stop_words

        if not query_words:
            return results

        for r in results:
            a = r["assessment"]
            raw_text = (a.get("raw_text", "") or "").lower()
            name = (a.get("name", "") or "").lower()

            # Count keyword hits
            keyword_hits = sum(1 for w in query_words if w in raw_text)
            name_hits = sum(1 for w in query_words if w in name)

            # Boost: 0.05 per keyword hit, 0.1 per name hit (capped at 0.3)
            keyword_boost = min(keyword_hits * 0.05 + name_hits * 0.1, 0.3)
            r["score"] = min(r["score"] + keyword_boost, 1.0)
            r["keyword_hits"] = keyword_hits

        # Re-sort by updated score
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def retrieve_for_comparison(
        self, assessment_names: list[str]
    ) -> list[dict]:
        """
        Retrieve specific assessments by name for comparison queries.
        """
        results = []
        for name in assessment_names:
            # Search specifically for this assessment
            candidates = self.vector_store.search(name, top_k=5)
            for c in candidates:
                a_name = c["assessment"].get("name", "").lower()
                if name.lower() in a_name or a_name in name.lower():
                    results.append(c)
                    break
            else:
                # Take best match
                if candidates:
                    results.append(candidates[0])

        return results

    def get_all_assessments(self) -> list[dict]:
        """Return all assessments in the store."""
        return self.vector_store._assessments if hasattr(self.vector_store, "_assessments") else []


def parse_filters_from_context(context: str) -> RetrievalFilter:
    """
    Parse retrieval filters from conversation context/requirements.
    
    Detects:
    - remote testing requirements
    - adaptive testing preferences
    - test type preferences
    - duration constraints
    """
    context_lower = context.lower()
    filters = RetrievalFilter()

    # Remote testing
    if any(kw in context_lower for kw in ["remote", "online", "virtual", "wfh", "work from home"]):
        filters.remote_testing = True

    # Adaptive testing
    if any(kw in context_lower for kw in ["adaptive", "cat ", "computer adaptive"]):
        filters.adaptive = True

    # Test types
    type_keywords = {
        "Cognitive Ability": ["cognitive", "reasoning", "numerical", "verbal", "iq", "aptitude", "ability"],
        "Personality": ["personality", "opq", "behaviour", "behavior", "character", "traits"],
        "Skills": ["skills", "coding", "programming", "technical", "microsoft", "excel", "java", "python"],
        "Situational Judgement": ["situational", "sjt", "scenario", "judgement"],
        "Language": ["language", "english", "communication", "grammar"],
        "Competency": ["competency", "leadership", "management", "executive"],
    }
    for t_type, keywords in type_keywords.items():
        if any(kw in context_lower for kw in keywords):
            filters.test_types.append(t_type)

    # Duration
    duration_match = re.search(r"(\d+)\s*minute", context_lower)
    if duration_match:
        minutes = int(duration_match.group(1))
        # Interpret as max duration
        filters.max_duration_minutes = minutes

    return filters
