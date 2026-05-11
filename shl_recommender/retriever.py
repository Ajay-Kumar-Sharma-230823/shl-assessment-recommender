"""
retriever.py — Vector Search + Catalog Lookup
===============================================
Retriever class backed by FAISS with sentence-transformers.
Supports:
- Semantic search (primary)
- Name-based exact/fuzzy lookup (for compare queries)
- Full catalog access
"""
from __future__ import annotations

import json
import logging
import pickle
import re
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class Retriever:
    """
    FAISS-backed retriever for SHL catalog items.
    Loads index from disk once at startup — no re-embedding needed.
    """

    def __init__(
        self,
        catalog_path: str = "./catalog.json",
        index_path: str = "./faiss_index",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self._catalog: list[dict] = []
        self._index = None
        self._model = None
        self._embedding_model_name = embedding_model
        self._loaded = False

        self._load(catalog_path, index_path)

    def _load(self, catalog_path: str, index_path: str) -> None:
        """Load catalog + FAISS index from disk."""
        # 1. Load catalog JSON
        cat_file = Path(catalog_path)
        if cat_file.exists():
            try:
                with open(cat_file, encoding="utf-8") as f:
                    self._catalog = json.load(f)
                logger.info(f"Loaded catalog: {len(self._catalog)} assessments from {catalog_path}")
            except Exception as e:
                logger.error(f"Failed to load catalog: {e}")

        # 2. Load FAISS index
        idx_dir = Path(index_path)
        faiss_file = idx_dir / "index.faiss"
        pkl_file = idx_dir / "index.pkl"

        if faiss_file.exists():
            try:
                import faiss
                self._index = faiss.read_index(str(faiss_file))
                logger.info(f"Loaded FAISS index: {self._index.ntotal} vectors")

                # Prefer pkl catalog (used for index) over json if different
                if pkl_file.exists():
                    with open(pkl_file, "rb") as f:
                        indexed_catalog = pickle.load(f)
                    # Use indexed catalog if it aligns with the index size
                    if len(indexed_catalog) == self._index.ntotal:
                        self._catalog = indexed_catalog

                self._loaded = True
            except Exception as e:
                logger.error(f"Failed to load FAISS index: {e}")

        # 3. Load embedding model (lazy — only when searching)
        # Model loaded on first search call to avoid startup delay

        if not self._loaded:
            logger.warning(
                "FAISS index not available — will fall back to keyword search only"
            )

    def _get_model(self):
        """Lazy-load the sentence transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading embedding model: {self._embedding_model_name}")
                self._model = SentenceTransformer(self._embedding_model_name)
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
        return self._model

    def search(self, query: str, top_k: int = 15) -> list[dict]:
        """
        Semantic search via FAISS.
        Returns list of catalog item dicts (with added 'score' and 'rank').
        Falls back to keyword search if FAISS unavailable.
        """
        if not query.strip():
            return self.get_all()[:top_k]

        # --- FAISS semantic search ---
        if self._loaded and self._index is not None:
            try:
                model = self._get_model()
                if model:
                    embedding = model.encode(
                        [query], normalize_embeddings=True
                    )
                    embedding = np.array(embedding, dtype=np.float32)

                    k = min(top_k, len(self._catalog))
                    scores, indices = self._index.search(embedding, k)

                    results = []
                    for rank, (score, idx) in enumerate(
                        zip(scores[0], indices[0]), start=1
                    ):
                        if 0 <= idx < len(self._catalog):
                            item = dict(self._catalog[idx])
                            item["_score"] = float(score)
                            item["_rank"] = rank
                            results.append(item)

                    # Wrap in the format expected by agent
                    return [
                        {"assessment": r, "score": r["_score"], "rank": r["_rank"]}
                        for r in results
                    ]
            except Exception as e:
                logger.error(f"FAISS search failed: {e}")

        # --- Keyword fallback ---
        return self._keyword_search(query, top_k)

    def _keyword_search(self, query: str, top_k: int) -> list[dict]:
        """Simple keyword-based search as fallback."""
        query_words = set(re.findall(r"\b[a-z]{3,}\b", query.lower()))
        scored = []
        for item in self._catalog:
            text = (item.get("raw_text", "") + " " + item.get("name", "")).lower()
            score = sum(1 for w in query_words if w in text)
            if score > 0:
                scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for rank, (score, item) in enumerate(scored[:top_k], start=1):
            results.append({
                "assessment": dict(item),
                "score": min(score / max(len(query_words), 1), 1.0),
                "rank": rank,
            })
        return results

    def get_by_name(self, name: str) -> Optional[dict]:
        """
        Exact or fuzzy name lookup — used for comparison queries.
        Returns the catalog dict or None if not found.
        """
        name_lower = name.lower().strip()

        # Exact match
        for item in self._catalog:
            if item.get("name", "").lower() == name_lower:
                return item

        # Partial/fuzzy match
        for item in self._catalog:
            item_name = item.get("name", "").lower()
            if name_lower in item_name or item_name in name_lower:
                return item

        # Token overlap match
        name_tokens = set(name_lower.split())
        best_item, best_score = None, 0
        for item in self._catalog:
            item_tokens = set(item.get("name", "").lower().split())
            overlap = len(name_tokens & item_tokens)
            if overlap > best_score:
                best_score = overlap
                best_item = item

        return best_item if best_score >= 2 else None

    def get_all(self) -> list[dict]:
        """Return the full catalog."""
        return list(self._catalog)

    @property
    def catalog_size(self) -> int:
        return len(self._catalog)

    @property
    def is_ready(self) -> bool:
        return len(self._catalog) > 0


# ============================================================
# Module-level singleton
# ============================================================
_retriever: Optional[Retriever] = None


def get_retriever(
    catalog_path: str = "./catalog.json",
    index_path: str = "./faiss_index",
) -> Retriever:
    """Get or create the singleton Retriever."""
    global _retriever
    if _retriever is None:
        _retriever = Retriever(catalog_path=catalog_path, index_path=index_path)
    return _retriever
