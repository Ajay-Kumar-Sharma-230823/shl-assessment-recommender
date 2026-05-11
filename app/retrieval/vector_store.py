"""
Vector Store Manager
=====================
Manages FAISS and ChromaDB vector stores for SHL assessment retrieval.
Supports:
- Index creation from catalog
- Semantic similarity search
- Metadata filtering
- Hybrid search (semantic + keyword)
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np

from app.retrieval.embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)


# ============================================================
# Abstract Base Vector Store
# ============================================================
class BaseVectorStore(ABC):
    """Abstract vector store interface."""

    @abstractmethod
    def build_index(self, assessments: list[dict]) -> None:
        """Build the index from a list of assessment dicts."""
        ...

    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Search for similar assessments. Returns list of (assessment, score) dicts."""
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the index to disk."""
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """Load the index from disk."""
        ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Return True if the index is ready for search."""
        ...


# ============================================================
# FAISS Vector Store
# ============================================================
class FAISSVectorStore(BaseVectorStore):
    """
    FAISS-based vector store with flat inner product index.
    Uses normalized embeddings so IP = cosine similarity.
    """

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedding_gen = EmbeddingGenerator(embedding_model)
        self._index = None
        self._assessments: list[dict] = []
        self._embeddings: Optional[np.ndarray] = None

    def build_index(self, assessments: list[dict]) -> None:
        """Build FAISS index from assessment data."""
        import faiss

        self._assessments = assessments
        texts = [a.get("raw_text", a.get("name", "")) for a in assessments]

        logger.info(f"Building FAISS index for {len(texts)} assessments...")
        embeddings = self.embedding_gen.embed(texts)
        self._embeddings = embeddings

        dim = embeddings.shape[1]
        # Inner product index (works as cosine with normalized vectors)
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings.astype(np.float32))

        logger.info(f"FAISS index built: {self._index.ntotal} vectors, dim={dim}")

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Semantic search using FAISS."""
        if not self.is_loaded:
            raise RuntimeError("FAISS index not loaded. Call build_index() or load() first.")

        query_embedding = self.embedding_gen.embed_single(query).astype(np.float32)
        query_embedding = query_embedding.reshape(1, -1)

        scores, indices = self._index.search(query_embedding, min(top_k, len(self._assessments)))

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx >= 0 and idx < len(self._assessments):
                results.append({
                    "assessment": self._assessments[idx],
                    "score": float(score),
                    "rank": rank,
                })

        return results

    def save(self, path: str) -> None:
        """Save FAISS index and metadata."""
        import faiss

        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(save_dir / "index.faiss"))

        with open(save_dir / "assessments.json", "w", encoding="utf-8") as f:
            json.dump(self._assessments, f, indent=2, ensure_ascii=False)

        np.save(str(save_dir / "embeddings.npy"), self._embeddings)

        logger.info(f"FAISS index saved to {path}")

    def load(self, path: str) -> None:
        """Load FAISS index and metadata."""
        import faiss

        save_dir = Path(path)

        index_file = save_dir / "index.faiss"
        catalog_file = save_dir / "assessments.json"

        if not index_file.exists():
            raise FileNotFoundError(f"FAISS index not found at {path}")

        self._index = faiss.read_index(str(index_file))

        with open(catalog_file, encoding="utf-8") as f:
            self._assessments = json.load(f)

        emb_file = save_dir / "embeddings.npy"
        if emb_file.exists():
            self._embeddings = np.load(str(emb_file))

        logger.info(f"FAISS index loaded: {self._index.ntotal} vectors")

    @property
    def is_loaded(self) -> bool:
        return self._index is not None and len(self._assessments) > 0


# ============================================================
# ChromaDB Vector Store
# ============================================================
class ChromaVectorStore(BaseVectorStore):
    """ChromaDB-based vector store."""

    COLLECTION_NAME = "shl_assessments"

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedding_gen = EmbeddingGenerator(embedding_model)
        self._client = None
        self._collection = None
        self._assessments: list[dict] = []

    def _get_client(self, path: str = "./vectorstore/chroma"):
        """Get or create ChromaDB client."""
        import chromadb

        Path(path).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=path)
        return self._client

    def build_index(self, assessments: list[dict], persist_path: str = "./vectorstore/chroma") -> None:
        """Build ChromaDB collection."""
        self._assessments = assessments
        client = self._get_client(persist_path)

        # Delete existing collection if exists
        try:
            client.delete_collection(self.COLLECTION_NAME)
        except Exception:
            pass

        self._collection = client.create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        texts = [a.get("raw_text", a.get("name", "")) for a in assessments]
        embeddings = self.embedding_gen.embed(texts)

        ids = [str(i) for i in range(len(assessments))]
        metadatas = [
            {
                "name": a.get("name", ""),
                "url": a.get("url", ""),
                "test_type": a.get("test_type", ""),
                "remote_testing": str(a.get("remote_testing", False)),
                "adaptive": str(a.get("adaptive", False)),
            }
            for a in assessments
        ]

        self._collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
        )

        logger.info(f"ChromaDB collection built with {len(assessments)} vectors")

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Semantic search using ChromaDB."""
        if not self.is_loaded:
            raise RuntimeError("ChromaDB collection not loaded.")

        query_embedding = self.embedding_gen.embed_single(query).tolist()
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, len(self._assessments)),
            include=["metadatas", "distances", "documents"],
        )

        output = []
        for rank, (idx, distance) in enumerate(
            zip(results["ids"][0], results["distances"][0]), start=1
        ):
            assessment_idx = int(idx)
            score = 1.0 - distance  # Convert distance to similarity
            if assessment_idx < len(self._assessments):
                output.append({
                    "assessment": self._assessments[assessment_idx],
                    "score": float(score),
                    "rank": rank,
                })

        return output

    def save(self, path: str) -> None:
        """ChromaDB auto-persists. Save assessment metadata."""
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "assessments.json", "w", encoding="utf-8") as f:
            json.dump(self._assessments, f, indent=2)
        logger.info(f"ChromaDB metadata saved to {path}")

    def load(self, path: str) -> None:
        """Load ChromaDB collection."""
        import chromadb

        self._client = chromadb.PersistentClient(path=str(Path(path) / "chroma"))
        try:
            self._collection = self._client.get_collection(self.COLLECTION_NAME)
        except Exception as e:
            raise RuntimeError(f"ChromaDB collection not found: {e}")

        catalog_file = Path(path) / "assessments.json"
        if catalog_file.exists():
            with open(catalog_file, encoding="utf-8") as f:
                self._assessments = json.load(f)

        logger.info(f"ChromaDB loaded: {self._collection.count()} vectors")

    @property
    def is_loaded(self) -> bool:
        return self._collection is not None


# ============================================================
# Vector Store Factory
# ============================================================
def get_vector_store(store_type: str = "faiss", embedding_model: str = "all-MiniLM-L6-v2") -> BaseVectorStore:
    """Factory function to create the appropriate vector store."""
    if store_type.lower() == "faiss":
        return FAISSVectorStore(embedding_model)
    elif store_type.lower() == "chroma":
        return ChromaVectorStore(embedding_model)
    else:
        raise ValueError(f"Unknown vector store type: {store_type}. Use 'faiss' or 'chroma'.")


# ============================================================
# Index Builder Script
# ============================================================
def build_vector_index(
    catalog_path: str = "./data/shl_catalog.json",
    store_type: str = "faiss",
    output_path: str = "./vectorstore",
    embedding_model: str = "all-MiniLM-L6-v2",
) -> BaseVectorStore:
    """Build and save vector index from catalog."""
    catalog_file = Path(catalog_path)
    if not catalog_file.exists():
        raise FileNotFoundError(f"Catalog not found: {catalog_path}")

    with open(catalog_file, encoding="utf-8") as f:
        assessments = json.load(f)

    logger.info(f"Building {store_type.upper()} index for {len(assessments)} assessments...")

    store = get_vector_store(store_type, embedding_model)
    store.build_index(assessments)
    store.save(output_path)

    logger.info(f"Vector index built and saved to {output_path}")
    return store


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    store = build_vector_index()
    
    # Test search
    results = store.search("cognitive ability test for software engineer", top_k=3)
    for r in results:
        print(f"[{r['rank']}] {r['assessment']['name']} (score={r['score']:.3f})")
