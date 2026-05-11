"""
Embedding Generator
====================
Generates embeddings for SHL assessments using Sentence Transformers.
Supports batch processing and caching.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generates semantic embeddings using Sentence Transformers.
    Uses all-MiniLM-L6-v2 by default (fast, high quality, 384-dim).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        logger.info(f"EmbeddingGenerator initialized with model: {model_name}")

    def _load_model(self):
        """Lazy load the model on first use."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")

    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            batch_size: Processing batch size
            
        Returns:
            numpy array of shape (n, embedding_dim)
        """
        self._load_model()

        if not texts:
            return np.array([])

        logger.info(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 50,
            normalize_embeddings=True,  # Normalize for cosine similarity
            convert_to_numpy=True,
        )
        logger.info(f"Generated embeddings shape: {embeddings.shape}")
        return embeddings

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        return self.embed([text])[0]

    @property
    def embedding_dim(self) -> int:
        """Return the dimension of the embedding vectors."""
        self._load_model()
        return self._model.get_sentence_embedding_dimension()


def generate_embeddings_for_catalog(
    catalog_path: str = "./data/shl_catalog.json",
    output_path: str = "./data/embeddings.npy",
    model_name: str = "all-MiniLM-L6-v2",
) -> tuple[np.ndarray, list[dict]]:
    """
    Generate embeddings for all assessments in the catalog.
    
    Returns:
        (embeddings array, list of assessment dicts)
    """
    catalog_file = Path(catalog_path)
    if not catalog_file.exists():
        raise FileNotFoundError(f"Catalog not found: {catalog_path}")

    with open(catalog_file, encoding="utf-8") as f:
        assessments = json.load(f)

    logger.info(f"Loaded {len(assessments)} assessments for embedding")

    # Build texts from raw_text field (pre-computed during cleaning)
    texts = []
    for a in assessments:
        raw_text = a.get("raw_text", "")
        if not raw_text:
            # Fallback: build from fields
            parts = [
                a.get("name", ""),
                a.get("description", ""),
                a.get("test_type", ""),
                " ".join(a.get("skills_measured", [])),
            ]
            raw_text = " | ".join(p for p in parts if p)
        texts.append(raw_text)

    generator = EmbeddingGenerator(model_name)
    embeddings = generator.embed(texts)

    # Save embeddings
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_file), embeddings)
    logger.info(f"Saved embeddings to {output_path}")

    return embeddings, assessments


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    embs, catalog = generate_embeddings_for_catalog()
    print(f"✅ Generated {embs.shape[0]} embeddings of dimension {embs.shape[1]}")
