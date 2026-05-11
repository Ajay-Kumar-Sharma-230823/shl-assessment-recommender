"""
Embedding Generator
====================
Lightweight embedding using ONNX Runtime + HuggingFace tokenizer.
No PyTorch, no Rust — works on all platforms including Render free tier.
Uses all-MiniLM-L6-v2 ONNX model (384-dim, same as sentence-transformers).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

ONNX_MODEL_ID = "optimum/all-MiniLM-L6-v2"   # HF repo with ONNX weights
MODEL_FILE    = "model.onnx"


class EmbeddingGenerator:
    """
    ONNX-based embedding generator.
    Produces identical 384-dim embeddings to sentence-transformers/all-MiniLM-L6-v2
    without requiring PyTorch.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._session = None
        self._tokenizer = None
        logger.info(f"EmbeddingGenerator ready (ONNX): {model_name}")

    def _load_model(self):
        """Lazy-load ONNX session and tokenizer on first use."""
        if self._session is not None:
            return

        logger.info("Loading ONNX embedding model...")
        try:
            import onnxruntime as ort
            from transformers import AutoTokenizer
            from huggingface_hub import hf_hub_download

            # Download ONNX model weights
            model_path = hf_hub_download(
                repo_id=ONNX_MODEL_ID,
                filename=MODEL_FILE,
            )
            self._session = ort.InferenceSession(
                model_path,
                providers=["CPUExecutionProvider"],
            )
            self._tokenizer = AutoTokenizer.from_pretrained(ONNX_MODEL_ID)
            logger.info("ONNX embedding model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise

    def _mean_pool(self, token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """Mean-pool token embeddings weighted by attention mask."""
        mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
        sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
        sum_mask = np.sum(mask_expanded, axis=1)
        sum_mask = np.clip(sum_mask, a_min=1e-9, a_max=None)
        return sum_embeddings / sum_mask

    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Generate normalized embeddings for a list of texts."""
        self._load_model()
        if not texts:
            return np.array([])

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="np",
            )

            inputs = {
                "input_ids":      encoded["input_ids"].astype(np.int64),
                "attention_mask": encoded["attention_mask"].astype(np.int64),
                "token_type_ids": encoded.get("token_type_ids", np.zeros_like(encoded["input_ids"])).astype(np.int64),
            }

            outputs = self._session.run(None, inputs)
            token_embeddings = outputs[0]  # (batch, seq_len, 384)

            pooled = self._mean_pool(token_embeddings, encoded["attention_mask"])

            # L2 normalize
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            pooled = pooled / norms

            all_embeddings.append(pooled)

        result = np.vstack(all_embeddings).astype(np.float32)
        logger.info(f"Embeddings shape: {result.shape}")
        return result

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        return self.embed([text])[0]

    @property
    def embedding_dim(self) -> int:
        return 384


def generate_embeddings_for_catalog(
    catalog_path: str = "./data/shl_catalog.json",
    output_path: str = "./data/embeddings.npy",
    model_name: str = "all-MiniLM-L6-v2",
):
    import json

    catalog_file = Path(catalog_path)
    if not catalog_file.exists():
        raise FileNotFoundError(f"Catalog not found: {catalog_path}")

    with open(catalog_file, encoding="utf-8") as f:
        assessments = json.load(f)

    logger.info(f"Loaded {len(assessments)} assessments")
    texts = []
    for a in assessments:
        raw_text = a.get("raw_text", "")
        if not raw_text:
            parts = [a.get("name", ""), a.get("description", ""), a.get("test_type", ""),
                     " ".join(a.get("skills_measured", []))]
            raw_text = " | ".join(p for p in parts if p)
        texts.append(raw_text)

    generator = EmbeddingGenerator(model_name)
    embeddings = generator.embed(texts)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_file), embeddings)
    logger.info(f"Saved embeddings to {output_path}")
    return embeddings, assessments


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    embs, _ = generate_embeddings_for_catalog()
    print(f"✅ {embs.shape[0]} embeddings, dim={embs.shape[1]}")
