"""
App Startup & Dependency Injection
=====================================
Manages singleton instances of:
- Vector Store (loaded from disk)
- Retrieval Engine
- LLM Client
- Recommendation Agent

Uses FastAPI's lifespan context manager for startup/shutdown.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from app.models.config import get_settings
from app.retrieval.retrieval_engine import RetrievalEngine
from app.retrieval.vector_store import BaseVectorStore, get_vector_store
from app.services.agent import RecommendationAgent
from app.services.llm_client import create_llm_client

logger = logging.getLogger(__name__)

# Global singletons
_vector_store: Optional[BaseVectorStore] = None
_retrieval_engine: Optional[RetrievalEngine] = None
_agent: Optional[RecommendationAgent] = None


def get_or_create_vector_store() -> BaseVectorStore:
    """Load or create vector store."""
    global _vector_store
    settings = get_settings()

    if _vector_store is not None and _vector_store.is_loaded:
        return _vector_store

    store = get_vector_store(settings.vector_store_type, settings.embedding_model)
    store_path = settings.vector_store_path

    # Try to load existing index
    if Path(store_path).exists():
        try:
            logger.info(f"Loading existing {settings.vector_store_type} index from {store_path}...")
            store.load(store_path)
            logger.info("Vector store loaded successfully")
            _vector_store = store
            return store
        except Exception as e:
            logger.warning(f"Failed to load existing index: {e}. Will build new index.")

    # Build index from catalog
    catalog_path = settings.clean_data_file
    if Path(catalog_path).exists():
        import json
        logger.info(f"Building vector index from catalog: {catalog_path}")
        with open(catalog_path, encoding="utf-8") as f:
            assessments = json.load(f)

        store.build_index(assessments)
        store.save(store_path)
        _vector_store = store
        return store
    else:
        # No catalog exists yet — run pipeline
        logger.warning(f"Catalog not found at {catalog_path}. Running data pipeline...")
        _run_data_pipeline(settings)

        # Try again
        if Path(catalog_path).exists():
            import json
            with open(catalog_path, encoding="utf-8") as f:
                assessments = json.load(f)
            store.build_index(assessments)
            store.save(store_path)
            _vector_store = store
            return store
        else:
            logger.error("Failed to build catalog. Using empty store.")
            _vector_store = store
            return store


def _run_data_pipeline(settings) -> None:
    """Run the full data pipeline: scrape -> clean -> index."""
    import json
    from pathlib import Path

    logger.info("Running full data pipeline...")

    # Step 1: Scrape
    raw_file = settings.raw_data_file
    if not Path(raw_file).exists():
        logger.info("Scraping SHL catalog...")
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from scraper.shl_scraper import run_scraper
            run_scraper(raw_file)
        except Exception as e:
            logger.error(f"Scraping failed: {e}")
            return

    # Step 2: Clean
    clean_file = settings.clean_data_file
    if not Path(clean_file).exists():
        logger.info("Cleaning scraped data...")
        try:
            from scraper.cleaner import run_cleaning_pipeline
            run_cleaning_pipeline(raw_file, clean_file)
        except Exception as e:
            logger.error(f"Cleaning failed: {e}")
            return


def get_retrieval_engine() -> RetrievalEngine:
    """Get or create the retrieval engine."""
    global _retrieval_engine
    if _retrieval_engine is not None:
        return _retrieval_engine

    settings = get_settings()
    store = get_or_create_vector_store()
    _retrieval_engine = RetrievalEngine(
        vector_store=store,
        top_k=settings.top_k_results,
        similarity_threshold=settings.similarity_threshold,
    )
    return _retrieval_engine


def get_agent() -> RecommendationAgent:
    """Get or create the recommendation agent."""
    global _agent
    if _agent is not None:
        return _agent

    settings = get_settings()
    retrieval_engine = get_retrieval_engine()
    llm_client = create_llm_client(settings)
    _agent = RecommendationAgent(
        retrieval_engine=retrieval_engine,
        llm_client=llm_client,
    )
    return _agent


async def startup() -> None:
    """Application startup handler — non-fatal so /health always passes."""
    logger.info("=" * 60)
    logger.info("SHL Assessment Recommendation System — Starting Up")
    logger.info("=" * 60)

    settings = get_settings()
    logger.info(f"Environment: {settings.app_env}")
    logger.info(f"LLM Provider: {settings.llm_provider} / {settings.active_model}")
    logger.info(f"Vector Store: {settings.vector_store_type}")

    # Initialize in background — do NOT block startup.
    # /health returns OK immediately; agent loads on first /chat request.
    try:
        get_agent()
        logger.info("✅ System initialized successfully")
    except Exception as e:
        # Log error but do NOT raise — /health must return 200 immediately
        # Agent will retry lazily on first /chat call
        logger.error(f"⚠️ Startup initialization deferred: {e}. Will retry on first request.")


async def shutdown() -> None:
    """Application shutdown handler."""
    global _vector_store, _retrieval_engine, _agent
    logger.info("Shutting down SHL Assessment Recommendation System...")
    _vector_store = None
    _retrieval_engine = None
    _agent = None
    logger.info("Shutdown complete")
