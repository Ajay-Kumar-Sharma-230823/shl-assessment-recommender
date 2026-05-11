"""
Data Pipeline Runner
=====================
Run this script to:
1. Scrape SHL catalog
2. Clean the data
3. Build vector index

Usage:
    python run_pipeline.py [--scrape] [--clean] [--index] [--all]
    
    --all: Run complete pipeline (scrape + clean + index)
    --scrape: Only scrape
    --clean: Only clean (requires raw data)
    --index: Only build index (requires clean data)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def run_scrape(raw_output: str = "./data/shl_raw.json") -> bool:
    """Run the SHL catalog scraper."""
    logger.info("=" * 60)
    logger.info("STEP 1: Scraping SHL Catalog")
    logger.info("=" * 60)
    try:
        from scraper.shl_scraper import run_scraper
        results = run_scraper(raw_output)
        logger.info(f"✅ Scraped {len(results)} assessments → {raw_output}")
        return len(results) > 0
    except Exception as e:
        logger.error(f"❌ Scraping failed: {e}")
        return False


def run_clean(
    raw_input: str = "./data/shl_raw.json",
    clean_output: str = "./data/shl_catalog.json",
) -> bool:
    """Run the data cleaning pipeline."""
    logger.info("=" * 60)
    logger.info("STEP 2: Cleaning Data")
    logger.info("=" * 60)
    try:
        from scraper.cleaner import run_cleaning_pipeline
        results = run_cleaning_pipeline(raw_input, clean_output)
        logger.info(f"✅ Cleaned {len(results)} assessments → {clean_output}")
        return len(results) > 0
    except Exception as e:
        logger.error(f"❌ Cleaning failed: {e}")
        return False


def run_index(
    catalog_input: str = "./data/shl_catalog.json",
    store_type: str = "faiss",
    output_path: str = "./vectorstore",
    embedding_model: str = "all-MiniLM-L6-v2",
) -> bool:
    """Build vector index from catalog."""
    logger.info("=" * 60)
    logger.info(f"STEP 3: Building {store_type.upper()} Vector Index")
    logger.info("=" * 60)
    try:
        from app.retrieval.vector_store import build_vector_index
        store = build_vector_index(
            catalog_path=catalog_input,
            store_type=store_type,
            output_path=output_path,
            embedding_model=embedding_model,
        )
        logger.info(f"✅ Vector index built → {output_path}")
        
        # Test search
        test_results = store.search("cognitive ability test for software developer", top_k=3)
        logger.info(f"✅ Test search: found {len(test_results)} results")
        for r in test_results:
            logger.info(f"   [{r['rank']}] {r['assessment'].get('name')} (score={r['score']:.3f})")
        
        return True
    except Exception as e:
        logger.error(f"❌ Indexing failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="SHL Assessment Recommendation System — Data Pipeline"
    )
    parser.add_argument("--scrape", action="store_true", help="Run scraper")
    parser.add_argument("--clean", action="store_true", help="Run data cleaner")
    parser.add_argument("--index", action="store_true", help="Build vector index")
    parser.add_argument("--all", action="store_true", help="Run full pipeline")
    parser.add_argument(
        "--store-type",
        default="faiss",
        choices=["faiss", "chroma"],
        help="Vector store type",
    )
    parser.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
        help="Sentence transformer model",
    )

    args = parser.parse_args()

    if not any([args.scrape, args.clean, args.index, args.all]):
        parser.print_help()
        print("\nRun with --all to execute the complete pipeline")
        sys.exit(0)

    success = True

    if args.all or args.scrape:
        success = run_scrape() and success

    if args.all or args.clean:
        success = run_clean() and success

    if args.all or args.index:
        success = run_index(
            store_type=args.store_type,
            embedding_model=args.embedding_model,
        ) and success

    if success:
        logger.info("\n✅ Pipeline complete! You can now start the server:")
        logger.info("   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload")
    else:
        logger.error("\n❌ Pipeline completed with errors. Check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
