"""
scraper.py — SHL Catalog Scraper
===================================
Scrapes Individual Test Solutions from:
  https://www.shl.com/solutions/products/product-catalog/

Features:
- requests + BeautifulSoup
- 3 retries with exponential backoff
- 1-2 second rate limiting
- Pagination support
- Falls back to hardcoded catalog if scraping fails
- Saves catalog.json, catalog.csv, and FAISS index
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import re
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import numpy as np
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

BASE_URL = "https://www.shl.com"
CATALOG_URL = "https://www.shl.com/solutions/products/product-catalog/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
}

# ============================================================
# Hardcoded Fallback Catalog (real SHL assessments)
# ============================================================
FALLBACK_CATALOG = [
    {
        "name": "OPQ32r",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/opq32r/",
        "description": "The OPQ32r is SHL's flagship occupational personality questionnaire measuring 32 personality characteristics relevant to work performance.",
        "test_type": "P",
        "remote_testing": True,
        "adaptive_support": False,
        "duration": "25-40 minutes",
        "languages": ["English", "Spanish", "French", "German", "Chinese"],
        "job_levels": ["graduate", "mid-professional", "manager", "director"],
        "job_families": ["HR", "Finance", "Sales", "Operations", "IT"],
        "category": "Personality",
        "skills_measured": ["personality", "behavior", "leadership", "teamwork"],
    },
    {
        "name": "Verify G+ (Global Skills Assessment)",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/verify-g-plus/",
        "description": "Verify G+ assesses general cognitive ability including numerical, verbal, and inductive reasoning for mid-to-senior level roles.",
        "test_type": "A",
        "remote_testing": True,
        "adaptive_support": True,
        "duration": "36 minutes",
        "languages": ["English", "Spanish", "French"],
        "job_levels": ["mid-professional", "manager", "director"],
        "job_families": ["IT", "Finance", "Engineering", "Management"],
        "category": "Cognitive Ability",
        "skills_measured": ["numerical reasoning", "verbal reasoning", "inductive reasoning"],
    },
    {
        "name": "Verify Numerical Reasoning",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/verify-numerical-reasoning/",
        "description": "Measures the ability to work with numbers, interpret data, and draw conclusions from numerical information.",
        "test_type": "A",
        "remote_testing": True,
        "adaptive_support": True,
        "duration": "17-25 minutes",
        "languages": ["English", "Spanish", "French", "German"],
        "job_levels": ["graduate", "mid-professional", "manager"],
        "job_families": ["Finance", "IT", "Engineering", "Analytics"],
        "category": "Cognitive Ability",
        "skills_measured": ["numerical reasoning", "data interpretation", "quantitative analysis"],
    },
    {
        "name": "Verify Verbal Reasoning",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/verify-verbal-reasoning/",
        "description": "Assesses the ability to understand written information, evaluate arguments, and draw logical conclusions from text.",
        "test_type": "A",
        "remote_testing": True,
        "adaptive_support": True,
        "duration": "17-19 minutes",
        "languages": ["English", "Spanish", "French", "German"],
        "job_levels": ["graduate", "mid-professional", "manager"],
        "job_families": ["HR", "Sales", "Marketing", "Operations"],
        "category": "Cognitive Ability",
        "skills_measured": ["verbal reasoning", "reading comprehension", "logical analysis"],
    },
    {
        "name": "Verify Inductive Reasoning",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/verify-inductive-reasoning/",
        "description": "Measures the ability to identify patterns and rules in abstract information — a key indicator of learning ability.",
        "test_type": "A",
        "remote_testing": True,
        "adaptive_support": True,
        "duration": "24 minutes",
        "languages": ["English"],
        "job_levels": ["graduate", "mid-professional"],
        "job_families": ["IT", "Engineering", "Technology"],
        "category": "Cognitive Ability",
        "skills_measured": ["inductive reasoning", "pattern recognition", "abstract thinking"],
    },
    {
        "name": "Verify Deductive Reasoning",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/verify-deductive-reasoning/",
        "description": "Assesses ability to draw logical conclusions from rules, apply them to new situations and identify logical flaws.",
        "test_type": "A",
        "remote_testing": True,
        "adaptive_support": False,
        "duration": "20 minutes",
        "languages": ["English"],
        "job_levels": ["graduate", "mid-professional"],
        "job_families": ["IT", "Legal", "Finance"],
        "category": "Cognitive Ability",
        "skills_measured": ["deductive reasoning", "logical thinking", "rule application"],
    },
    {
        "name": "Occupational Personality Questionnaire (OPQ32)",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/occupational-personality-questionnaire-opq32/",
        "description": "Comprehensive personality assessment measuring 32 work-relevant personality traits for selection and development.",
        "test_type": "P",
        "remote_testing": True,
        "adaptive_support": False,
        "duration": "25-45 minutes",
        "languages": ["English", "Spanish", "French", "German", "Chinese", "Arabic"],
        "job_levels": ["graduate", "mid-professional", "manager", "director", "executive"],
        "job_families": ["All"],
        "category": "Personality",
        "skills_measured": ["personality", "leadership", "teamwork", "communication"],
    },
    {
        "name": "Motivation Questionnaire (MQ)",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/motivation-questionnaire/",
        "description": "Assesses what motivates and energizes candidates at work, covering 18 motivational dimensions.",
        "test_type": "P",
        "remote_testing": True,
        "adaptive_support": False,
        "duration": "25 minutes",
        "languages": ["English", "Spanish", "French"],
        "job_levels": ["mid-professional", "manager", "director"],
        "job_families": ["Sales", "HR", "Management"],
        "category": "Personality",
        "skills_measured": ["motivation", "engagement", "drive", "values"],
    },
    {
        "name": "Situational Judgement Test (SJT)",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/situational-judgement-test/",
        "description": "Presents realistic work scenarios and assesses how candidates would respond — measuring judgment and decision-making.",
        "test_type": "S",
        "remote_testing": True,
        "adaptive_support": False,
        "duration": "30-45 minutes",
        "languages": ["English"],
        "job_levels": ["graduate", "mid-professional"],
        "job_families": ["Customer Service", "Sales", "Operations"],
        "category": "Situational Judgement",
        "skills_measured": ["situational judgment", "decision making", "problem solving"],
    },
    {
        "name": "ADEPT-15",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/adept-15/",
        "description": "A 15-factor personality assessment measuring adaptive behaviors and how people adapt to different work situations.",
        "test_type": "P",
        "remote_testing": True,
        "adaptive_support": False,
        "duration": "25 minutes",
        "languages": ["English"],
        "job_levels": ["mid-professional", "manager", "director"],
        "job_families": ["IT", "Finance", "Operations", "HR"],
        "category": "Personality",
        "skills_measured": ["adaptability", "leadership", "interpersonal skills"],
    },
    {
        "name": "RemoteWorkQ",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/remoteworkq/",
        "description": "Assesses competencies critical for success in remote working environments including self-discipline, communication, and tech comfort.",
        "test_type": "P",
        "remote_testing": True,
        "adaptive_support": False,
        "duration": "10-15 minutes",
        "languages": ["English"],
        "job_levels": ["graduate", "mid-professional", "manager"],
        "job_families": ["IT", "Operations", "HR", "Finance"],
        "category": "Personality",
        "skills_measured": ["remote work", "self-discipline", "digital communication"],
    },
    {
        "name": "Coding Pro for Python",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/coding-pro-for-python/",
        "description": "Technical coding assessment for Python developers covering data structures, algorithms, OOP, and Python-specific patterns.",
        "test_type": "K",
        "remote_testing": True,
        "adaptive_support": False,
        "duration": "40-90 minutes",
        "languages": ["English"],
        "job_levels": ["graduate", "mid-professional", "senior"],
        "job_families": ["IT", "Engineering", "Technology"],
        "category": "Technical Skills",
        "skills_measured": ["python", "coding", "algorithms", "data structures"],
    },
    {
        "name": "Coding Pro for Java",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/coding-pro-for-java/",
        "description": "Technical coding assessment for Java developers covering Java syntax, OOP, Spring, data structures, and algorithms.",
        "test_type": "K",
        "remote_testing": True,
        "adaptive_support": False,
        "duration": "40-90 minutes",
        "languages": ["English"],
        "job_levels": ["graduate", "mid-professional", "senior"],
        "job_families": ["IT", "Engineering", "Technology"],
        "category": "Technical Skills",
        "skills_measured": ["java", "coding", "OOP", "algorithms", "Spring"],
    },
    {
        "name": "Coding Pro for JavaScript",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/coding-pro-for-javascript/",
        "description": "Technical coding assessment for JavaScript developers covering ES6+, DOM, async patterns, and Node.js fundamentals.",
        "test_type": "K",
        "remote_testing": True,
        "adaptive_support": False,
        "duration": "40-90 minutes",
        "languages": ["English"],
        "job_levels": ["graduate", "mid-professional"],
        "job_families": ["IT", "Engineering", "Technology"],
        "category": "Technical Skills",
        "skills_measured": ["javascript", "coding", "web development", "Node.js"],
    },
    {
        "name": "Microsoft Excel (Intermediate Level)",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/microsoft-excel-intermediate-level/",
        "description": "Assesses intermediate Excel skills including formulas, pivot tables, data analysis, and charting.",
        "test_type": "K",
        "remote_testing": True,
        "adaptive_support": False,
        "duration": "45 minutes",
        "languages": ["English"],
        "job_levels": ["graduate", "mid-professional"],
        "job_families": ["Finance", "Operations", "HR", "Analytics"],
        "category": "Technical Skills",
        "skills_measured": ["excel", "spreadsheet", "data analysis", "formulas"],
    },
    {
        "name": "Verify Interactive - Numerical",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/verify-interactive-numerical/",
        "description": "An interactive version of the numerical reasoning test with engaging graphics and realistic business scenarios.",
        "test_type": "A",
        "remote_testing": True,
        "adaptive_support": True,
        "duration": "20-25 minutes",
        "languages": ["English"],
        "job_levels": ["graduate", "mid-professional"],
        "job_families": ["Finance", "Banking", "Analytics"],
        "category": "Cognitive Ability",
        "skills_measured": ["numerical reasoning", "data interpretation"],
    },
    {
        "name": "Verify Interactive - Inductive",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/verify-interactive-inductive/",
        "description": "Interactive inductive reasoning test with engaging visuals and scenario-based abstract pattern questions.",
        "test_type": "A",
        "remote_testing": True,
        "adaptive_support": True,
        "duration": "20-25 minutes",
        "languages": ["English"],
        "job_levels": ["graduate", "mid-professional"],
        "job_families": ["IT", "Engineering"],
        "category": "Cognitive Ability",
        "skills_measured": ["inductive reasoning", "abstract thinking"],
    },
    {
        "name": "Graduate 8.0 (Entry Level 8)",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/graduate-8/",
        "description": "A cognitive ability battery for graduate-level candidates assessing numerical, verbal, and inductive reasoning.",
        "test_type": "A",
        "remote_testing": True,
        "adaptive_support": False,
        "duration": "36 minutes",
        "languages": ["English"],
        "job_levels": ["graduate", "entry-level"],
        "job_families": ["All"],
        "category": "Cognitive Ability",
        "skills_measured": ["numerical reasoning", "verbal reasoning", "inductive reasoning"],
    },
    {
        "name": "Customer Contact Styles Questionnaire",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/customer-contact-styles-questionnaire/",
        "description": "Personality questionnaire designed for customer-facing roles, measuring service orientation and interpersonal style.",
        "test_type": "P",
        "remote_testing": True,
        "adaptive_support": False,
        "duration": "20-25 minutes",
        "languages": ["English"],
        "job_levels": ["entry-level", "graduate", "mid-professional"],
        "job_families": ["Customer Service", "Sales", "Retail"],
        "category": "Personality",
        "skills_measured": ["customer orientation", "communication", "service attitude"],
    },
    {
        "name": "Verify Calculation",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/verify-calculation/",
        "description": "Tests basic to intermediate calculation skills for roles requiring numerical accuracy and arithmetic ability.",
        "test_type": "A",
        "remote_testing": True,
        "adaptive_support": False,
        "duration": "10-15 minutes",
        "languages": ["English", "Spanish"],
        "job_levels": ["entry-level", "graduate"],
        "job_families": ["Finance", "Retail", "Operations"],
        "category": "Cognitive Ability",
        "skills_measured": ["arithmetic", "calculation", "numerical accuracy"],
    },
    {
        "name": "Verify Checking",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/verify-checking/",
        "description": "Measures speed and accuracy in checking data, names, codes, and other clerical information.",
        "test_type": "A",
        "remote_testing": True,
        "adaptive_support": False,
        "duration": "10 minutes",
        "languages": ["English"],
        "job_levels": ["entry-level", "graduate"],
        "job_families": ["Operations", "Finance", "HR", "Admin"],
        "category": "Cognitive Ability",
        "skills_measured": ["attention to detail", "data checking", "clerical accuracy"],
    },
    {
        "name": "Structured Interview Guide",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/structured-interview-guide/",
        "description": "Provides evidence-based interview questions aligned to competencies to help interviewers assess candidates consistently.",
        "test_type": "C",
        "remote_testing": True,
        "adaptive_support": False,
        "duration": "Variable",
        "languages": ["English"],
        "job_levels": ["manager", "director", "executive"],
        "job_families": ["HR", "Management"],
        "category": "Competency",
        "skills_measured": ["leadership", "competency", "interview"],
    },
    {
        "name": "Sales Assessment",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/sales-assessment/",
        "description": "Comprehensive assessment for sales roles measuring cognitive ability, personality traits, and sales-specific competencies.",
        "test_type": "A,P",
        "remote_testing": True,
        "adaptive_support": False,
        "duration": "45-60 minutes",
        "languages": ["English", "Spanish"],
        "job_levels": ["graduate", "mid-professional"],
        "job_families": ["Sales", "Business Development"],
        "category": "Sales",
        "skills_measured": ["sales acumen", "persuasion", "resilience", "customer focus"],
    },
]


def _add_raw_text(item: dict) -> dict:
    """Generate embedding-ready text from assessment fields."""
    parts = [
        item.get("name", ""),
        item.get("description", ""),
        item.get("test_type", ""),
        item.get("category", ""),
        " ".join(item.get("job_families", [])),
        " ".join(item.get("skills_measured", [])),
        " ".join(item.get("job_levels", [])),
    ]
    item["raw_text"] = " | ".join(p for p in parts if p)
    return item


# ============================================================
# HTTP Fetch with Retry
# ============================================================
def fetch_page(url: str, params: Optional[dict] = None, retries: int = 3) -> Optional[BeautifulSoup]:
    """Fetch a page with exponential backoff retry."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "lxml")
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None


# ============================================================
# Catalog Page Parsing
# ============================================================
def parse_catalog_page(soup: BeautifulSoup) -> list[dict]:
    """Parse catalog listing page and extract product links."""
    products = []
    seen_urls = set()

    for a in soup.find_all("a", href=re.compile(r"/product-catalog/view/", re.I)):
        href = a.get("href", "")
        full_url = urljoin(BASE_URL, href)
        name = a.get_text(strip=True)
        if name and full_url not in seen_urls:
            seen_urls.add(full_url)
            products.append({"name": name, "url": full_url})

    return products


def get_all_catalog_pages(base_url: str) -> list[dict]:
    """Fetch all catalog pages handling pagination."""
    all_products = []
    start = 0
    page_size = 12

    while True:
        params = {"type": "1", "start": start}
        logger.info(f"Fetching catalog page start={start}...")
        soup = fetch_page(base_url, params=params)

        if not soup:
            logger.error("Failed to fetch catalog page — stopping pagination")
            break

        page_products = parse_catalog_page(soup)
        if not page_products:
            logger.info("No more products found — pagination complete")
            break

        # Deduplicate
        existing_urls = {p["url"] for p in all_products}
        new_products = [p for p in page_products if p["url"] not in existing_urls]
        all_products.extend(new_products)
        logger.info(f"Found {len(new_products)} new products (total: {len(all_products)})")

        if len(new_products) == 0:
            break

        # Check for next page
        next_btn = soup.find(["a", "li"], class_=re.compile(r"next", re.I))
        if not next_btn:
            next_btn = soup.find("a", string=re.compile(r"next|»|›", re.I))
        if not next_btn:
            break

        start += page_size
        if start > 600:
            logger.warning("Safety limit reached — stopping")
            break

        time.sleep(1.5)  # Rate limiting

    return all_products


def scrape_product_detail(url: str) -> dict:
    """Scrape metadata from a single product page."""
    soup = fetch_page(url)
    if not soup:
        return {}

    data = {"url": url}

    # Name
    h1 = soup.find("h1")
    if h1:
        data["name"] = h1.get_text(strip=True)

    # Description
    for sel in [{"class_": re.compile(r"description|overview|intro", re.I)}, {"itemprop": "description"}]:
        el = soup.find(["div", "p", "section"], **sel)
        if el:
            data["description"] = el.get_text(separator=" ", strip=True)[:800]
            break

    if "description" not in data:
        for p in soup.find_all("p"):
            text = p.get_text(strip=True)
            if len(text) > 80:
                data["description"] = text[:800]
                break

    # Key-value pairs from detail tables
    kv = {}
    for dt in soup.find_all("dt"):
        dd = dt.find_next_sibling("dd")
        if dd:
            kv[dt.get_text(strip=True).lower()] = dd.get_text(strip=True)

    for tr in soup.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if len(cells) >= 2:
            kv[cells[0].get_text(strip=True).lower()] = cells[1].get_text(strip=True)

    for key, val in kv.items():
        if any(k in key for k in ["assessment type", "test type", "type"]):
            data["test_type"] = val
        elif any(k in key for k in ["duration", "time", "minutes"]):
            data["duration"] = val
        elif "remote" in key or "online" in key:
            data["remote_testing"] = val.lower() in ("yes", "true", "✓", "y")
        elif "adaptive" in key or "irt" in key:
            data["adaptive_support"] = val.lower() in ("yes", "true", "✓", "y")

    return data


# ============================================================
# FAISS Index Builder
# ============================================================
def build_faiss_index(catalog: list[dict], index_path: str = "./faiss_index") -> bool:
    """Build and save FAISS index from catalog."""
    try:
        import faiss
        from sentence_transformers import SentenceTransformer

        logger.info(f"Building FAISS index for {len(catalog)} assessments...")
        model = SentenceTransformer("all-MiniLM-L6-v2")

        texts = [a.get("raw_text", a.get("name", "")) for a in catalog]
        embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype=np.float32)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        Path(index_path).mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(Path(index_path) / "index.faiss"))

        with open(Path(index_path) / "index.pkl", "wb") as f:
            pickle.dump(catalog, f)

        logger.info(f"FAISS index saved to {index_path} ({index.ntotal} vectors)")
        return True

    except Exception as e:
        logger.error(f"Failed to build FAISS index: {e}")
        return False


# ============================================================
# Main Entry Point
# ============================================================
def run_scraper(
    output_json: str = "./catalog.json",
    output_csv: str = "./catalog.csv",
    faiss_dir: str = "./faiss_index",
) -> list[dict]:
    """
    Main scraper: scrape SHL catalog → save JSON/CSV → build FAISS index.
    Falls back to hardcoded catalog if scraping fails.
    """
    logger.info("=" * 60)
    logger.info("SHL Catalog Scraper — Starting")
    logger.info("=" * 60)

    catalog: list[dict] = []

    # --- Step 1: Attempt live scraping ---
    try:
        logger.info("Step 1: Fetching product list from SHL catalog...")
        products = get_all_catalog_pages(CATALOG_URL)
        logger.info(f"Found {len(products)} products in catalog listing")

        if products:
            logger.info("Step 2: Scraping individual product pages...")
            detailed = []
            for i, p in enumerate(products, 1):
                url = p.get("url", "")
                if not url:
                    continue
                logger.info(f"  [{i}/{len(products)}] Scraping: {url}")
                detail = scrape_product_detail(url)
                merged = {**p, **detail}
                merged = _add_raw_text(merged)
                detailed.append(merged)
                time.sleep(1)  # Rate limiting

            if detailed:
                catalog = detailed
                logger.info(f"Successfully scraped {len(catalog)} assessments")

    except Exception as e:
        logger.error(f"Live scraping failed: {e}")

    # --- Step 2: Fall back if needed ---
    if len(catalog) < 10:
        logger.warning(
            f"Only {len(catalog)} items scraped — using fallback hardcoded catalog"
        )
        catalog = [_add_raw_text(dict(item)) for item in FALLBACK_CATALOG]

    # Merge scraped + fallback (add fallback items not already in scraped)
    if len(catalog) > 0 and catalog is not FALLBACK_CATALOG:
        existing_names = {a.get("name", "").lower() for a in catalog}
        for fb in FALLBACK_CATALOG:
            if fb["name"].lower() not in existing_names:
                catalog.append(_add_raw_text(dict(fb)))

    logger.info(f"Final catalog size: {len(catalog)} assessments")

    # --- Step 3: Save JSON ---
    out_json = Path(output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved catalog.json → {out_json}")

    # --- Step 4: Save CSV ---
    try:
        import pandas as pd
        df = pd.DataFrame(catalog)
        out_csv = Path(output_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        logger.info(f"Saved catalog.csv → {out_csv}")
    except Exception as e:
        logger.warning(f"CSV save skipped: {e}")

    # --- Step 5: Build FAISS index ---
    build_faiss_index(catalog, index_path=faiss_dir)

    logger.info("=" * 60)
    logger.info(f"Scraping complete! {len(catalog)} assessments ready.")
    logger.info("=" * 60)
    return catalog


if __name__ == "__main__":
    import sys
    base = Path(__file__).parent
    run_scraper(
        output_json=str(base / "catalog.json"),
        output_csv=str(base / "catalog.csv"),
        faiss_dir=str(base / "faiss_index"),
    )
    print(f"\n✅ Done! Check catalog.json and faiss_index/")
