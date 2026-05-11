"""
SHL Catalog Scraper
====================
Scrapes Individual Test Solutions from:
  https://www.shl.com/solutions/products/product-catalog/

Uses requests + BeautifulSoup.
Playwright fallback for JS-rendered pages.

Strategy:
1. Fetch the catalog page
2. Parse pagination to get all products
3. For each product URL, extract detailed metadata
4. Clean and structure data
5. Save to JSON
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

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
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
}

# SHL catalog uses query params for filtering
# type=1 = Individual Test Solutions (NOT packaged solutions)
INDIVIDUAL_SOLUTIONS_PARAMS = {
    "type": "1",  # Individual Test Solutions
}


def fetch_page(url: str, params: Optional[dict] = None, retries: int = 3) -> Optional[BeautifulSoup]:
    """Fetch a page with retry logic."""
    for attempt in range(retries):
        try:
            response = requests.get(
                url,
                headers=HEADERS,
                params=params,
                timeout=30,
            )
            response.raise_for_status()
            return BeautifulSoup(response.text, "lxml")
        except requests.RequestException as e:
            logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None


def fetch_with_playwright(url: str) -> Optional[BeautifulSoup]:
    """Fallback: fetch JS-rendered page with Playwright."""
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="networkidle", timeout=30000)
            content = page.content()
            browser.close()
            return BeautifulSoup(content, "lxml")
    except ImportError:
        logger.warning("Playwright not available. Using requests only.")
        return None
    except Exception as e:
        logger.error(f"Playwright error for {url}: {e}")
        return None


def parse_catalog_page(soup: BeautifulSoup) -> list[dict]:
    """Parse a catalog listing page and extract product links + basic info."""
    products = []

    # SHL catalog table structure
    # Look for product rows in the catalog table
    table = soup.find("table", class_=re.compile(r"product|catalog", re.I))
    if not table:
        # Try alternative selectors
        table = soup.find("div", class_=re.compile(r"product.list|catalog.list", re.I))

    # Try to find product links - SHL uses custom catalog tables
    # Parse all anchor tags that point to product pages
    product_links = set()

    # Pattern 1: Direct product catalog links
    for a in soup.find_all("a", href=re.compile(r"/product-catalog/view/", re.I)):
        href = a.get("href", "")
        full_url = urljoin(BASE_URL, href)
        name = a.get_text(strip=True)
        if name and full_url not in product_links:
            product_links.add(full_url)
            products.append({"name": name, "url": full_url})

    # Pattern 2: Data table rows
    rows = soup.find_all("tr")
    for row in rows:
        link = row.find("a", href=re.compile(r"/product-catalog/view/", re.I))
        if link:
            href = link.get("href", "")
            full_url = urljoin(BASE_URL, href)
            name = link.get_text(strip=True)
            if name and full_url not in product_links:
                product_links.add(full_url)
                products.append({"name": name, "url": full_url})

    logger.info(f"Found {len(products)} products on page")
    return products


def get_all_catalog_pages(base_url: str) -> list[dict]:
    """Fetch all pages of the catalog (handles pagination)."""
    all_products = []
    start = 0
    page_size = 12  # SHL typically shows 12 items per page

    while True:
        params = {
            **INDIVIDUAL_SOLUTIONS_PARAMS,
            "start": start,
        }
        logger.info(f"Fetching catalog page with start={start}")
        soup = fetch_page(base_url, params=params)

        if not soup:
            # Try playwright fallback
            url_with_params = f"{base_url}?type=1&start={start}"
            soup = fetch_with_playwright(url_with_params)

        if not soup:
            logger.error("Failed to fetch catalog page")
            break

        page_products = parse_catalog_page(soup)
        if not page_products:
            logger.info("No more products found — pagination complete")
            break

        all_products.extend(page_products)
        logger.info(f"Total products found so far: {len(all_products)}")

        # Check if there's a next page
        next_btn = soup.find(
            ["a", "button"],
            string=re.compile(r"next|›|»", re.I),
        )
        if not next_btn:
            # Also check for pagination with data-next
            pagination = soup.find(class_=re.compile(r"paginat", re.I))
            if pagination:
                next_link = pagination.find("a", string=re.compile(r"next|›|»", re.I))
                if not next_link:
                    break
            else:
                break

        start += page_size

        # Safety limit
        if start > 500:
            logger.warning("Pagination safety limit reached")
            break

        time.sleep(1)  # Be respectful

    return all_products


def extract_detail_text(label: str, soup: BeautifulSoup) -> str:
    """Extract text value for a given label from product detail page."""
    # Look for label-value pairs in various formats
    patterns = [
        lambda s: s.find(string=re.compile(re.escape(label), re.I)),
        lambda s: s.find(class_=re.compile(label.lower().replace(" ", "-"), re.I)),
    ]
    for pattern in patterns:
        found = pattern(soup)
        if found:
            # Get next sibling text
            parent = found.parent if hasattr(found, "parent") else None
            if parent:
                next_sib = parent.find_next_sibling()
                if next_sib:
                    return next_sib.get_text(strip=True)
    return ""


def parse_yes_no(text: str) -> bool:
    """Parse Yes/No or check marks to boolean."""
    if not text:
        return False
    t = text.lower().strip()
    return t in ("yes", "true", "✓", "✔", "y", "1")


def scrape_product_detail(url: str) -> dict:
    """Scrape detailed metadata from a single product page."""
    soup = fetch_page(url)
    if not soup:
        soup = fetch_with_playwright(url)
    if not soup:
        return {}

    data = {"url": url}

    # ---- Name ----
    h1 = soup.find("h1")
    if h1:
        data["name"] = h1.get_text(strip=True)

    # ---- Description ----
    # Try multiple common patterns
    desc_selectors = [
        {"class_": re.compile(r"product.desc|description|overview|intro", re.I)},
        {"id": re.compile(r"description|overview", re.I)},
        {"itemprop": "description"},
    ]
    for sel in desc_selectors:
        el = soup.find(["div", "p", "section"], **sel)
        if el:
            data["description"] = el.get_text(separator=" ", strip=True)[:1000]
            break

    if "description" not in data:
        # Fallback: first substantial paragraph
        for p in soup.find_all("p"):
            text = p.get_text(strip=True)
            if len(text) > 80:
                data["description"] = text[:1000]
                break

    # ---- Assessment type from page metadata / breadcrumbs ----
    breadcrumb = soup.find(class_=re.compile(r"breadcrumb", re.I))
    if breadcrumb:
        crumbs = [a.get_text(strip=True) for a in breadcrumb.find_all("a")]
        if crumbs:
            data["category"] = crumbs[-1] if len(crumbs) > 1 else ""

    # ---- Find details table / definition list ----
    # SHL product pages usually have a details section
    detail_section = soup.find(
        ["div", "section"],
        class_=re.compile(r"product.detail|assessment.detail|spec", re.I),
    )
    if not detail_section:
        detail_section = soup

    # Extract key-value pairs from tables and definition lists
    kv_pairs = {}
    for dt in detail_section.find_all("dt"):
        dd = dt.find_next_sibling("dd")
        if dd:
            key = dt.get_text(strip=True).lower()
            val = dd.get_text(strip=True)
            kv_pairs[key] = val

    for tr in detail_section.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if len(cells) >= 2:
            key = cells[0].get_text(strip=True).lower()
            val = cells[1].get_text(strip=True)
            kv_pairs[key] = val

    # ---- Map extracted KV to our schema ----
    for key, val in kv_pairs.items():
        if any(k in key for k in ["assessment type", "test type", "type"]):
            data["test_type"] = val
        elif any(k in key for k in ["duration", "time", "minutes"]):
            data["duration"] = val
        elif any(k in key for k in ["remote", "online"]):
            data["remote_testing"] = parse_yes_no(val)
        elif any(k in key for k in ["adaptive", "irt"]):
            data["adaptive"] = parse_yes_no(val)
        elif any(k in key for k in ["skills", "measures", "competencies"]):
            data["skills_measured"] = [
                s.strip() for s in re.split(r"[,;\n|•]", val) if s.strip()
            ]

    # ---- Infer test_type from page content if not found ----
    if not data.get("test_type"):
        page_text = soup.get_text().lower()
        type_map = {
            "Cognitive Ability": ["cognitive", "reasoning", "numerical", "verbal", "inductive", "deductive"],
            "Personality": ["personality", "opq", "behavior", "behaviour", "trait"],
            "Skills": ["skills", "coding", "programming", "microsoft", "excel", "java", "python"],
            "Situational Judgement": ["situational", "sjt", "judgement", "judgment"],
            "Biodata": ["biodata", "background", "experience"],
            "Language": ["language", "english", "grammar", "reading"],
            "Competency": ["competency", "competencies", "leadership"],
        }
        for t_type, keywords in type_map.items():
            if any(kw in page_text for kw in keywords):
                data["test_type"] = t_type
                break

    # ---- Tags from meta keywords or page content ----
    meta_kw = soup.find("meta", attrs={"name": "keywords"})
    if meta_kw:
        content = meta_kw.get("content", "")
        data["tags"] = [t.strip() for t in content.split(",") if t.strip()]

    # ---- Generate keywords from name + description ----
    name = data.get("name", "")
    desc = data.get("description", "")
    combined = f"{name} {desc}".lower()
    # Extract meaningful words
    words = re.findall(r"\b[a-z]{4,}\b", combined)
    stop_words = {"this", "that", "with", "from", "have", "been", "will", "your", "their",
                  "they", "also", "which", "these", "those", "what", "when", "where", "into"}
    data["keywords"] = list(set(w for w in words if w not in stop_words))[:20]

    logger.info(f"Scraped: {data.get('name', url)}")
    return data


def run_scraper(output_path: str = "./data/shl_raw.json") -> list[dict]:
    """
    Main scraper entry point.
    Returns list of scraped assessment dicts.
    """
    logger.info("=" * 60)
    logger.info("Starting SHL Catalog Scraper")
    logger.info(f"Catalog URL: {CATALOG_URL}")
    logger.info("=" * 60)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Get all product URLs from catalog
    logger.info("Step 1: Fetching catalog product list...")
    products = get_all_catalog_pages(CATALOG_URL)
    logger.info(f"Found {len(products)} products in catalog")

    if not products:
        logger.warning("No products found. Trying alternative approaches...")
        # Try direct fetch with playwright
        soup = fetch_with_playwright(f"{CATALOG_URL}?type=1")
        if soup:
            products = parse_catalog_page(soup)

    # Step 2: Scrape each product detail page
    logger.info("Step 2: Scraping individual product pages...")
    detailed_products = []

    for i, product in enumerate(products, 1):
        url = product.get("url", "")
        if not url:
            continue

        logger.info(f"[{i}/{len(products)}] Scraping: {url}")
        detail = scrape_product_detail(url)

        if detail:
            # Merge basic info with detail
            merged = {**product, **detail}
            detailed_products.append(merged)
        else:
            # Keep basic info if detail scraping failed
            detailed_products.append(product)

        time.sleep(0.5)  # Polite delay

    # Step 3: Save raw data
    logger.info(f"Step 3: Saving {len(detailed_products)} products to {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(detailed_products, f, indent=2, ensure_ascii=False)

    logger.info("Scraping complete!")
    return detailed_products


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    results = run_scraper()
    print(f"\n✅ Scraped {len(results)} assessments")
