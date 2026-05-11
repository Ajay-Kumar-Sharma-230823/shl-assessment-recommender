"""
Data Cleaning Pipeline
========================
Cleans, normalizes, deduplicates, and validates raw scraped data.
Produces a clean, structured SHL catalog JSON.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---- Known SHL test types (for normalization) ----
TEST_TYPE_NORMALIZATION = {
    # Cognitive
    "cognitive": "Cognitive Ability",
    "cognitive ability": "Cognitive Ability",
    "reasoning": "Cognitive Ability",
    "numerical": "Cognitive Ability",
    "verbal": "Cognitive Ability",
    "inductive": "Cognitive Ability",
    "deductive": "Cognitive Ability",
    "abstract": "Cognitive Ability",
    "general ability": "Cognitive Ability",
    "verify": "Cognitive Ability",
    # Personality
    "personality": "Personality",
    "personality questionnaire": "Personality",
    "opq": "Personality",
    "behaviour": "Personality",
    "behavior": "Personality",
    # Skills / Knowledge
    "skills": "Skills",
    "knowledge": "Skills",
    "coding": "Skills",
    "programming": "Skills",
    "microsoft office": "Skills",
    "technical": "Skills",
    # Situational Judgement
    "situational judgement": "Situational Judgement",
    "situational judgment": "Situational Judgement",
    "sjt": "Situational Judgement",
    # Biodata
    "biodata": "Biodata",
    "biographical": "Biodata",
    # Language
    "language": "Language",
    "english": "Language",
    # Competency
    "competency": "Competency",
    "competencies": "Competency",
    "leadership": "Competency",
    # Simulation
    "simulation": "Simulation",
    "game": "Simulation",
    "gamified": "Simulation",
}


def normalize_test_type(raw_type: str) -> str:
    """Normalize a raw test type string to a canonical category."""
    if not raw_type:
        return "General"
    lower = raw_type.lower().strip()
    for keyword, normalized in TEST_TYPE_NORMALIZATION.items():
        if keyword in lower:
            return normalized
    # Return title-cased original if no match
    return raw_type.strip().title()


def clean_text(text: str) -> str:
    """Remove excessive whitespace and normalize text."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def clean_url(url: str) -> str:
    """Ensure URL is properly formed."""
    if not url:
        return ""
    url = url.strip()
    if not url.startswith("http"):
        url = "https://www.shl.com" + url
    # Remove trailing query params that are not part of the canonical URL
    url = url.split("?")[0]
    if not url.endswith("/"):
        url += "/"
    return url


def extract_duration_minutes(duration_str: str) -> Optional[int]:
    """Extract numeric duration in minutes from string."""
    if not duration_str:
        return None
    match = re.search(r"(\d+)", duration_str)
    if match:
        return int(match.group(1))
    return None


def deduplicate(assessments: list[dict]) -> list[dict]:
    """Remove duplicate assessments by URL."""
    seen_urls = set()
    unique = []
    for a in assessments:
        url = a.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique.append(a)
        elif not url:
            unique.append(a)  # Keep items without URL (will be flagged)
    return unique


def validate_assessment(a: dict) -> tuple[bool, list[str]]:
    """Validate an assessment record. Returns (is_valid, errors)."""
    errors = []
    if not a.get("name"):
        errors.append("Missing name")
    if not a.get("url"):
        errors.append("Missing URL")
    elif "shl.com" not in a.get("url", ""):
        errors.append(f"Non-SHL URL: {a.get('url')}")
    return len(errors) == 0, errors


def build_raw_text(a: dict) -> str:
    """Build combined raw text for embedding."""
    parts = [
        a.get("name", ""),
        a.get("description", ""),
        a.get("test_type", ""),
        a.get("category", ""),
        " ".join(a.get("skills_measured", [])),
        " ".join(a.get("tags", [])),
        " ".join(a.get("keywords", [])),
    ]
    return " | ".join(p for p in parts if p)


def clean_pipeline(raw_data: list[dict]) -> list[dict]:
    """
    Full cleaning pipeline.
    1. Normalize fields
    2. Deduplicate
    3. Validate
    4. Build raw_text for embeddings
    """
    logger.info(f"Starting clean pipeline with {len(raw_data)} raw records")

    cleaned = []
    skipped = 0

    for raw in raw_data:
        a: dict[str, Any] = {}

        # ---- Core fields ----
        a["name"] = clean_text(raw.get("name", ""))
        a["url"] = clean_url(raw.get("url", ""))
        a["description"] = clean_text(raw.get("description", ""))
        a["test_type"] = normalize_test_type(raw.get("test_type", ""))
        a["category"] = clean_text(raw.get("category", ""))

        # ---- List fields ----
        skills = raw.get("skills_measured", [])
        if isinstance(skills, str):
            skills = [s.strip() for s in re.split(r"[,;\n|•]", skills) if s.strip()]
        a["skills_measured"] = [clean_text(s) for s in skills if s]

        tags = raw.get("tags", [])
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]
        a["tags"] = [clean_text(t) for t in tags if t]

        keywords = raw.get("keywords", [])
        if isinstance(keywords, str):
            keywords = [k.strip() for k in keywords.split(",") if k.strip()]
        a["keywords"] = [clean_text(k) for k in keywords if k]

        # ---- Boolean fields ----
        a["remote_testing"] = bool(raw.get("remote_testing", False))
        a["adaptive"] = bool(raw.get("adaptive", False))

        # ---- Duration ----
        duration = raw.get("duration", "")
        a["duration"] = clean_text(str(duration)) if duration else None
        a["duration_minutes"] = extract_duration_minutes(a["duration"])

        # ---- Build raw text ----
        a["raw_text"] = build_raw_text(a)

        # ---- Validate ----
        is_valid, errors = validate_assessment(a)
        if not is_valid:
            logger.warning(f"Skipping invalid record: {errors} — {a.get('name', 'unnamed')}")
            skipped += 1
            continue

        cleaned.append(a)

    # ---- Deduplicate ----
    before = len(cleaned)
    cleaned = deduplicate(cleaned)
    after = len(cleaned)
    if before != after:
        logger.info(f"Removed {before - after} duplicate records")

    logger.info(f"Clean pipeline complete: {len(cleaned)} valid records, {skipped} skipped")
    return cleaned


def run_cleaning_pipeline(
    input_path: str = "./data/shl_raw.json",
    output_path: str = "./data/shl_catalog.json",
) -> list[dict]:
    """Load raw data, clean it, and save to output path."""
    input_file = Path(input_path)

    if not input_file.exists():
        raise FileNotFoundError(f"Raw data file not found: {input_path}")

    with open(input_file, encoding="utf-8") as f:
        raw_data = json.load(f)

    logger.info(f"Loaded {len(raw_data)} raw records from {input_path}")
    cleaned = clean_pipeline(raw_data)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(cleaned)} clean records to {output_path}")
    return cleaned


# Add Optional to fix type annotation
from typing import Optional

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data = run_cleaning_pipeline()
    print(f"✅ Cleaned {len(data)} assessments")
