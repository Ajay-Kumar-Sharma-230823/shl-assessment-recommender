"""
agent.py — Core Agent Logic
=============================
Orchestrates the full pipeline:
1. Scope guard (injection + off-topic detection)
2. Query classification
3. Context extraction
4. FAISS retrieval
5. Prompt building
6. LLM call (Anthropic → Gemini → Groq → OpenAI fallback chain)
7. Response validation (catalog-only recommendations)
"""
from __future__ import annotations

# Auto-load .env files (shl_recommender/.env or parent SHL/.env)
try:
    from dotenv import load_dotenv
    from pathlib import Path as _Path
    _here = _Path(__file__).parent
    load_dotenv(_here / ".env", override=False)          # shl_recommender/.env
    load_dotenv(_here.parent / ".env", override=False)   # SHL/.env
except ImportError:
    pass  # python-dotenv not installed — keys must be set manually

import json
import logging
import os
import re
import time
from typing import Optional

from models import ChatRequest, ChatResponse, Recommendation
from prompts import (
    CLARIFY_PROMPT,
    COMPARE_PROMPT,
    FORCE_RECOMMEND_PROMPT,
    RECOMMEND_PROMPT,
    REFUSE_PROMPT,
    REFINE_PROMPT,
    CLOSE_PROMPT,
    format_catalog_summary,
    format_conversation_history,
    format_retrieved_assessments,
    build_system_prompt,
)
from retriever import Retriever, get_retriever

logger = logging.getLogger(__name__)

# ============================================================
# Scope Guard — runs before LLM
# ============================================================

_INJECTION_PATTERNS = [
    # "ignore ... instructions" — flexible word order
    r"ignore\s+(?:\w+\s+)*instructions",
    r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+",
    r"forget\s+(everything|all|your|previous)",
    r"you\s+are\s+now\b",
    r"pretend\s+(you\s+are|to\s+be)",
    r"act\s+as\s+(if\s+you|a\s+different|dan\b)",
    r"(new|override|change)\s+(instructions|rules|system\s+prompt)",
    r"disregard\s+(your|all|previous)",
    r"\bjailbreak\b",
    r"\bdan\s+mode\b",
    r"\bdan\b.{0,30}\bmode\b",
    r"reveal\s+(your|the)\s+(system\s+prompt|instructions)",
    r"print\s+(your|the)\s+(system\s+prompt|instructions)",
    r"do\s+anything\s+now",
    r"(as|act\s+as)\s+(an?\s+)?(unrestricted|unfiltered|evil)",
    r"forget\s+your\s+(role|rules|instructions|constraints)",
    r"new\s+persona",
]

_OFF_TOPIC_PATTERNS = [
    # Salary / compensation
    r"\b(salary|compensation|pay\s+range|pay\s+scale|wage)\s+(advice|benchmark|negotiation|range)\b",
    r"\bnegotiate\s+(my\s+)?(salary|compensation|offer|package)\b",
    r"\b(best|typical|average)\s+salary\b",
    r"\bhow\s+much\s+(should\s+I|to)\s+pay\b",
    # Legal
    r"\bhiring\s+law\b",
    r"\b(employment|discrimination|labor)\s+law\b",
    r"\bGDPR\b",
    # Competitor products
    r"\b(Mercer|Korn\s+Ferry|Hogan|Predictive\s+Index|Caliper|Gallup|Wonderlic)\b",
    # Completely off-topic
    r"\b(weather|recipe|cooking|sports|politics|religion|bitcoin|crypto|stock\s+price)\b",
    r"\bhow\s+to\s+(write|draft)\s+(a\s+)?(job\s+description|resume|CV)\b",
    r"\b(medical|health)\s+advice\b",
]

_COMPARISON_PATTERNS = [
    r"\b(compare|versus|vs\.?|difference\s+between|which\s+is\s+better)\b",
    r"\bwhat'?s?\s+(the\s+)?(difference|distinction)\b",
]

_REFINEMENT_PATTERNS = [
    r"\b(also\s+add|add|include|remove|exclude|drop|replace|swap)\b",
    r"\b(actually|instead|change|update|modify)\b",
    r"\bcan\s+you\s+(also|add|remove|change|include)\b",
]

_CLOSING_PATTERNS = [
    r"\b(thank\s+you|thanks|perfect|great)\b",
    r"\bthat'?s?\s+(all|it|great|perfect|what\s+I\s+needed)\b",
    r"\b(done|finished|looks?\s+good|bye|goodbye)\b",
    r"\bno\s+more\s+(questions|help)\b",
    r"\bi'?m?\s+(done|good|satisfied|happy\s+with)\b",
]

_COMPILED_INJECTION = [re.compile(p, re.I) for p in _INJECTION_PATTERNS]
_COMPILED_OFF_TOPIC = [re.compile(p, re.I) for p in _OFF_TOPIC_PATTERNS]
_COMPILED_COMPARISON = [re.compile(p, re.I) for p in _COMPARISON_PATTERNS]
_COMPILED_REFINEMENT = [re.compile(p, re.I) for p in _REFINEMENT_PATTERNS]
_COMPILED_CLOSING = [re.compile(p, re.I) for p in _CLOSING_PATTERNS]


def scope_guard(user_message: str) -> tuple[bool, str]:
    """
    Returns (is_in_scope: bool, refusal_message: str).
    refusal_message is empty string if in scope.
    """
    # Injection check first
    for pat in _COMPILED_INJECTION:
        if pat.search(user_message):
            return False, (
                "I'm the SHL Assessment Advisor and can only help with SHL assessment "
                "recommendations. How can I assist you with finding the right assessment "
                "for your hiring needs?"
            )

    # Off-topic check
    for pat in _COMPILED_OFF_TOPIC:
        if pat.search(user_message):
            return False, (
                "I can only help with SHL assessment recommendations. "
                "I'm not able to assist with that topic. "
                "Are you looking to assess candidates for a specific role? I can help with that!"
            )

    return True, ""


def _classify_query(user_message: str, messages: list[dict]) -> str:
    """
    Classify query type: injection | off_topic | closing | comparison |
    refinement | vague | specific
    """
    in_scope, _ = scope_guard(user_message)
    if not in_scope:
        for pat in _COMPILED_INJECTION:
            if pat.search(user_message):
                return "injection"
        return "off_topic"

    text = user_message.lower()

    if any(p.search(text) for p in _COMPILED_CLOSING):
        return "closing"

    if any(p.search(text) for p in _COMPILED_COMPARISON):
        return "comparison"

    has_prev_recs = _has_previous_recommendations(messages)
    if has_prev_recs and any(p.search(text) for p in _COMPILED_REFINEMENT):
        return "refinement"

    # Vague: less than 5 meaningful words
    words = re.findall(r"\b[a-zA-Z]{3,}\b", user_message)
    stop = {"the", "and", "for", "are", "with", "that", "this", "from", "have", "need", "want"}
    meaningful = [w for w in words if w.lower() not in stop]
    if len(meaningful) < 5:
        return "vague"

    return "specific"


def _has_previous_recommendations(messages: list[dict]) -> bool:
    """Check if any previous assistant message contained recommendations."""
    for msg in messages:
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            try:
                data = json.loads(content)
                if data.get("recommendations"):
                    return True
            except Exception:
                if "recommend" in content.lower():
                    return True
    return False


def _extract_context(messages: list[dict]) -> dict:
    """Extract job context from full conversation history."""
    all_user = " ".join(
        m.get("content", "") for m in messages if m.get("role") == "user"
    ).lower()

    return {
        "has_role": bool(re.search(
            r"\b(developer|engineer|manager|analyst|designer|sales|nurse|"
            r"java|python|data|software|hr|finance|marketing|operations)\b",
            all_user,
        )),
        "has_seniority": bool(re.search(
            r"\b(junior|senior|mid|entry|graduate|director|executive|intern)\b",
            all_user,
        )),
        "turn_count": sum(1 for m in messages if m.get("role") == "user"),
        "all_user_text": all_user,
    }


def _has_enough_context(ctx: dict) -> bool:
    """Return True if we have enough context to recommend."""
    return ctx["has_role"] and (ctx["has_seniority"] or ctx["turn_count"] >= 2)


def _build_search_query(query_type: str, user_message: str, ctx: dict) -> str:
    """Build an effective FAISS search query."""
    if query_type in ("comparison", "refinement"):
        return f"{ctx['all_user_text']} {user_message}"
    if ctx.get("has_role"):
        return ctx["all_user_text"]
    return user_message


# ============================================================
# LLM Client — Anthropic primary, Gemini fallback
# ============================================================

def _call_anthropic(system_prompt: str, model: str = "claude-sonnet-4-20250514") -> dict:
    """Call Anthropic Claude API."""
    import anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=1000,
        system=system_prompt,
        messages=[{"role": "user", "content": "Process the conversation above and respond."}],
    )
    raw = message.content[0].text
    return _parse_llm_response(raw)


def _call_gemini(system_prompt: str) -> dict:
    """Call Google Gemini API as fallback."""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")

    try:
        # Try new google.genai package first
        import google.genai as genai
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=system_prompt,
        )
        raw = response.text
    except ImportError:
        # Fall back to deprecated google.generativeai
        import google.generativeai as genai_old  # type: ignore
        genai_old.configure(api_key=api_key)
        model = genai_old.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(system_prompt)
        raw = response.text

    return _parse_llm_response(raw)


def _parse_llm_response(raw: str) -> dict:
    """Parse JSON from LLM response with error handling."""
    if not raw:
        return _error_dict("Empty response from LLM")

    text = raw.strip()
    # Strip markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    text = text.strip()

    # Extract JSON object
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        text = match.group(0)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}. Raw: {text[:300]}")
        return _error_dict(f"JSON parse error: {e}")

    # Ensure required keys
    if "reply" not in data:
        data["reply"] = "I encountered an issue. Please try again."
    if "recommendations" not in data:
        data["recommendations"] = []
    if "end_of_conversation" not in data:
        data["end_of_conversation"] = False

    # Normalize recommendations
    recs = data.get("recommendations", [])
    if not isinstance(recs, list):
        data["recommendations"] = []
    else:
        valid = []
        for r in recs[:10]:
            if isinstance(r, dict) and r.get("name") and r.get("url"):
                valid.append({
                    "name": str(r.get("name", "")),
                    "url": str(r.get("url", "")),
                    "test_type": str(r.get("test_type", "Assessment")),
                })
        data["recommendations"] = valid

    data["end_of_conversation"] = bool(data.get("end_of_conversation", False))
    return data


def _error_dict(msg: str) -> dict:
    logger.error(f"LLM error: {msg}")
    return {
        "reply": "I'm experiencing technical difficulties. Please try again.",
        "recommendations": [],
        "end_of_conversation": False,
    }


def _call_groq(system_prompt: str) -> dict:
    """Call Groq API (fast, free tier available)."""
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")
    from groq import Groq
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": system_prompt}],
        max_tokens=1000,
        temperature=0.3,
    )
    raw = response.choices[0].message.content
    return _parse_llm_response(raw)


def _call_openai(system_prompt: str) -> dict:
    """Call OpenAI API."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": system_prompt}],
        max_tokens=1000,
        temperature=0.3,
    )
    raw = response.choices[0].message.content
    return _parse_llm_response(raw)


def call_llm(system_prompt: str) -> dict:
    """Call LLM: Anthropic → Gemini → Groq → OpenAI → safe error."""
    providers = [
        ("Anthropic", _call_anthropic),
        ("Gemini",    _call_gemini),
        ("Groq",      _call_groq),
        ("OpenAI",    _call_openai),
    ]
    for name, fn in providers:
        try:
            result = fn(system_prompt)
            logger.info(f"LLM call succeeded via {name}")
            return result
        except Exception as e:
            logger.warning(f"{name} failed: {e}")

    return _error_dict("All LLM providers failed — please set at least one API key")


# ============================================================
# Recommendation Validator
# ============================================================

def _validate_recommendations(raw_recs: list[dict], retrieved: list[dict]) -> list[Recommendation]:
    """
    Only accept recommendations that exist in the retrieved catalog items.
    Prevents hallucination.
    """
    # Build lookup maps from retrieved items
    retrieved_by_name: dict[str, dict] = {}
    retrieved_by_url: dict[str, dict] = {}
    for r in retrieved:
        a = r.get("assessment", r)
        name = a.get("name", "").lower()
        url = a.get("url", "")
        if name:
            retrieved_by_name[name] = a
        if url:
            retrieved_by_url[url] = a

    valid = []
    for rec in raw_recs:
        rec_name = rec.get("name", "").lower()
        rec_url = rec.get("url", "")

        matched = None

        # Match by name (partial)
        for cat_name, item in retrieved_by_name.items():
            if rec_name in cat_name or cat_name in rec_name:
                matched = item
                break

        # Match by URL
        if not matched and rec_url in retrieved_by_url:
            matched = retrieved_by_url[rec_url]

        if matched:
            valid.append(Recommendation(
                name=matched.get("name", rec.get("name", "")),
                url=matched.get("url", rec.get("url", "")),
                test_type=matched.get("test_type", rec.get("test_type", "Assessment")),
            ))
        else:
            logger.warning(f"Rejected non-catalog recommendation: {rec.get('name')} / {rec.get('url')}")

    return valid[:10]


# ============================================================
# Main Agent
# ============================================================

class RecommendationAgent:
    """
    Stateless recommendation agent.
    All context is derived from the request messages.
    """

    def __init__(self, retriever: Retriever):
        self.retriever = retriever

    def process(self, request: ChatRequest) -> ChatResponse:
        """Process a ChatRequest and return a ChatResponse."""
        messages = [m.model_dump() for m in request.messages]

        # Trim to last 16 messages if history is very long
        if len(messages) > 20:
            messages = messages[-16:]

        user_message = messages[-1].get("content", "")

        logger.info(f"Processing turn {len(messages)}, query: '{user_message[:80]}'")

        # ---- Step 1: Scope guard ----
        in_scope, refusal = scope_guard(user_message)
        if not in_scope:
            return ChatResponse(reply=refusal, recommendations=[], end_of_conversation=False)

        # ---- Step 2: Classify query ----
        query_type = _classify_query(user_message, messages)
        logger.info(f"Query type: {query_type}")

        # ---- Step 3: Extract context ----
        ctx = _extract_context(messages)
        turn_count = ctx["turn_count"]

        # ---- Step 4: Select instructions ----
        if query_type == "closing":
            instructions = CLOSE_PROMPT
        elif query_type == "comparison":
            instructions = COMPARE_PROMPT
        elif query_type == "refinement":
            instructions = REFINE_PROMPT
        elif query_type == "vague" or not _has_enough_context(ctx):
            # Force recommendation after turn 6
            if turn_count >= 6:
                instructions = FORCE_RECOMMEND_PROMPT
            else:
                instructions = CLARIFY_PROMPT
        else:
            instructions = RECOMMEND_PROMPT

        # ---- Step 5: Retrieve assessments ----
        retrieved = []
        if query_type != "closing":
            search_q = _build_search_query(query_type, user_message, ctx)
            top_k = 15 if query_type != "comparison" else 5
            retrieved = self.retriever.search(search_q, top_k=top_k)

        logger.info(f"Retrieved {len(retrieved)} candidates")

        # ---- Step 6: Build prompt ----
        catalog_summary = format_catalog_summary(self.retriever.get_all())
        retrieved_text = format_retrieved_assessments(retrieved)
        history_text = format_conversation_history(messages)

        system_prompt = build_system_prompt(
            catalog_summary=catalog_summary,
            retrieved_assessments=retrieved_text,
            conversation_history=history_text,
            user_message=user_message,
            instructions=instructions,
        )

        # ---- Step 7: Call LLM ----
        try:
            llm_data = call_llm(system_prompt)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return ChatResponse(
                reply="I'm experiencing technical difficulties. Please try again.",
                recommendations=[],
                end_of_conversation=False,
            )

        # ---- Step 8: Validate recommendations ----
        raw_recs = llm_data.get("recommendations", [])
        valid_recs = _validate_recommendations(raw_recs, retrieved)

        # Determine end_of_conversation
        eoc = bool(llm_data.get("end_of_conversation", False))
        if query_type == "closing":
            eoc = True

        reply = llm_data.get("reply", "")
        if not reply:
            reply = "I'm here to help you find the right SHL assessment. Please tell me about the role."

        return ChatResponse(
            reply=reply,
            recommendations=valid_recs,
            end_of_conversation=eoc,
        )


# ============================================================
# Module-level singleton
# ============================================================
_agent: Optional[RecommendationAgent] = None


def get_agent(
    catalog_path: str = "./catalog.json",
    index_path: str = "./faiss_index",
) -> RecommendationAgent:
    """Get or create the singleton agent."""
    global _agent
    if _agent is None:
        retriever = get_retriever(catalog_path=catalog_path, index_path=index_path)
        _agent = RecommendationAgent(retriever=retriever)
    return _agent
