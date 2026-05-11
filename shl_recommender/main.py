"""
main.py — FastAPI Application Entry Point
==========================================
Provides exactly two endpoints:
  GET  /health  → {"status": "ok"}
  POST /chat    → ChatResponse

Stateless: no session storage.
CORS enabled for all origins.
"""
from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ---- Path setup so imports work whether run from shl_recommender/ or parent ----
_THIS_DIR = Path(__file__).parent.resolve()
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from models import ChatRequest, ChatResponse, HealthResponse
from agent import get_agent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================
# Catalog / index paths (relative to this file's location)
# ============================================================
CATALOG_PATH = str(_THIS_DIR / "catalog.json")
INDEX_PATH = str(_THIS_DIR / "faiss_index")


# ============================================================
# Lifespan — load agent once at startup
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("SHL Assessment Recommender — Starting Up")
    logger.info("=" * 60)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.warning("⚠️  ANTHROPIC_API_KEY not set — LLM calls will fail!")

    try:
        agent = get_agent(catalog_path=CATALOG_PATH, index_path=INDEX_PATH)
        logger.info(f"✅ Agent ready — catalog size: {agent.retriever.catalog_size}")
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")

    yield

    logger.info("SHL Assessment Recommender — Shutting Down")


# ============================================================
# FastAPI App
# ============================================================
app = FastAPI(
    title="SHL Assessment Recommender",
    description=(
        "Conversational AI agent that helps recruiters find the right SHL assessments. "
        "All recommendations are grounded in the official SHL product catalog."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# ============================================================
# CORS — allow all origins (evaluator calls from external server)
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# GET /health
# ============================================================
@app.get(
    "/health",
    response_model=HealthResponse,
    status_code=200,
    summary="Health Check",
    tags=["System"],
)
async def health_check() -> HealthResponse:
    """Returns {"status": "ok"} within 1 second."""
    return HealthResponse(status="ok")


# ============================================================
# POST /chat
# ============================================================
@app.post(
    "/chat",
    response_model=ChatResponse,
    status_code=200,
    summary="Conversational SHL Assessment Recommendation",
    tags=["Chat"],
)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Stateless chat endpoint. Send the FULL conversation history on every call.
    Returns reply, recommendations (0-10 items from SHL catalog), and end_of_conversation flag.
    """
    try:
        # Handle empty messages gracefully (schema allows min_length but guard anyway)
        if not request.messages:
            return ChatResponse(
                reply="Hello! I'm the SHL Assessment Advisor. How can I help you find the right assessment?",
                recommendations=[],
                end_of_conversation=False,
            )

        agent = get_agent(catalog_path=CATALOG_PATH, index_path=INDEX_PATH)
        response = agent.process(request)

        logger.info(
            f"Chat response: {len(response.recommendations)} recommendations, "
            f"eoc={response.end_of_conversation}"
        )
        return response

    except Exception as e:
        logger.error(f"Unexpected error in /chat: {e}", exc_info=True)
        return ChatResponse(
            reply="I'm experiencing technical difficulties. Please try again in a moment.",
            recommendations=[],
            end_of_conversation=False,
        )


# ============================================================
# Root — redirect hint
# ============================================================
@app.get("/", include_in_schema=False)
async def root():
    return {
        "name": "SHL Assessment Recommender",
        "version": "1.0.0",
        "endpoints": {"health": "GET /health", "chat": "POST /chat", "docs": "GET /docs"},
    }


# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        workers=1,
    )
