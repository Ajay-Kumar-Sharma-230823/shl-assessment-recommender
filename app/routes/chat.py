"""
FastAPI Routes
================
Implements EXACTLY TWO endpoints:
1. GET /health → {status: "ok"}
2. POST /chat → ChatResponse schema
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from app.models.schemas import ChatRequest, ChatResponse, HealthResponse
from app.services.startup import get_agent

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================
# GET /health
# ============================================================
@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=200,
    summary="Health Check",
    description="Returns system health status.",
    tags=["System"],
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    
    Returns:
        {"status": "ok"}
    """
    return HealthResponse(status="ok")


# ============================================================
# POST /chat
# ============================================================
@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=200,
    summary="Conversational SHL Assessment Recommendation",
    description=(
        "Stateless conversational endpoint. Send full conversation history on every request. "
        "Returns assessment recommendations, clarifying questions, or comparisons. "
        "Recommendations are ONLY from the official SHL catalog."
    ),
    tags=["Chat"],
    responses={
        200: {
            "description": "Successful response",
            "content": {
                "application/json": {
                    "example": {
                        "reply": "Based on your requirements for a Java developer, here are the top SHL assessments...",
                        "recommendations": [
                            {
                                "name": "Verify G+ (Global Skills Assessment)",
                                "url": "https://www.shl.com/solutions/products/product-catalog/view/verify-g-plus/",
                                "test_type": "Cognitive Ability",
                            }
                        ],
                        "end_of_conversation": False,
                    }
                }
            },
        },
        422: {"description": "Validation error — check request format"},
        500: {"description": "Internal server error"},
    },
)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Stateless conversational SHL assessment recommendation endpoint.
    
    The FULL conversation history must be included in every request.
    The server does NOT store any session state.
    
    Flow:
    1. Validates input
    2. Checks security (injection, off-topic)  
    3. Extracts context from history
    4. Retrieves relevant SHL assessments from vector store
    5. Generates grounded response via LLM
    6. Validates recommendations against catalog
    7. Returns structured response
    
    Args:
        request: ChatRequest with full messages list
        
    Returns:
        ChatResponse with reply, recommendations (0-10), end_of_conversation flag
    """
    try:
        agent = get_agent()
        response = agent.process(request)
        logger.info(
            f"Chat response — recommendations: {len(response.recommendations)}, "
            f"eoc: {response.end_of_conversation}"
        )
        return response

    except ValueError as e:
        logger.warning(f"Validation error in chat: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in chat: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error. Please try again.",
        )
