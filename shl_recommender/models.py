"""
models.py — Pydantic Request/Response Models
=============================================
NON-NEGOTIABLE schema as required by the automated evaluator.
Do NOT change any field names or types.
"""
from __future__ import annotations

from typing import List
from pydantic import BaseModel, Field, field_validator, ConfigDict


# ============================================================
# Message
# ============================================================
class Message(BaseModel):
    """Single conversation message."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {"role": "user", "content": "I need to hire a Java developer"}
        }
    )

    role: str = Field(..., description="Must be 'user' or 'assistant'")
    content: str = Field(..., min_length=1, description="Message content")

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v not in ("user", "assistant", "system"):
            raise ValueError(f"role must be 'user' or 'assistant', got: {v}")
        return v

    @field_validator("content")
    @classmethod
    def content_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("content cannot be blank")
        return v


# ============================================================
# ChatRequest — POST /chat
# ============================================================
class ChatRequest(BaseModel):
    """
    Stateless chat request.
    Full conversation history must be included on every call.
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "messages": [
                    {"role": "user", "content": "I need to hire a Java developer"}
                ]
            }
        }
    )

    messages: List[Message] = Field(
        ...,
        description="Full conversation history (stateless — include all turns)",
    )


# ============================================================
# Recommendation — single catalog item
# ============================================================
class Recommendation(BaseModel):
    """
    Single assessment recommendation.
    ALL fields must come from catalog.json — no hallucination.
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "Verify G+ (Global Skills Assessment)",
                "url": "https://www.shl.com/solutions/products/product-catalog/view/verify-g-plus/",
                "test_type": "A",
            }
        }
    )

    name: str = Field(..., description="Exact assessment name from SHL catalog")
    url: str = Field(..., description="Exact URL from SHL catalog")
    test_type: str = Field(..., description="Assessment type code(s), e.g. 'K', 'P', 'A,K'")


# ============================================================
# ChatResponse — NON-NEGOTIABLE SCHEMA
# ============================================================
class ChatResponse(BaseModel):
    """
    EXACT response schema required by the automated evaluator.
    DO NOT MODIFY FIELD NAMES OR TYPES.
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "reply": "Based on your requirements, here are the top SHL assessments:",
                "recommendations": [
                    {
                        "name": "Verify G+ (Global Skills Assessment)",
                        "url": "https://www.shl.com/solutions/products/product-catalog/view/verify-g-plus/",
                        "test_type": "A",
                    }
                ],
                "end_of_conversation": False,
            }
        }
    )

    reply: str = Field(..., description="Agent's conversational text response")
    recommendations: List[Recommendation] = Field(
        default_factory=list,
        description="0–10 SHL assessments from catalog. Empty while gathering context.",
    )
    end_of_conversation: bool = Field(
        default=False,
        description="True only when the recommendation task is fully complete",
    )

    @field_validator("recommendations")
    @classmethod
    def max_ten_recommendations(cls, v: List[Recommendation]) -> List[Recommendation]:
        if len(v) > 10:
            return v[:10]
        return v


# ============================================================
# HealthResponse
# ============================================================
class HealthResponse(BaseModel):
    """Health check response."""
    model_config = ConfigDict(json_schema_extra={"example": {"status": "ok"}})

    status: str = Field(default="ok")
