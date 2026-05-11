"""
SHL Assessment Recommendation System
=====================================
Pydantic models for request/response schemas.
All schemas are NON-NEGOTIABLE as per assignment requirements.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator


# ============================================================
# Conversation Role Enum
# ============================================================
class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


# ============================================================
# Conversation Message
# ============================================================
class Message(BaseModel):
    """Single message in the conversation history."""

    role: MessageRole = Field(..., description="Message role: user or assistant")
    content: str = Field(
        ...,
        min_length=1,
        max_length=8192,
        description="Message content",
    )

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Message content cannot be blank")
        return v.strip()


# ============================================================
# Chat Request — POST /chat
# ============================================================
class ChatRequest(BaseModel):
    """
    STATELESS chat request.
    Full conversation history must be included on every request.
    """

    messages: list[Message] = Field(
        ...,
        min_length=1,
        max_length=50,  # prevent abuse / extremely long histories
        description="Full conversation history (stateless — must include all turns)",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "messages": [
                    {"role": "user", "content": "I need to hire a Java developer"}
                ]
            }
        }


# ============================================================
# Assessment Recommendation Item
# ============================================================
class RecommendationItem(BaseModel):
    """
    Single assessment recommendation from SHL catalog.
    ONLY real catalog data — no hallucinations.
    """

    name: str = Field(..., description="Official SHL assessment name")
    url: str = Field(..., description="Official SHL product URL")
    test_type: str = Field(..., description="Assessment type (e.g., Cognitive, Personality)")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Verify G+ (Global Skills Assessment)",
                "url": "https://www.shl.com/solutions/products/product-catalog/view/verify-g-plus/",
                "test_type": "Cognitive Ability",
            }
        }


# ============================================================
# Chat Response — NON-NEGOTIABLE SCHEMA
# ============================================================
class ChatResponse(BaseModel):
    """
    EXACT response schema as per assignment requirements.
    DO NOT MODIFY THIS SCHEMA.
    """

    reply: str = Field(..., description="Agent's natural language reply")
    recommendations: list[RecommendationItem] = Field(
        default_factory=list,
        description="List of 0–10 SHL assessments. Empty while gathering info.",
    )
    end_of_conversation: bool = Field(
        default=False,
        description="True only when the recommendation task is fully complete",
    )

    @field_validator("recommendations")
    @classmethod
    def validate_recommendation_count(
        cls, v: list[RecommendationItem]
    ) -> list[RecommendationItem]:
        if len(v) > 10:
            raise ValueError("Cannot recommend more than 10 assessments")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "reply": "Based on your requirements for a senior Java developer, here are the top SHL assessments:",
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


# ============================================================
# Health Response — GET /health
# ============================================================
class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(default="ok", description="Service health status")

    class Config:
        json_schema_extra = {"example": {"status": "ok"}}


# ============================================================
# Internal: SHL Assessment Catalog Item
# ============================================================
class SHLAssessment(BaseModel):
    """Internal representation of a scraped SHL assessment."""

    name: str
    url: str
    description: str = ""
    test_type: str = ""
    category: str = ""
    skills_measured: list[str] = Field(default_factory=list)
    duration: Optional[str] = None
    remote_testing: bool = False
    adaptive: bool = False
    tags: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    raw_text: str = ""  # Combined text for embedding

    def to_embedding_text(self) -> str:
        """Generate combined text for embedding generation."""
        parts = [
            self.name,
            self.description,
            self.test_type,
            self.category,
            " ".join(self.skills_measured),
            " ".join(self.tags),
            " ".join(self.keywords),
        ]
        return " | ".join(p for p in parts if p)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    class Config:
        json_schema_extra = {
            "example": {
                "name": "OPQ32",
                "url": "https://www.shl.com/solutions/products/product-catalog/view/opq32/",
                "description": "The OPQ32 is an occupational personality questionnaire...",
                "test_type": "Personality",
                "category": "Personality & Behavior",
                "skills_measured": ["teamwork", "leadership", "communication"],
                "duration": "25-40 minutes",
                "remote_testing": True,
                "adaptive": False,
                "tags": ["personality", "behavior", "leadership"],
                "keywords": ["OPQ", "personality questionnaire", "occupational"],
            }
        }


# ============================================================
# Internal: Retrieval Result
# ============================================================
class RetrievalResult(BaseModel):
    """Internal result from vector store retrieval."""

    assessment: SHLAssessment
    score: float = Field(ge=0.0, le=1.0, description="Semantic similarity score")
    rank: int = Field(ge=1, description="Rank in result list")
