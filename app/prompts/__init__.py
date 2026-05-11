"""Prompts package init."""
from app.prompts.templates import (
    SYSTEM_PROMPT,
    INSTRUCTIONS_CLARIFY,
    INSTRUCTIONS_RECOMMEND,
    INSTRUCTIONS_REFINE,
    INSTRUCTIONS_COMPARE,
    INSTRUCTIONS_REFUSE,
    INSTRUCTIONS_CLOSE,
    format_catalog_context,
    format_conversation_history,
    build_system_prompt,
)

__all__ = [
    "SYSTEM_PROMPT",
    "INSTRUCTIONS_CLARIFY",
    "INSTRUCTIONS_RECOMMEND",
    "INSTRUCTIONS_REFINE",
    "INSTRUCTIONS_COMPARE",
    "INSTRUCTIONS_REFUSE",
    "INSTRUCTIONS_CLOSE",
    "format_catalog_context",
    "format_conversation_history",
    "build_system_prompt",
]
