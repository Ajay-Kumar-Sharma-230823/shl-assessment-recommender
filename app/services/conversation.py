"""
Conversation Manager (Stateless)
==================================
Manages conversation state WITHIN a single request only.
No server-side session storage.

Each request contains the FULL conversation history.
The manager extracts context, determines state, and builds the prompt.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Number of turns to consider for "enough context"
CONTEXT_THRESHOLD_TURNS = 2  # At least 1 Q&A exchange


@dataclass
class ConversationContext:
    """
    Extracted context from conversation history.
    Used to inform retrieval and response generation.
    """
    # Job/role context
    role: str = ""
    seniority: str = ""
    department: str = ""

    # Requirements
    needs_coding: Optional[bool] = None
    needs_personality: Optional[bool] = None
    needs_cognitive: Optional[bool] = None
    needs_language: Optional[bool] = None
    needs_leadership: Optional[bool] = None
    needs_remote: Optional[bool] = None
    needs_adaptive: Optional[bool] = None

    # Constraints
    max_duration_minutes: Optional[int] = None
    
    # Previous recommendations (names from assistant messages)
    previous_recommendation_names: list[str] = field(default_factory=list)
    previous_recommendation_urls: list[str] = field(default_factory=list)

    # Refinement info
    refinements: list[str] = field(default_factory=list)

    # Turn count
    turn_count: int = 0

    # Summary of what's been established
    established_context: str = ""

    def has_enough_context(self) -> bool:
        """Return True if we have enough info to make recommendations."""
        has_role = bool(self.role or self.department)
        has_requirement = any([
            self.needs_coding is not None,
            self.needs_personality is not None,
            self.needs_cognitive is not None,
            self.needs_language is not None,
            self.needs_leadership is not None,
            self.seniority,
        ])
        # After 2+ turns, we can usually recommend
        if self.turn_count >= CONTEXT_THRESHOLD_TURNS and has_role:
            return True
        return has_role and has_requirement

    def build_search_query(self) -> str:
        """Build an effective semantic search query from extracted context."""
        parts = []

        if self.role:
            parts.append(self.role)
        if self.seniority:
            parts.append(self.seniority)
        if self.department:
            parts.append(self.department)

        if self.needs_cognitive:
            parts.append("cognitive ability reasoning")
        if self.needs_personality:
            parts.append("personality assessment behavior")
        if self.needs_coding:
            parts.append("coding programming technical skills")
        if self.needs_language:
            parts.append("language communication verbal")
        if self.needs_leadership:
            parts.append("leadership management competency")
        if self.needs_remote:
            parts.append("remote online testing")

        if self.refinements:
            parts.extend(self.refinements[-2:])  # Recent refinements

        return " ".join(parts) if parts else "SHL assessment"

    def to_summary(self) -> str:
        """Build a human-readable context summary."""
        lines = []
        if self.role:
            lines.append(f"Role: {self.role}")
        if self.seniority:
            lines.append(f"Seniority: {self.seniority}")
        if self.needs_coding is not None:
            lines.append(f"Coding required: {'Yes' if self.needs_coding else 'No'}")
        if self.needs_personality is not None:
            lines.append(f"Personality assessment: {'Yes' if self.needs_personality else 'No'}")
        if self.needs_cognitive is not None:
            lines.append(f"Cognitive test: {'Yes' if self.needs_cognitive else 'No'}")
        if self.needs_remote is not None:
            lines.append(f"Remote testing: {'Yes' if self.needs_remote else 'No'}")
        if self.max_duration_minutes:
            lines.append(f"Max duration: {self.max_duration_minutes} minutes")
        return "; ".join(lines) if lines else "Limited context gathered"


class ConversationManager:
    """
    Stateless conversation manager.
    Extracts context from full conversation history on each request.
    """

    # Role extraction patterns
    ROLE_PATTERNS = [
        r"\b(java|python|javascript|react|angular|vue|node|\.net|c\+\+|golang|rust|ios|android)\b",
        r"\b(developer|engineer|programmer|architect|devops|sre|qa|tester)\b",
        r"\b(manager|director|vp|ceo|cto|cfo|head\s+of|lead|principal|staff)\b",
        r"\b(analyst|scientist|researcher|consultant|specialist|coordinator)\b",
        r"\b(sales|marketing|hr|finance|legal|accounting|operations)\b",
        r"\b(nurse|doctor|pharmacist|teacher|professor|driver|pilot)\b",
        r"\b(customer\s+service|support|customer\s+success)\b",
        r"\b(graduate|intern|entry.level|junior|senior|mid.level)\b",
    ]

    SENIORITY_PATTERNS = {
        "entry": r"\b(junior|entry.level|entry|graduate|intern|fresher|new\s+grad)\b",
        "mid": r"\b(mid.level|mid|intermediate|experienced)\b",
        "senior": r"\b(senior|sr\.?|lead|principal|staff|expert)\b",
        "executive": r"\b(executive|vp|director|head\s+of|c-level|c\s+suite|ceo|cto|cfo|coo)\b",
    }

    def extract_context(self, messages: list[dict]) -> ConversationContext:
        """
        Extract conversation context from full message history.
        
        Processes ALL messages to build cumulative understanding.
        """
        ctx = ConversationContext()
        ctx.turn_count = sum(1 for m in messages if m.get("role") == "user")

        # Combine all user messages for analysis
        all_user_text = " ".join(
            m.get("content", "") for m in messages if m.get("role") == "user"
        ).lower()

        # Extract role
        for pattern in self.ROLE_PATTERNS:
            match = re.search(pattern, all_user_text, re.IGNORECASE)
            if match:
                if not ctx.role:
                    ctx.role = match.group(0).strip()
                else:
                    ctx.role += f" {match.group(0).strip()}"

        # Extract seniority
        for level, pattern in self.SENIORITY_PATTERNS.items():
            if re.search(pattern, all_user_text, re.IGNORECASE):
                ctx.seniority = level
                break

        # Extract requirements
        ctx.needs_coding = self._extract_bool_need(
            all_user_text,
            positive=["coding", "programming", "technical", "code", "algorithm", "software"],
            negative=["no coding", "not coding", "non-technical", "without coding"],
        )

        ctx.needs_personality = self._extract_bool_need(
            all_user_text,
            positive=["personality", "behavior", "behaviour", "culture", "fit", "soft skills", "opq"],
            negative=["no personality", "not personality"],
        )

        ctx.needs_cognitive = self._extract_bool_need(
            all_user_text,
            positive=["cognitive", "reasoning", "aptitude", "intelligence", "iq", "numerical", "verbal", "logical"],
            negative=["no cognitive", "not cognitive"],
        )

        ctx.needs_language = self._extract_bool_need(
            all_user_text,
            positive=["language", "english", "communication", "verbal", "grammar", "writing"],
            negative=["no language"],
        )

        ctx.needs_leadership = self._extract_bool_need(
            all_user_text,
            positive=["leadership", "management", "leader", "manager", "team lead", "executive"],
            negative=[],
        )

        ctx.needs_remote = self._extract_bool_need(
            all_user_text,
            positive=["remote", "online", "virtual", "work from home", "wfh", "distributed"],
            negative=["no remote", "in-person", "on-site"],
        )

        # Duration
        duration_match = re.search(r"(\d+)\s*(?:minute|min)", all_user_text)
        if duration_match:
            ctx.max_duration_minutes = int(duration_match.group(1))

        # Extract refinements (from recent messages)
        recent_messages = messages[-4:] if len(messages) >= 4 else messages
        refinement_text = " ".join(
            m.get("content", "") for m in recent_messages if m.get("role") == "user"
        )
        if re.search(r"\b(also|additionally|add|include|but|however|change|update)\b", refinement_text, re.I):
            ctx.refinements.append(refinement_text[:200])

        # Extract previous recommendations from assistant messages
        for msg in messages:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                try:
                    data = json.loads(content)
                    for rec in data.get("recommendations", []):
                        if rec.get("name"):
                            ctx.previous_recommendation_names.append(rec["name"])
                        if rec.get("url"):
                            ctx.previous_recommendation_urls.append(rec["url"])
                except (json.JSONDecodeError, AttributeError, TypeError):
                    pass

        # Build established context summary
        ctx.established_context = ctx.to_summary()

        logger.debug(f"Extracted context: {ctx.established_context}")
        return ctx

    def _extract_bool_need(
        self,
        text: str,
        positive: list[str],
        negative: list[str],
    ) -> Optional[bool]:
        """
        Extract True/False/None for a boolean need from text.
        None means not mentioned.
        """
        # Check negatives first
        for neg_phrase in negative:
            if neg_phrase in text:
                return False

        # Check positives
        for pos_phrase in positive:
            if pos_phrase in text:
                return True

        return None

    def get_conversation_summary(self, messages: list[dict]) -> str:
        """
        Generate a summary of the conversation for context compression.
        Used when conversation history is long.
        """
        ctx = self.extract_context(messages)
        user_messages = [m["content"] for m in messages if m.get("role") == "user"]

        summary_parts = [
            f"Conversation with {ctx.turn_count} user turn(s).",
            f"Context: {ctx.established_context}",
        ]

        if ctx.previous_recommendation_names:
            unique_names = list(dict.fromkeys(ctx.previous_recommendation_names))
            summary_parts.append(
                f"Previously recommended: {', '.join(unique_names[:5])}"
            )

        if ctx.refinements:
            summary_parts.append(f"Refinements requested: {ctx.refinements[-1][:100]}")

        return " | ".join(summary_parts)

    def should_compress_history(self, messages: list[dict]) -> bool:
        """Return True if the conversation history is too long and needs compression."""
        total_chars = sum(len(m.get("content", "")) for m in messages)
        return total_chars > 8000 or len(messages) > 12


# Singleton instance
_manager = ConversationManager()


def get_conversation_manager() -> ConversationManager:
    """Get the singleton ConversationManager."""
    return _manager
