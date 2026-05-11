"""
Security Module
================
Implements:
- Prompt injection detection and prevention
- Input validation and sanitization
- Malicious instruction filtering
- Scope enforcement
- Off-topic query detection
"""
from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# ============================================================
# Query Classification
# ============================================================
class QueryType(str, Enum):
    VAGUE = "vague"                    # Needs clarification
    SPECIFIC = "specific"              # Has enough info to recommend
    COMPARISON = "comparison"          # Comparing assessments
    REFINEMENT = "refinement"          # Refining previous recommendations
    OFF_TOPIC = "off_topic"            # Outside scope
    CLOSING = "closing"               # User is done
    PROMPT_INJECTION = "prompt_injection"  # Malicious input


# ============================================================
# Prompt Injection Patterns
# ============================================================
INJECTION_PATTERNS = [
    # Classic injection attempts
    r"ignore\s+(previous|prior|above|all)\s+instructions",
    r"ignore\s+all\s+(previous|prior|above)\s+instructions",
    r"forget\s+(everything|all|your|previous)\s+(you know|instructions|rules|constraints)",
    r"you are now",
    r"pretend\s+(you are|to be|that)",
    r"act\s+as\s+(if\s+you\s+are|a\s+different)",
    r"(new|override|change)\s+(instructions|rules|system\s+prompt)",
    r"disregard\s+(your|all|previous)",
    r"(jailbreak|bypass|override|hack)\s+(your|the)\s+(rules|instructions|system)",
    r"you are (actually|really|secretly)",
    r"reveal\s+(your|the)\s+(system\s+prompt|instructions|prompt)",
    r"repeat\s+.{0,30}above",
    r"print\s+(your|the)\s+(system\s+prompt|instructions)",
    r"tell\s+me\s+(your|the)\s+(system|hidden)\s+(prompt|instructions)",
    r"what\s+(are\s+your|is\s+your)\s+(system\s+)?prompt",
    # Role-playing attacks
    r"as\s+(a|an)\s+(unrestricted|unfiltered|evil|dangerous)",
    r"do\s+anything\s+now",
    r"dan\s+(mode|prompt)",
    # DAN and similar
    r"\bdan\b",
    r"\bjailbreak\b",
]

COMPILED_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS
]

# ============================================================
# Off-Topic Patterns
# ============================================================
OFF_TOPIC_PATTERNS = [
    r"\blegal\s+(advice|counsel|questions?|requirements?)\b",
    r"\b(salary|compensation|pay|wage)\s+(advice|range|benchmark|recommendation|guidance)\b",
    r"\bwhat\s+(salary|compensation|pay|wage)\b",
    r"\bhow\s+much\s+should\s+i\s+(pay|offer)\b",
    r"\bdrug\s+testing\b",
    r"\bhiring\s+law\b",
    r"\bdiscrimination\b",
    r"\blabor\s+(law|regulation)\b",
    r"\bwrite\s+(me\s+a|an?\s+)(essay|code|script|email)\b",
    r"\b(python|javascript|java|coding)\s+(help|tutorial|homework)\b",
    r"\b(weather|news|sports|recipe)\b",
    r"\b(stock|crypto|bitcoin|investment)\s+(advice|price|prediction)\b",
    r"\b(medical|health|doctor|diagnosis)\s+advice\b",
    r"\bhow\s+to\s+hack\b",
    r"\bpolitics?\b",
    r"\breligion\b",
]

COMPILED_OFF_TOPIC_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in OFF_TOPIC_PATTERNS
]

# ============================================================
# Comparison Patterns
# ============================================================
COMPARISON_PATTERNS = [
    r"\b(compare|comparison|versus|vs\.?|difference\s+between|which\s+is\s+better)\b",
    r"\b(between|among)\s+.{3,50}\s+and\s+.{3,50}\b",
    r"\bwhat'?s?\s+(the\s+)?(difference|distinction)\b",
]

COMPILED_COMPARISON_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in COMPARISON_PATTERNS
]

# ============================================================
# Refinement Patterns
# ============================================================
REFINEMENT_PATTERNS = [
    r"\b(actually|wait|instead|change|update|modify|also\s+add|remove|exclude|include)\b",
    r"\b(no|not)\s+(that|those|the\s+coding|the\s+personality)\b",
    r"\b(add|drop|replace|swap)\s+(the|a|an)?\s*\w+\s+(test|assessment)\b",
    r"\bcan\s+you\s+(also|instead|change|update|add|remove)\b",
    r"\bmake\s+(it|them)\s+(remote|shorter|longer|more|less)\b",
]

COMPILED_REFINEMENT_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in REFINEMENT_PATTERNS
]

# ============================================================
# Closing Patterns
# ============================================================
CLOSING_PATTERNS = [
    r"\b(thank\s+you|thanks|perfect|great|that'?s?\s+(all|it|helpful|perfect|what\s+I\s+need))\b",
    r"\b(done|finished|that'?s?\s+(enough|all|great))\b",
    r"\b(no\s+more\s+(questions|help)|i'?m?\s+(done|good|satisfied))\b",
    r"\b(bye|goodbye|see\s+you)\b",
]

COMPILED_CLOSING_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in CLOSING_PATTERNS
]

# ============================================================
# Vague Query Patterns
# ============================================================
VAGUE_INDICATORS = [
    r"^(i\s+need|we\s+need|hiring|need\s+to\s+hire|looking\s+for)\s+\w{1,15}\s*$",
    r"^(give\s+me|suggest|recommend)\s+(some|a\s+few)?\s*(assessments?|tests?)\s*$",
    r"^what\s+assessments?\s+(do\s+you\s+have|are\s+available)\s*\??$",
    r"^(help|help\s+me)\s*$",
    r"^(assessment|test)\s*(please|needed|required)?\s*$",
]

COMPILED_VAGUE_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in VAGUE_INDICATORS
]

# Minimum context threshold (words beyond stop words)
MIN_CONTEXT_WORDS = 5


# ============================================================
# Security Checks
# ============================================================
class SecurityGuard:
    """
    Comprehensive security guard for input validation,
    prompt injection detection, and scope enforcement.
    """

    def check_prompt_injection(self, text: str) -> bool:
        """Return True if prompt injection is detected."""
        for pattern in COMPILED_INJECTION_PATTERNS:
            if pattern.search(text):
                logger.warning(f"Prompt injection detected: '{text[:100]}'")
                return True
        return False

    def check_off_topic(self, text: str) -> bool:
        """Return True if the query is clearly off-topic."""
        for pattern in COMPILED_OFF_TOPIC_PATTERNS:
            if pattern.search(text):
                logger.info(f"Off-topic query detected: '{text[:100]}'")
                return True
        return False

    def is_comparison_query(self, text: str) -> bool:
        """Return True if user wants to compare assessments."""
        return any(p.search(text) for p in COMPILED_COMPARISON_PATTERNS)

    def is_refinement_query(self, text: str, has_previous_recommendations: bool = False) -> bool:
        """Return True if user wants to refine recommendations."""
        if not has_previous_recommendations:
            return False
        return any(p.search(text) for p in COMPILED_REFINEMENT_PATTERNS)

    def is_closing_query(self, text: str) -> bool:
        """Return True if user is closing the conversation."""
        return any(p.search(text) for p in COMPILED_CLOSING_PATTERNS)

    def is_vague_query(self, text: str, conversation_length: int = 0) -> bool:
        """Return True if the query is too vague to recommend."""
        # Single-turn vague patterns
        if conversation_length <= 1:
            for pattern in COMPILED_VAGUE_PATTERNS:
                if pattern.search(text.strip()):
                    return True

        # Word count heuristic
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text)
        stop_words = {"the", "and", "for", "are", "with", "that", "this", "from", "have", "need", "want", "some"}
        meaningful_words = [w for w in words if w.lower() not in stop_words]

        return len(meaningful_words) < MIN_CONTEXT_WORDS

    def sanitize_input(self, text: str) -> str:
        """
        Sanitize user input:
        - Strip excessive whitespace
        - Remove null bytes and control characters
        - Truncate to max length
        """
        # Remove null bytes and control chars (except newlines and tabs)
        text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", text)
        # Normalize whitespace
        text = re.sub(r" {2,}", " ", text)
        text = text.strip()
        # Truncate
        if len(text) > 8192:
            text = text[:8192] + "..."
        return text

    def classify_query(
        self,
        user_message: str,
        conversation_messages: list[dict],
    ) -> QueryType:
        """
        Classify the user's query type for routing logic.
        
        Args:
            user_message: Current user message
            conversation_messages: Full conversation history
            
        Returns:
            QueryType enum value
        """
        text = user_message.lower().strip()
        conv_length = len(conversation_messages)

        # Has the agent already made recommendations?
        has_recommendations = self._has_previous_recommendations(conversation_messages)

        # Priority order matters

        # 1. Injection check (highest priority)
        if self.check_prompt_injection(user_message):
            return QueryType.PROMPT_INJECTION

        # 2. Off-topic
        if self.check_off_topic(user_message):
            return QueryType.OFF_TOPIC

        # 3. Closing
        if self.is_closing_query(text):
            return QueryType.CLOSING

        # 4. Comparison
        if self.is_comparison_query(text):
            return QueryType.COMPARISON

        # 5. Refinement (only if previous recommendations exist)
        if has_recommendations and self.is_refinement_query(text, has_recommendations):
            return QueryType.REFINEMENT

        # 6. Vague (needs clarification)
        if self.is_vague_query(user_message, conv_length):
            return QueryType.VAGUE

        # 7. Default: specific enough to retrieve
        return QueryType.SPECIFIC

    def _has_previous_recommendations(self, messages: list[dict]) -> bool:
        """Check if previous assistant messages contained recommendations."""
        import json
        for msg in messages:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                try:
                    data = json.loads(content)
                    if data.get("recommendations"):
                        return True
                except (json.JSONDecodeError, AttributeError):
                    # May be natural text response
                    if "recommend" in content.lower() or "assessment" in content.lower():
                        return True
        return False

    def validate_messages(self, messages: list[dict]) -> tuple[bool, str]:
        """
        Validate the messages list for security and format.
        Returns (is_valid, error_message).
        """
        if not messages:
            return False, "Messages list cannot be empty"

        for i, msg in enumerate(messages):
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role not in ("user", "assistant", "system"):
                return False, f"Invalid role at index {i}: {role}"

            if not content or not content.strip():
                return False, f"Empty content at index {i}"

            if len(content) > 8192:
                return False, f"Message too long at index {i}: {len(content)} chars (max 8192)"

        # Last message must be from user
        if messages[-1].get("role") != "user":
            return False, "Last message must be from the user"

        return True, ""


# Singleton instance
_guard = SecurityGuard()


def get_security_guard() -> SecurityGuard:
    """Get the singleton SecurityGuard instance."""
    return _guard
