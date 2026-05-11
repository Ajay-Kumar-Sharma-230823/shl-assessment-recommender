"""
Recommendation Agent
======================
Core orchestration engine that:
1. Classifies the query
2. Extracts conversation context
3. Performs retrieval
4. Builds the prompt
5. Calls the LLM
6. Validates and returns the response

This is the central brain of the system.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    RecommendationItem,
    SHLAssessment,
)
from app.models.config import get_settings
from app.prompts.templates import (
    INSTRUCTIONS_CLARIFY,
    INSTRUCTIONS_CLOSE,
    INSTRUCTIONS_COMPARE,
    INSTRUCTIONS_RECOMMEND,
    INSTRUCTIONS_REFUSE,
    INSTRUCTIONS_REFINE,
    build_system_prompt,
    format_catalog_context,
    format_conversation_history,
)
from app.retrieval.retrieval_engine import RetrievalEngine, parse_filters_from_context
from app.retrieval.vector_store import BaseVectorStore, get_vector_store
from app.security.guards import QueryType, get_security_guard
from app.services.conversation import ConversationContext, get_conversation_manager
from app.services.llm_client import LLMClient, create_llm_client

logger = logging.getLogger(__name__)

# Off-topic refusal message
REFUSAL_RESPONSE = {
    "reply": (
        "I'm the SHL Assessment Advisor and I'm specialized only in helping you find "
        "the right SHL assessments. I can't assist with that topic. "
        "Would you like help discovering SHL assessments for a specific role?"
    ),
    "recommendations": [],
    "end_of_conversation": False,
}

# Injection refusal message
INJECTION_RESPONSE = {
    "reply": (
        "I'm the SHL Assessment Advisor and can only help with SHL assessment "
        "recommendations. How can I assist you with finding the right assessment "
        "for your hiring needs?"
    ),
    "recommendations": [],
    "end_of_conversation": False,
}


class RecommendationAgent:
    """
    Main orchestration agent for SHL assessment recommendations.
    Stateless: all context is derived from the request messages.
    """

    def __init__(
        self,
        retrieval_engine: RetrievalEngine,
        llm_client: LLMClient,
    ):
        self.retrieval_engine = retrieval_engine
        self.llm_client = llm_client
        self.security_guard = get_security_guard()
        self.conversation_manager = get_conversation_manager()
        self.settings = get_settings()

    def process(self, request: ChatRequest) -> ChatResponse:
        """
        Process a chat request and return a ChatResponse.
        
        Full pipeline:
        1. Validate input
        2. Security checks
        3. Classify query
        4. Extract context
        5. Retrieve assessments
        6. Build prompt
        7. Call LLM
        8. Validate & return response
        """
        messages = [m.model_dump() for m in request.messages]
        user_message = messages[-1]["content"]

        # Count user turns only
        user_turn_count = sum(1 for m in messages if m.get("role") == "user")
        logger.info(f"Processing request — turn {user_turn_count}, query: '{user_message[:80]}'")

        # ---- Hard 8-turn cap (per assignment: evaluator caps at 8 turns) ----
        max_turns = self.settings.max_turns  # default: 8
        if user_turn_count > max_turns:
            logger.warning(f"Turn cap exceeded: {user_turn_count} > {max_turns}")
            return ChatResponse(
                reply=(
                    "Thank you for using the SHL Assessment Advisor! "
                    "We've reached the maximum conversation length. "
                    "Please start a new conversation if you need further assistance."
                ),
                recommendations=[],
                end_of_conversation=True,
            )

        # ---- Step 1: Input validation ----
        is_valid, error = self.security_guard.validate_messages(messages)
        if not is_valid:
            logger.warning(f"Invalid input: {error}")
            return ChatResponse(
                reply=f"I encountered an input error: {error}. Please try again.",
                recommendations=[],
                end_of_conversation=False,
            )

        # ---- Step 2: Sanitize input ----
        user_message_clean = self.security_guard.sanitize_input(user_message)

        # ---- Step 3: Security classification ----
        query_type = self.security_guard.classify_query(user_message_clean, messages)
        logger.info(f"Query type: {query_type}")

        # ---- Handle injection immediately ----
        if query_type == QueryType.PROMPT_INJECTION:
            return ChatResponse(**INJECTION_RESPONSE)

        # ---- Handle off-topic immediately ----
        if query_type == QueryType.OFF_TOPIC:
            return ChatResponse(**REFUSAL_RESPONSE)

        # ---- Step 4: Extract conversation context ----
        ctx = self.conversation_manager.extract_context(messages)

        # ---- Step 5: Determine instruction type ----
        instructions = self._select_instructions(query_type, ctx)

        # ---- Step 6: Build search query ----
        search_query = self._build_search_query(
            query_type, user_message_clean, ctx
        )

        # ---- Step 7: Retrieve assessments ----
        retrieved = []
        if query_type != QueryType.CLOSING:
            filters = parse_filters_from_context(
                " ".join(m.get("content", "") for m in messages if m.get("role") == "user")
            )
            top_k = self.settings.top_k_results

            # For comparison, retrieve specifically named assessments
            if query_type == QueryType.COMPARISON:
                named = self._extract_assessment_names(user_message_clean)
                if named:
                    retrieved = self.retrieval_engine.retrieve_for_comparison(named)

            if not retrieved:
                retrieved = self.retrieval_engine.retrieve(
                    query=search_query,
                    filters=filters,
                    top_k=top_k,
                )

        logger.info(f"Retrieved {len(retrieved)} assessments")

        # ---- Step 8: Build prompt ----
        catalog_ctx = format_catalog_context(retrieved)
        history_str = format_conversation_history(messages)

        # Compress history if needed
        if self.conversation_manager.should_compress_history(messages):
            history_str = self.conversation_manager.get_conversation_summary(messages)

        system_prompt = build_system_prompt(
            catalog_context=catalog_ctx,
            conversation_history=history_str,
            user_message=user_message_clean,
            instructions=instructions,
        )

        # ---- Step 9: Call LLM ----
        try:
            llm_response = self.llm_client.chat(system_prompt=system_prompt)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return ChatResponse(
                reply=(
                    "I'm experiencing a technical issue. Please try again in a moment. "
                    "I'm here to help you find the right SHL assessments."
                ),
                recommendations=[],
                end_of_conversation=False,
            )

        # ---- Step 10: Validate and build response ----
        return self._build_response(llm_response, retrieved, query_type, ctx)

    def _select_instructions(self, query_type: QueryType, ctx: ConversationContext) -> str:
        """Select the appropriate instruction block based on query type and context."""
        instruction_map = {
            QueryType.VAGUE: INSTRUCTIONS_CLARIFY,
            QueryType.SPECIFIC: INSTRUCTIONS_RECOMMEND if ctx.has_enough_context() else INSTRUCTIONS_CLARIFY,
            QueryType.COMPARISON: INSTRUCTIONS_COMPARE,
            QueryType.REFINEMENT: INSTRUCTIONS_REFINE,
            QueryType.CLOSING: INSTRUCTIONS_CLOSE,
            QueryType.OFF_TOPIC: INSTRUCTIONS_REFUSE,
            QueryType.PROMPT_INJECTION: INSTRUCTIONS_REFUSE,
        }
        return instruction_map.get(query_type, INSTRUCTIONS_CLARIFY)

    def _build_search_query(
        self,
        query_type: QueryType,
        user_message: str,
        ctx: ConversationContext,
    ) -> str:
        """Build an effective search query from context and current message."""
        if query_type == QueryType.COMPARISON:
            return user_message  # Search for mentioned assessments

        if query_type == QueryType.REFINEMENT:
            # Combine context with refinement request
            base = ctx.build_search_query()
            return f"{base} {user_message}"

        if ctx.has_enough_context():
            return ctx.build_search_query()

        # For vague queries, use the message directly
        return user_message if user_message else ctx.build_search_query()

    def _extract_assessment_names(self, text: str) -> list[str]:
        """
        Extract mentioned assessment names from comparison query.
        Common SHL assessment name patterns.
        """
        known_names = [
            "OPQ32", "OPQ", "Verify G+", "Verify", "MQ", "MQ+",
            "GSA", "GIA", "Numerical Reasoning", "Verbal Reasoning",
            "Inductive Reasoning", "Deductive Reasoning", "Calculation",
            "Checking", "Abstract Reasoning", "CCSQ", "RemoteWorkQ",
            "ADEPT-15", "Hogan", "Motives Values Preferences Inventory",
            "Work Personality Index", "Occupational Personality Questionnaire",
        ]

        found = []
        text_lower = text.lower()
        for name in known_names:
            if name.lower() in text_lower:
                found.append(name)

        return found

    def _build_response(
        self,
        llm_data: dict,
        retrieved: list[dict],
        query_type: QueryType,
        ctx: ConversationContext,
    ) -> ChatResponse:
        """
        Build the final ChatResponse from LLM output.
        Validates recommendations against catalog data.
        """
        reply = llm_data.get("reply", "")
        raw_recs = llm_data.get("recommendations", [])
        eoc = llm_data.get("end_of_conversation", False)

        # ---- Validate recommendations against catalog ----
        # Only accept recommendations that match retrieved catalog items
        valid_recs = []
        if raw_recs:
            catalog_items = {
                a["assessment"].get("name", "").lower(): a["assessment"]
                for a in retrieved
            }
            catalog_urls = {
                a["assessment"].get("url", ""): a["assessment"]
                for a in retrieved
            }

            for rec in raw_recs:
                rec_name_lower = rec.get("name", "").lower()
                rec_url = rec.get("url", "")

                # Check if it matches a retrieved item by name or URL
                matched_item = None
                for catalog_name, item in catalog_items.items():
                    if (
                        rec_name_lower in catalog_name
                        or catalog_name in rec_name_lower
                        or rec_url == item.get("url", "")
                    ):
                        matched_item = item
                        break

                if matched_item:
                    # Use catalog URL (prevents hallucinated URLs)
                    valid_recs.append(
                        RecommendationItem(
                            name=matched_item.get("name", rec.get("name", "")),
                            url=matched_item.get("url", rec.get("url", "")),
                            test_type=matched_item.get("test_type", rec.get("test_type", "Assessment")),
                        )
                    )
                elif rec_url and "shl.com" in rec_url:
                    # URL matches SHL domain — include cautiously
                    valid_recs.append(
                        RecommendationItem(
                            name=rec.get("name", ""),
                            url=rec_url,
                            test_type=rec.get("test_type", "Assessment"),
                        )
                    )
                else:
                    logger.warning(
                        f"Rejected hallucinated recommendation: {rec.get('name')} / {rec.get('url')}"
                    )

        # Cap at 10
        valid_recs = valid_recs[:10]

        # ---- Set end_of_conversation ----
        if query_type == QueryType.CLOSING:
            eoc = True
        elif eoc and not valid_recs and query_type not in (QueryType.CLOSING, QueryType.OFF_TOPIC):
            # Don't close if we haven't recommended anything
            eoc = False

        return ChatResponse(
            reply=reply,
            recommendations=valid_recs,
            end_of_conversation=eoc,
        )
