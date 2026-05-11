"""
LLM Client
============
Unified LLM client supporting multiple providers:
- Groq (primary, fast)
- OpenAI
- Google Gemini
- OpenRouter

Features:
- Automatic retry with exponential backoff
- JSON response parsing and validation
- Timeout handling
- Provider fallback
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Optional

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

logger = logging.getLogger(__name__)

# Expected response schema
REQUIRED_KEYS = {"reply", "recommendations", "end_of_conversation"}


class LLMClient:
    """
    Unified LLM client with provider abstraction.
    Handles JSON schema enforcement and retries.
    """

    def __init__(
        self,
        provider: str = "groq",
        api_key: str = "",
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.3,
        max_tokens: int = 2048,
        timeout: int = 30,
    ):
        self.provider = provider.lower()
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._client = None

    def _get_client(self):
        """Lazy initialize the provider client."""
        if self._client is not None:
            return self._client

        if self.provider == "groq":
            from groq import Groq
            self._client = Groq(api_key=self.api_key)
        elif self.provider == "openai":
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        elif self.provider == "openrouter":
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key,
                base_url="https://openrouter.ai/api/v1",
            )
        elif self.provider == "google":
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._client = genai.GenerativeModel(self.model)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        return self._client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def chat(self, system_prompt: str, user_message: str = "") -> dict:
        """
        Send a chat completion request and return parsed JSON response.
        
        Returns the agent response dict with keys:
        - reply: str
        - recommendations: list
        - end_of_conversation: bool
        """
        client = self._get_client()
        start_time = time.time()

        try:
            if self.provider in ("groq", "openai", "openrouter"):
                messages = [
                    {"role": "system", "content": system_prompt},
                ]
                if user_message:
                    messages.append({"role": "user", "content": user_message})

                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout,
                )
                raw_text = response.choices[0].message.content

            elif self.provider == "google":
                combined = system_prompt
                if user_message:
                    combined += f"\n\nUser: {user_message}"
                response = client.generate_content(
                    combined,
                    request_options={"timeout": self.timeout},
                )
                raw_text = response.text

            else:
                raise ValueError(f"Unsupported provider: {self.provider}")

            elapsed = time.time() - start_time
            logger.info(f"LLM response in {elapsed:.2f}s ({self.provider}/{self.model})")

            return self._parse_response(raw_text)

        except Exception as e:
            logger.error(f"LLM call failed ({self.provider}): {e}")
            raise

    def _parse_response(self, raw_text: str) -> dict:
        """
        Parse and validate LLM response as JSON.
        Handles common formatting issues (markdown fences, etc.)
        """
        if not raw_text:
            return self._error_response("Empty response from LLM")

        text = raw_text.strip()

        # Remove markdown code fences if present
        text = re.sub(r"^```(?:json)?\n?", "", text, flags=re.MULTILINE)
        text = re.sub(r"\n?```$", "", text, flags=re.MULTILINE)
        text = text.strip()

        # Try to extract JSON object if mixed with text
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            text = json_match.group(0)

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}. Raw text: {text[:200]}")
            return self._error_response(f"Invalid JSON response: {e}")

        # Validate required keys
        missing_keys = REQUIRED_KEYS - set(data.keys())
        if missing_keys:
            logger.warning(f"Missing keys in response: {missing_keys}")
            # Add defaults for missing keys
            if "reply" not in data:
                data["reply"] = "I apologize, I encountered an error. Please try again."
            if "recommendations" not in data:
                data["recommendations"] = []
            if "end_of_conversation" not in data:
                data["end_of_conversation"] = False

        # Validate recommendations
        recs = data.get("recommendations", [])
        if not isinstance(recs, list):
            data["recommendations"] = []
        else:
            valid_recs = []
            for rec in recs[:10]:  # Max 10
                if isinstance(rec, dict) and rec.get("name") and rec.get("url"):
                    valid_recs.append({
                        "name": str(rec.get("name", "")),
                        "url": str(rec.get("url", "")),
                        "test_type": str(rec.get("test_type", "Assessment")),
                    })
            data["recommendations"] = valid_recs

        # Validate end_of_conversation
        data["end_of_conversation"] = bool(data.get("end_of_conversation", False))

        return data

    def _error_response(self, error_msg: str) -> dict:
        """Return a safe error response conforming to schema."""
        logger.error(f"LLM Error: {error_msg}")
        return {
            "reply": (
                "I apologize for the technical issue. Could you please rephrase your request "
                "and I'll do my best to help you find the right SHL assessment?"
            ),
            "recommendations": [],
            "end_of_conversation": False,
        }


def create_llm_client(settings) -> LLMClient:
    """Factory: create LLM client from app settings."""
    return LLMClient(
        provider=settings.llm_provider,
        api_key=settings.active_api_key,
        model=settings.active_model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        timeout=30,
    )
