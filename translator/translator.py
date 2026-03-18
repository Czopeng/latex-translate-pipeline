"""LLM provider abstraction and mock implementation."""

from __future__ import annotations

from typing import Dict, Protocol


class LLMClient(Protocol):
    """Abstraction for provider-swappable translation backends."""

    def translate_text(self, text: str, glossary_context: Dict[str, dict]) -> str:
        """Translate text using provided glossary context."""


class MockLLMClient:
    """Mock implementation for architecture-first development."""

    def translate_text(self, text: str, glossary_context: Dict[str, dict]) -> str:
        # For the first architecture step we do not perform real translation.
        _ = glossary_context
        return text
