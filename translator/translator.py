"""LLM provider abstraction and mock implementation."""

from __future__ import annotations

import os
from pathlib import Path
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


def _read_env_key_from_file(env_path: str | Path = ".env") -> str | None:
    """Read Gemini API key from a local .env file without external dependencies."""

    path = Path(env_path)
    if not path.exists():
        return None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        normalized_key = key.strip()
        normalized_value = value.strip().strip('"').strip("'")
        if normalized_key in {"GEMINI_API_KEY", "geminiAPIkey"} and normalized_value:
            return normalized_value
    return None


class GeminiLLMClient:
    """Google Gemini implementation of the translation backend."""

    def __init__(self, model: str = "gemini-2.0-flash", api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("geminiAPIkey")
        if not self.api_key:
            self.api_key = _read_env_key_from_file()

        if not self.api_key:
            raise ValueError("Gemini API key not found. Set GEMINI_API_KEY or geminiAPIkey in .env.")

        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "google-genai is not installed. Install it via requirements.txt or pip install google-genai."
            ) from exc

        self._client = genai.Client(api_key=self.api_key)

    def _build_prompt(self, text: str, glossary_context: Dict[str, dict]) -> str:
        style_context = glossary_context.get("__style__", {}) if isinstance(glossary_context, dict) else {}
        style_lines: list[str] = []
        if isinstance(style_context, dict):
            tone = style_context.get("tone", "")
            domain = style_context.get("domain", "")
            keep_units = style_context.get("keep_units", True)
            if tone:
                style_lines.append(f"- Tone: {tone}")
            if domain:
                style_lines.append(f"- Domain: {domain}")
            style_lines.append(f"- Keep units/symbol formatting unchanged: {bool(keep_units)}")

        glossary_lines: list[str] = []
        for term, data in glossary_context.items():
            if term == "__style__":
                continue
            de = data.get("de", "")
            en = data.get("en", "")
            entry_type = data.get("type", "")
            glossary_lines.append(f"- {term} | type={entry_type} | de={de} | en={en}")

        glossary_block = "\n".join(glossary_lines) if glossary_lines else "- (none)"
        style_block = "\n".join(style_lines) if style_lines else "- Keep formal scientific writing style"

        return (
            "Translate German scientific LaTeX text to formal English.\n"
            "Rules:\n"
            "1) Never modify placeholders like __CMD123__.\n"
            "2) Never add/remove placeholders.\n"
            "3) Preserve structure and line breaks.\n"
            "4) Use glossary terms when relevant.\n\n"
            f"Style:\n{style_block}\n\n"
            f"Glossary:\n{glossary_block}\n\n"
            f"Input:\n{text}"
        )

    def translate_text(self, text: str, glossary_context: Dict[str, dict]) -> str:
        prompt = self._build_prompt(text, glossary_context)

        response = self._client.models.generate_content(model=self.model, contents=prompt)
        output = getattr(response, "text", None)
        if not output:
            raise RuntimeError("Gemini returned an empty response.")
        return output
