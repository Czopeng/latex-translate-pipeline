"""Utilities for masking and restoring LaTeX commands safely."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, Tuple


PLACEHOLDER_TEMPLATE = "__CMD{index}__"
PLACEHOLDER_PATTERN = re.compile(r"__CMD\d+__")


@dataclass
class MaskResult:
    """Container for masked text and its placeholder mapping."""

    masked_text: str
    placeholder_map: Dict[str, str]


class LatexMasker:
    """Mask LaTeX commands before translation and restore them afterwards."""

    # Regex-only strategy by design, as requested for this phase.
    # Order matters: broader protected blocks first, then smaller inline fragments.
    _protected_environment_pattern = re.compile(
        r"\\begin\{(?P<env>verbatim\*?|lstlisting|minted)\}.*?\\end\{(?P=env)\}",
        re.DOTALL,
    )
    _math_environment_pattern = re.compile(
        r"\\begin\{(?P<env>equation\*?|align\*?|alignat\*?|gather\*?|multline\*?|flalign\*?|displaymath|math|cases|split)\}.*?\\end\{(?P=env)\}",
        re.DOTALL,
    )
    _display_math_pattern = re.compile(
        r"(\\\[(?:.|\n)*?\\\])|((?<!\\)\$\$(?:.|\n)*?(?<!\\)\$\$)",
        re.DOTALL,
    )
    _inline_math_pattern = re.compile(r"(?<!\\)\$(?!\$)(?:\\.|[^$\\\n])*(?<!\\)\$")
    _command_pattern = re.compile(
        r"""
        (\\(?:begin|end)\s*\{[^{}]+\})
        |(\\[a-zA-Z@]+\*?(?:\s*\[[^\[\]]*\])*(?:\s*\{[^{}]*\})*)
        |(\\[%$&#_{}])
        |(\\\\)
        """,
        re.VERBOSE | re.DOTALL,
    )

    def _next_placeholder(self, source_text: str, placeholder_map: Dict[str, str], start_index: int) -> tuple[str, int]:
        """Return the next collision-free placeholder token and next index."""

        index = start_index
        while True:
            candidate = PLACEHOLDER_TEMPLATE.format(index=index)
            if candidate not in source_text and candidate not in placeholder_map:
                return candidate, index + 1
            index += 1

    def _mask_with_pattern(
        self,
        source_text: str,
        pattern: re.Pattern[str],
        placeholder_map: Dict[str, str],
        start_index: int,
    ) -> tuple[str, int]:
        """Mask pattern matches and return updated text plus next placeholder index."""

        counter = start_index

        def _replace(match: re.Match[str]) -> str:
            nonlocal counter
            original = match.group(0)
            placeholder, counter = self._next_placeholder(source_text, placeholder_map, counter)
            placeholder_map[placeholder] = original
            return placeholder

        return pattern.sub(_replace, source_text), counter

    def mask_latex_commands(self, text: str) -> MaskResult:
        """Replace LaTeX command fragments with deterministic placeholders.

        Args:
            text: Raw LaTeX text.

        Returns:
            MaskResult with masked text and placeholder->original mapping.
        """

        placeholder_map: Dict[str, str] = {}
        counter = 1

        masked_text = text
        for pattern in (
            self._protected_environment_pattern,
            self._math_environment_pattern,
            self._display_math_pattern,
            self._inline_math_pattern,
            self._command_pattern,
        ):
            masked_text, counter = self._mask_with_pattern(masked_text, pattern, placeholder_map, counter)

        return MaskResult(masked_text=masked_text, placeholder_map=placeholder_map)

    def restore_latex_commands(self, text: str, placeholder_map: Dict[str, str]) -> str:
        """Restore exact original LaTeX command fragments by placeholder.

        Args:
            text: Text that may contain placeholder tokens.
            placeholder_map: Mapping created by ``mask_latex_commands``.

        Returns:
            Text with placeholders restored to original LaTeX fragments.
        """

        if not isinstance(placeholder_map, dict):
            raise TypeError("placeholder_map must be a dictionary")

        def _restore(match: re.Match[str]) -> str:
            token = match.group(0)
            return placeholder_map.get(token, token)

        restored = text
        for _ in range(len(placeholder_map) + 1):
            updated = PLACEHOLDER_PATTERN.sub(_restore, restored)
            if updated == restored:
                break
            restored = updated
        return restored

    def mask(self, text: str) -> Tuple[str, Dict[str, str]]:
        """Convenience wrapper returning tuple output for pipeline use."""

        result = self.mask_latex_commands(text)
        return result.masked_text, result.placeholder_map
