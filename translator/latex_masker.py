"""Utilities for masking and restoring LaTeX commands safely."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, Tuple


PLACEHOLDER_TEMPLATE = "__CMD{index:05d}__"


@dataclass
class MaskResult:
    """Container for masked text and its placeholder mapping."""

    masked_text: str
    placeholder_map: Dict[str, str]


class LatexMasker:
    """Mask LaTeX commands before translation and restore them afterwards."""

    # Regex-only strategy by design, as requested for this phase.
    _command_pattern = re.compile(
        r"""
        (\\(?:begin|end)\s*\{[^{}]+\})
        |(\\[a-zA-Z@]+\*?(?:\s*\[[^\[\]]*\])*(?:\s*\{[^{}]*\})*)
        |(\\[%$&#_{}])
        |(\\\\)
        """,
        re.VERBOSE | re.DOTALL,
    )

    def mask_latex_commands(self, text: str) -> MaskResult:
        """Replace LaTeX command fragments with deterministic placeholders."""

        placeholder_map: Dict[str, str] = {}
        counter = 1

        def _replace(match: re.Match[str]) -> str:
            nonlocal counter
            original = match.group(0)
            placeholder = PLACEHOLDER_TEMPLATE.format(index=counter)
            placeholder_map[placeholder] = original
            counter += 1
            return placeholder

        masked_text = self._command_pattern.sub(_replace, text)
        return MaskResult(masked_text=masked_text, placeholder_map=placeholder_map)

    def restore_latex_commands(self, text: str, placeholder_map: Dict[str, str]) -> str:
        """Restore exact original LaTeX command fragments by placeholder."""

        restored = text
        for placeholder in sorted(placeholder_map, key=len, reverse=True):
            restored = restored.replace(placeholder, placeholder_map[placeholder])
        return restored

    def mask(self, text: str) -> Tuple[str, Dict[str, str]]:
        """Convenience wrapper returning tuple output for pipeline use."""

        result = self.mask_latex_commands(text)
        return result.masked_text, result.placeholder_map
