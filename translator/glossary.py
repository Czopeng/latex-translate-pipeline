"""Glossary parsing, translation helpers, and JSON conversion."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Callable, Dict

from .utils import extract_braced_content


@dataclass
class GlossaryEntry:
    de: str
    en: str
    type: str
    abbreviation: str = ""


class GlossaryManager:
    """Parse glossary commands and manage glossary transformations."""

    _acronym_pattern = re.compile(
        r"\\newacronym\{(?P<label>[^}]+)\}\{(?P<abbr>[^}]*)\}\{(?P<desc>[^}]*)\}"
    )
    _entry_start_pattern = re.compile(r"\\newglossaryentry\{(?P<label>[^}]+)\}\s*\{")

    def extract_glossary(self, latex_text: str) -> Dict[str, GlossaryEntry]:
        glossary: Dict[str, GlossaryEntry] = {}

        glossary.update(self._parse_acronyms(latex_text))
        glossary.update(self._parse_glossary_entries(latex_text))
        return glossary

    def _parse_acronyms(self, latex_text: str) -> Dict[str, GlossaryEntry]:
        entries: Dict[str, GlossaryEntry] = {}
        for match in self._acronym_pattern.finditer(latex_text):
            label = match.group("label").strip()
            entries[label] = GlossaryEntry(
                de=match.group("desc").strip(),
                en="",
                type="acronym",
                abbreviation=match.group("abbr").strip(),
            )
        return entries

    def _parse_glossary_entries(self, latex_text: str) -> Dict[str, GlossaryEntry]:
        entries: Dict[str, GlossaryEntry] = {}

        for match in self._entry_start_pattern.finditer(latex_text):
            label = match.group("label").strip()
            opening_brace_index = match.end() - 1

            try:
                body, _ = extract_braced_content(latex_text, opening_brace_index)
            except ValueError:
                continue

            name_match = re.search(r"name\s*=\s*\{([^{}]*)\}", body, flags=re.DOTALL)
            desc_match = re.search(r"description\s*=\s*\{([^{}]*)\}", body, flags=re.DOTALL)

            de_value = ""
            if desc_match:
                de_value = desc_match.group(1).strip()
            elif name_match:
                de_value = name_match.group(1).strip()

            entries[label] = GlossaryEntry(
                de=de_value,
                en="",
                type="glossary",
                abbreviation="",
            )

        return entries

    def to_json_dict(self, glossary: Dict[str, GlossaryEntry]) -> Dict[str, dict]:
        return {key: asdict(value) for key, value in glossary.items()}

    def from_json_dict(self, data: Dict[str, dict]) -> Dict[str, GlossaryEntry]:
        result: Dict[str, GlossaryEntry] = {}
        for key, raw in data.items():
            result[key] = GlossaryEntry(
                de=raw.get("de", ""),
                en=raw.get("en", ""),
                type=raw.get("type", "glossary"),
                abbreviation=raw.get("abbreviation", ""),
            )
        return result

    def translate_glossary(
        self,
        glossary: Dict[str, GlossaryEntry],
        translator: Callable[[str], str],
    ) -> Dict[str, GlossaryEntry]:
        """Translate DE descriptions to EN while preserving abbreviations."""

        translated: Dict[str, GlossaryEntry] = {}

        for key, entry in glossary.items():
            translated_entry = GlossaryEntry(
                de=entry.de,
                en=entry.en,
                type=entry.type,
                abbreviation=entry.abbreviation,
            )

            if entry.de and not entry.en:
                translated_entry.en = translator(entry.de)

            translated[key] = translated_entry

        return translated
