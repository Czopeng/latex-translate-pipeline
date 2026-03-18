"""High-level translation pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from .glossary import GlossaryEntry, GlossaryManager
from .latex_masker import LatexMasker
from .translator import LLMClient
from .utils import detect_candidate_terms, load_json, read_text, save_json


@dataclass
class PipelineResult:
    translated_text: str
    glossary: Dict[str, GlossaryEntry]
    suggested_terms: list[str]


class TranslationPipeline:
    """Modular German->English LaTeX translation pipeline skeleton."""

    def __init__(self, llm_client: LLMClient, glossary_path: str | Path = "glossary.json") -> None:
        self.llm_client = llm_client
        self.glossary_path = Path(glossary_path)
        self.masker = LatexMasker()
        self.glossary_manager = GlossaryManager()

    def load_latex(self, path: str | Path) -> str:
        return read_text(path)

    def extract_glossary(self, latex_text: str) -> Dict[str, GlossaryEntry]:
        return self.glossary_manager.extract_glossary(latex_text)

    def mask_latex_commands(self, text: str) -> tuple[str, Dict[str, str]]:
        return self.masker.mask(text)

    def translate_text(self, masked_text: str, glossary: Dict[str, GlossaryEntry]) -> str:
        glossary_context = self.glossary_manager.to_json_dict(glossary)
        return self.llm_client.translate_text(masked_text, glossary_context)

    def restore_latex_commands(self, text: str, placeholder_map: Dict[str, str]) -> str:
        return self.masker.restore_latex_commands(text, placeholder_map)

    def update_glossary(
        self,
        base_glossary: Dict[str, GlossaryEntry],
        extracted_glossary: Dict[str, GlossaryEntry],
        suggested_terms: list[str],
    ) -> Dict[str, GlossaryEntry]:
        merged = dict(base_glossary)

        for key, entry in extracted_glossary.items():
            merged[key] = entry

        for term in suggested_terms:
            if term not in merged:
                merged[term] = GlossaryEntry(de=term, en="", type="suggested", abbreviation="")

        save_json(self.glossary_path, self.glossary_manager.to_json_dict(merged))
        return merged

    def run(self, latex_path: str | Path) -> PipelineResult:
        latex_text = self.load_latex(latex_path)
        extracted_glossary = self.extract_glossary(latex_text)

        existing_raw = load_json(self.glossary_path, default={})
        base_glossary = self.glossary_manager.from_json_dict(existing_raw)

        working_glossary = dict(base_glossary)
        working_glossary.update(extracted_glossary)

        masked_text, placeholder_map = self.mask_latex_commands(latex_text)
        translated_masked = self.translate_text(masked_text, working_glossary)
        restored_text = self.restore_latex_commands(translated_masked, placeholder_map)

        suggested_terms = detect_candidate_terms(restored_text)
        updated_glossary = self.update_glossary(base_glossary, extracted_glossary, suggested_terms)

        return PipelineResult(
            translated_text=restored_text,
            glossary=updated_glossary,
            suggested_terms=suggested_terms,
        )
