"""High-level translation pipeline orchestration."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import fnmatch
from pathlib import Path
import re
from typing import Dict

from .glossary import GlossaryEntry, GlossaryManager
from .latex_masker import LatexMasker, PLACEHOLDER_PATTERN
from .translator import LLMClient
from .utils import detect_candidate_terms, load_json, read_text, save_json, write_text


@dataclass
class PipelineResult:
    translated_text: str
    glossary: Dict[str, GlossaryEntry]
    suggested_terms: list[str]
    chunk_count: int


class PlaceholderIntegrityError(RuntimeError):
    """Raised when placeholder tokens are altered by the translation backend."""


class GlossaryReviewQueueError(RuntimeError):
    """Raised when a glossary review queue cannot be applied safely."""


class TranslationPipeline:
    """Modular German->English LaTeX translation pipeline skeleton."""

    _heading_boundary_pattern = re.compile(r"(?m)^(?=\\(?:chapter|section|subsection)\*?\{)")

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

    def discover_latex_files(
        self,
        input_folder: str | Path,
        include_glob: str = "*.tex",
        exclude_globs: list[str] | None = None,
        glossary_source_path: str | Path | None = None,
    ) -> list[Path]:
        """Discover LaTeX files deterministically for a folder job."""

        root = Path(input_folder)
        if not root.exists() or not root.is_dir():
            raise ValueError(f"Input folder does not exist or is not a directory: {root}")

        excluded = exclude_globs or []
        glossary_source_resolved = Path(glossary_source_path).resolve() if glossary_source_path is not None else None

        files: list[Path] = []
        for candidate in root.rglob(include_glob):
            if not candidate.is_file():
                continue

            rel = candidate.relative_to(root).as_posix()
            if any(fnmatch.fnmatch(rel, pattern) for pattern in excluded):
                continue

            if glossary_source_resolved is not None and candidate.resolve() == glossary_source_resolved:
                continue

            files.append(candidate)

        return sorted(files, key=lambda p: p.as_posix())

    def sync_glossary_from_source(self, glossary_source_path: str | Path) -> dict:
        """Sync glossary.json with read-only LaTeX glossary source while preserving approved EN values."""

        source_text = self.load_latex(glossary_source_path)
        source_entries = self.extract_glossary(source_text)

        existing_raw = load_json(self.glossary_path, default={})
        existing_entries = self.glossary_manager.from_json_dict(existing_raw)

        merged: Dict[str, GlossaryEntry] = {}
        preserved_en_count = 0

        for term, source_entry in source_entries.items():
            existing = existing_entries.get(term)
            preserved_en = existing.en if existing and existing.en else ""
            if preserved_en:
                preserved_en_count += 1

            merged[term] = GlossaryEntry(
                de=source_entry.de,
                en=preserved_en,
                type=source_entry.type,
                abbreviation=source_entry.abbreviation,
            )

        # Keep previously approved non-source terms (e.g. suggested terms) instead of dropping data.
        for term, existing_entry in existing_entries.items():
            if term not in merged:
                merged[term] = existing_entry

        save_json(self.glossary_path, self.glossary_manager.to_json_dict(merged))
        return {
            "source_file": str(glossary_source_path),
            "source_terms": len(source_entries),
            "preserved_en": preserved_en_count,
            "total_terms_after_sync": len(merged),
        }

    def split_into_chunks_by_headings(self, latex_text: str) -> list[str]:
        """Split LaTeX text at chapter/section/subsection boundaries.

        Delimiters are preserved by splitting on zero-width lookahead boundaries.
        """

        boundary_indices = [match.start() for match in self._heading_boundary_pattern.finditer(latex_text)]
        if not boundary_indices:
            return [latex_text]

        chunks: list[str] = []
        start_index = 0

        for boundary_index in boundary_indices:
            if boundary_index > start_index:
                chunks.append(latex_text[start_index:boundary_index])
            start_index = boundary_index

        chunks.append(latex_text[start_index:])
        return [chunk for chunk in chunks if chunk]

    def _placeholder_inventory(self, text: str) -> Counter[str]:
        """Count placeholder tokens to verify output integrity."""

        return Counter(PLACEHOLDER_PATTERN.findall(text))

    def _assert_placeholder_integrity(self, masked_input: str, masked_output: str) -> None:
        """Ensure output has exactly the same placeholder inventory as input."""

        expected = self._placeholder_inventory(masked_input)
        actual = self._placeholder_inventory(masked_output)
        if expected != actual:
            raise PlaceholderIntegrityError(
                "Placeholder inventory mismatch after translation. "
                f"Expected={dict(expected)}, Actual={dict(actual)}"
            )

    def translate_text(
        self,
        masked_text: str,
        glossary: Dict[str, GlossaryEntry],
        max_integrity_retries: int = 1,
    ) -> str:
        glossary_context = self.glossary_manager.to_json_dict(glossary)

        attempts = 0
        last_error: PlaceholderIntegrityError | None = None

        while attempts <= max_integrity_retries:
            translated = self.llm_client.translate_text(masked_text, glossary_context)
            try:
                self._assert_placeholder_integrity(masked_text, translated)
                return translated
            except PlaceholderIntegrityError as exc:
                last_error = exc
                attempts += 1

        if last_error is not None:
            raise last_error
        raise PlaceholderIntegrityError("Unknown placeholder integrity failure.")

    def restore_latex_commands(self, text: str, placeholder_map: Dict[str, str]) -> str:
        return self.masker.restore_latex_commands(text, placeholder_map)

    def _assert_no_unresolved_placeholders(self, restored_text: str, placeholder_map: Dict[str, str]) -> None:
        """Abort if any injected placeholders survive restoration."""

        unresolved = [token for token in placeholder_map if token in restored_text]
        if unresolved:
            raise PlaceholderIntegrityError(
                "Unresolved placeholders remain after restoration. "
                f"Tokens={unresolved[:10]}"
            )

    def build_glossary_review_queue(
        self,
        glossary_source_path: str | Path,
        output_path: str | Path = "glossary_review_queue.json",
    ) -> dict:
        """Translate glossary descriptions and write pending review suggestions."""

        source_text = self.load_latex(glossary_source_path)
        extracted_glossary = self.extract_glossary(source_text)

        queue_items: list[dict] = []
        for key, entry in sorted(extracted_glossary.items()):
            masked_description, placeholder_map = self.mask_latex_commands(entry.de)
            translated_masked = self.translate_text(masked_description, extracted_glossary)
            translated_description = self.restore_latex_commands(translated_masked, placeholder_map)

            queue_items.append(
                {
                    "term": key,
                    "type": entry.type,
                    "abbreviation": entry.abbreviation,
                    "de": entry.de,
                    "en_suggested": translated_description,
                    "status": "pending_review",
                }
            )

        review_payload = {
            "source_file": str(glossary_source_path),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "count": len(queue_items),
            "items": queue_items,
        }
        save_json(output_path, review_payload)
        return review_payload

    def apply_glossary_review_queue(
        self,
        review_queue_path: str | Path,
        approved_only: bool = True,
    ) -> dict:
        """Apply approved glossary review suggestions into glossary.json storage."""

        review_data = load_json(review_queue_path, default={})
        items = review_data.get("items", [])
        if not isinstance(items, list):
            raise GlossaryReviewQueueError("Invalid review queue format: 'items' must be a list.")

        base_raw = load_json(self.glossary_path, default={})
        merged = self.glossary_manager.from_json_dict(base_raw)

        applied_count = 0
        skipped_count = 0

        for item in items:
            if not isinstance(item, dict):
                skipped_count += 1
                continue

            status = str(item.get("status", "")).strip().lower()
            if approved_only and status != "approved":
                skipped_count += 1
                continue

            term = str(item.get("term", "")).strip()
            if not term:
                skipped_count += 1
                continue

            suggested_en = str(item.get("en_suggested", "")).strip()
            if not suggested_en:
                skipped_count += 1
                continue

            existing = merged.get(term, GlossaryEntry(de=str(item.get("de", "")).strip(), en="", type="glossary"))
            merged[term] = GlossaryEntry(
                de=existing.de or str(item.get("de", "")).strip(),
                en=suggested_en,
                type=existing.type or str(item.get("type", "glossary")).strip(),
                abbreviation=existing.abbreviation or str(item.get("abbreviation", "")).strip(),
            )
            applied_count += 1

        save_json(self.glossary_path, self.glossary_manager.to_json_dict(merged))
        return {
            "applied": applied_count,
            "skipped": skipped_count,
            "approved_only": approved_only,
            "glossary_path": str(self.glossary_path),
        }

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

    def run(self, latex_path: str | Path, glossary_source_path: str | Path | None = None) -> PipelineResult:
        latex_text = self.load_latex(latex_path)
        extracted_glossary = self.extract_glossary(latex_text)

        if glossary_source_path is not None:
            self.sync_glossary_from_source(glossary_source_path)
            glossary_source_text = self.load_latex(glossary_source_path)
            extracted_glossary.update(self.extract_glossary(glossary_source_text))

        existing_raw = load_json(self.glossary_path, default={})
        base_glossary = self.glossary_manager.from_json_dict(existing_raw)

        working_glossary = dict(base_glossary)
        working_glossary.update(extracted_glossary)

        chunks = self.split_into_chunks_by_headings(latex_text)
        restored_chunks: list[str] = []

        for chunk in chunks:
            masked_text, placeholder_map = self.mask_latex_commands(chunk)
            translated_masked = self.translate_text(masked_text, working_glossary)
            restored_chunk = self.restore_latex_commands(translated_masked, placeholder_map)
            self._assert_no_unresolved_placeholders(restored_chunk, placeholder_map)
            restored_chunks.append(restored_chunk)

        restored_text = "".join(restored_chunks)

        suggested_terms = detect_candidate_terms(restored_text)
        updated_glossary = self.update_glossary(base_glossary, extracted_glossary, suggested_terms)

        return PipelineResult(
            translated_text=restored_text,
            glossary=updated_glossary,
            suggested_terms=suggested_terms,
            chunk_count=len(chunks),
        )

    def run_folder(
        self,
        input_folder: str | Path,
        output_folder: str | Path,
        glossary_source_path: str | Path,
        include_glob: str = "*.tex",
        exclude_globs: list[str] | None = None,
    ) -> dict:
        """Run translation pipeline for all matched LaTeX files in a folder."""

        input_root = Path(input_folder)
        output_root = Path(output_folder)
        output_root.mkdir(parents=True, exist_ok=True)

        sync_summary = self.sync_glossary_from_source(glossary_source_path)
        files = self.discover_latex_files(
            input_folder=input_root,
            include_glob=include_glob,
            exclude_globs=exclude_globs,
            glossary_source_path=glossary_source_path,
        )

        processed: list[dict] = []
        for file_path in files:
            result = self.run(file_path, glossary_source_path=glossary_source_path)
            rel_path = file_path.relative_to(input_root)
            output_path = output_root / rel_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            write_text(output_path, result.translated_text)

            processed.append(
                {
                    "input": str(file_path),
                    "output": str(output_path),
                    "chunks": result.chunk_count,
                    "suggested_terms": len(result.suggested_terms),
                }
            )

        return {
            "input_folder": str(input_root),
            "output_folder": str(output_root),
            "glossary_source": str(glossary_source_path),
            "glossary_sync": sync_summary,
            "processed_count": len(processed),
            "processed": processed,
        }
