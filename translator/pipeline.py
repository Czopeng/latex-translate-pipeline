"""High-level translation pipeline orchestration."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import fnmatch
from time import perf_counter
from pathlib import Path
import re
from typing import Dict

from .glossary import GlossaryEntry, GlossaryManager
from .latex_masker import LatexMasker, PLACEHOLDER_PATTERN
from .translator import LLMClient
from .utils import (
    detect_candidate_terms,
    extract_candidate_term_counts,
    load_json,
    rank_candidate_terms,
    read_text,
    save_json,
    write_text,
)


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


class GlossarySourceValidationError(RuntimeError):
    """Raised when glossary source input does not contain parseable entries."""


class TranslationPipeline:
    """Modular German->English LaTeX translation pipeline skeleton."""

    _heading_boundary_pattern = re.compile(r"(?m)^(?=\\(?:chapter|section|subsection)\*?\{)")

    def __init__(
        self,
        llm_client: LLMClient,
        glossary_path: str | Path = "glossary.json",
        max_chunk_chars: int = 12000,
        max_chunk_tokens: int = 3000,
    ) -> None:
        self.llm_client = llm_client
        self.glossary_path = Path(glossary_path)
        self.max_chunk_chars = max_chunk_chars
        self.max_chunk_tokens = max_chunk_tokens
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
        if not source_entries:
            raise GlossarySourceValidationError(
                "Glossary source contains no parseable entries. "
                "Expected at least one \\newacronym or \\newglossaryentry definition."
            )

        existing_raw = load_json(self.glossary_path, default={})
        existing_entries = self.glossary_manager.from_json_dict(existing_raw)

        merged: Dict[str, GlossaryEntry] = {}
        preserved_en_count = 0
        stale_terms: list[str] = []

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
                stale_terms.append(term)
                merged[term] = existing_entry

        save_json(self.glossary_path, self.glossary_manager.to_json_dict(merged))
        stale_type_counts: dict[str, int] = {}
        for term in stale_terms:
            entry = existing_entries.get(term)
            stale_type = entry.type if entry is not None else "unknown"
            stale_type_counts[stale_type] = stale_type_counts.get(stale_type, 0) + 1

        return {
            "source_file": str(glossary_source_path),
            "source_terms": len(source_entries),
            "preserved_en": preserved_en_count,
            "total_terms_after_sync": len(merged),
            "stale_terms_count": len(stale_terms),
            "stale_type_counts": stale_type_counts,
            "stale_terms": sorted(stale_terms),
        }

    def build_stale_glossary_report(
        self,
        stale_terms: list[str],
        output_path: str | Path = "stale_glossary_report.json",
    ) -> dict:
        """Build report for glossary terms that are not present in source glossary files."""

        glossary_raw = load_json(self.glossary_path, default={})

        items: list[dict] = []
        for term in sorted(stale_terms):
            raw = glossary_raw.get(term, {})
            items.append(
                {
                    "term": term,
                    "status": "stale_non_source",
                    "type": str(raw.get("type", "")),
                    "abbreviation": str(raw.get("abbreviation", "")),
                    "de": str(raw.get("de", "")),
                    "en": str(raw.get("en", "")),
                }
            )

        payload = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "glossary_path": str(self.glossary_path),
            "count": len(items),
            "items": items,
        }
        save_json(output_path, payload)
        return payload

    @staticmethod
    def _sanitize_for_label(value: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9]+", "_", value.strip())
        cleaned = cleaned.strip("_")
        return cleaned or "term"

    def render_translated_glossary_latex(self, glossary: Dict[str, GlossaryEntry]) -> str:
        """Render a standalone translated glossary LaTeX artifact from glossary storage."""

        lines: list[str] = ["% Auto-generated translated glossary artifact"]

        for term in sorted(glossary):
            entry = glossary[term]
            en_value = entry.en.strip() or entry.de.strip()

            if entry.type == "acronym":
                abbr = entry.abbreviation.strip() or term
                lines.append(
                    f"\\newacronym{{{term}}}{{{abbr}}}{{{en_value}}}"
                )
                continue

            label = self._sanitize_for_label(term)
            lines.extend(
                [
                    f"\\newglossaryentry{{{label}}}",
                    "{",
                    f"  name={{{term}}},",
                    f"  description={{{en_value}}}",
                    "}",
                ]
            )

        return "\n".join(lines) + "\n"

    def generate_rollout_glossary_outputs(
        self,
        glossary_source_path: str | Path,
        output_folder: str | Path,
    ) -> dict:
        """Generate translated glossary outputs for first-rollout folder workflows."""

        output_root = Path(output_folder)
        output_root.mkdir(parents=True, exist_ok=True)

        queue_output_path = output_root / "glossary_review_queue.auto.json"
        queue_payload = self.build_glossary_review_queue(
            glossary_source_path=glossary_source_path,
            output_path=queue_output_path,
        )
        apply_result = self.apply_glossary_review_queue(queue_output_path, approved_only=False)

        glossary_entries = self.glossary_manager.from_json_dict(load_json(self.glossary_path, default={}))
        source_name = Path(glossary_source_path).name
        translated_latex_name = f"translated_{source_name}"
        translated_json_name = "translated_glossary.json"
        translated_latex_path = output_root / translated_latex_name
        translated_json_path = output_root / translated_json_name

        translated_latex = self.render_translated_glossary_latex(glossary_entries)
        write_text(translated_latex_path, translated_latex)
        save_json(translated_json_path, self.glossary_manager.to_json_dict(glossary_entries))

        return {
            "auto_applied": True,
            "queue": {
                "output": str(queue_output_path),
                "count": queue_payload.get("count", 0),
                "conflict_count": queue_payload.get("conflict_count", 0),
            },
            "apply": apply_result,
            "outputs": {
                "latex": str(translated_latex_path),
                "json": str(translated_json_path),
            },
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
        heading_chunks = [chunk for chunk in chunks if chunk]

        final_chunks: list[str] = []
        for heading_chunk in heading_chunks:
            final_chunks.extend(self._split_large_chunk(heading_chunk))
        return final_chunks

    def _split_large_chunk(self, chunk: str) -> list[str]:
        """Split oversized chunks on paragraph boundaries, preserving full content."""

        if not self._chunk_exceeds_budget(chunk):
            return [chunk]

        # Keep delimiters attached to segments so re-join remains byte-identical.
        segments: list[str] = []
        cursor = 0
        for match in re.finditer(r"\n\s*\n", chunk):
            end = match.end()
            segments.append(chunk[cursor:end])
            cursor = end
        if cursor < len(chunk):
            segments.append(chunk[cursor:])

        output_chunks: list[str] = []
        buffer = ""

        for segment in segments:
            # Hard split a segment that alone exceeds the limit.
            if self._chunk_exceeds_budget(segment):
                if buffer:
                    output_chunks.append(buffer)
                    buffer = ""
                start = 0
                while start < len(segment):
                    window = segment[start:start + self.max_chunk_chars]
                    while window and self._estimate_tokens(window) > self.max_chunk_tokens:
                        window = window[: max(1, len(window) // 2)]
                    output_chunks.append(window)
                    start += len(window)
                continue

            if not buffer:
                buffer = segment
                continue

            merged = buffer + segment
            if not self._chunk_exceeds_budget(merged):
                buffer += segment
            else:
                output_chunks.append(buffer)
                buffer = segment

        if buffer:
            output_chunks.append(buffer)

        # Safety: never lose or reorder content.
        if "".join(output_chunks) != chunk:
            raise RuntimeError("Chunk split failed to preserve original content")

        return output_chunks

    @staticmethod
    def _normalized_text(value: str) -> str:
        return " ".join(value.strip().lower().split())

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate used for pre-translation chunk budgeting."""

        return max(1, len(text) // 4)

    def _chunk_exceeds_budget(self, text: str) -> bool:
        return len(text) > self.max_chunk_chars or self._estimate_tokens(text) > self.max_chunk_tokens

    def _build_style_context(self, chunk: str) -> dict:
        """Build style metadata for chunk-level scientific writing guidance."""

        heading_match = re.search(r"\\(chapter|section|subsection)\*?\{", chunk)
        heading_level = heading_match.group(1) if heading_match else "body"
        return {
            "tone": "formal scientific",
            "domain": "technical engineering",
            "keep_units": True,
            "heading_level": heading_level,
        }

    def _summarize_glossary_conflicts(self) -> dict:
        """Summarize glossary consistency risks for manifests."""

        raw = load_json(self.glossary_path, default={})
        glossary = self.glossary_manager.from_json_dict(raw)

        missing_en_count = 0
        de_to_en: dict[str, set[str]] = {}

        for entry in glossary.values():
            if not entry.en.strip():
                missing_en_count += 1

            de_key = self._normalized_text(entry.de)
            en_val = self._normalized_text(entry.en)
            if not de_key:
                continue

            de_to_en.setdefault(de_key, set())
            if en_val:
                de_to_en[de_key].add(en_val)

        conflicting_de_entries = sum(1 for variants in de_to_en.values() if len(variants) > 1)
        return {
            "missing_en_count": missing_en_count,
            "conflicting_de_entries": conflicting_de_entries,
            "total_terms": len(glossary),
        }

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
        style_context: dict | None = None,
    ) -> str:
        glossary_context = self.glossary_manager.to_json_dict(glossary)
        if style_context is not None:
            glossary_context["__style__"] = style_context

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
        existing_glossary = self.glossary_manager.from_json_dict(load_json(self.glossary_path, default={}))

        queue_items: list[dict] = []
        conflict_count = 0
        for key, entry in sorted(extracted_glossary.items()):
            masked_description, placeholder_map = self.mask_latex_commands(entry.de)
            translated_masked = self.translate_text(masked_description, extracted_glossary)
            translated_description = self.restore_latex_commands(translated_masked, placeholder_map)

            status = "pending_review"
            existing = existing_glossary.get(key)
            existing_en = existing.en if existing else ""

            if existing_en and self._normalized_text(existing_en) != self._normalized_text(translated_description):
                status = "conflict_review"
                conflict_count += 1

            queue_items.append(
                {
                    "term": key,
                    "type": entry.type,
                    "abbreviation": entry.abbreviation,
                    "de": entry.de,
                    "en_suggested": translated_description,
                    "en_existing": existing_en,
                    "status": status,
                }
            )

        review_payload = {
            "source_file": str(glossary_source_path),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "count": len(queue_items),
            "conflict_count": conflict_count,
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

        # Suggested terms are handled via review queue and not auto-committed.
        _ = suggested_terms

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
            translated_masked = self.translate_text(
                masked_text,
                working_glossary,
                style_context=self._build_style_context(chunk),
            )
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

    def build_term_review_queue(
        self,
        translated_documents: list[dict],
        output_path: str | Path = "term_review_queue.json",
        min_frequency: int = 2,
    ) -> dict:
        """Build a review queue for newly observed candidate terms with source evidence."""

        glossary_raw = load_json(self.glossary_path, default={})
        existing_terms = set(glossary_raw.keys())

        aggregate_counts: dict[str, int] = {}
        evidence: dict[str, dict[str, int]] = {}

        for doc in translated_documents:
            text = str(doc.get("text", ""))
            source = str(doc.get("source", ""))
            counts = extract_candidate_term_counts(text, existing_terms=existing_terms)

            for term, count in counts.items():
                aggregate_counts[term] = aggregate_counts.get(term, 0) + count
                evidence.setdefault(term, {})
                evidence[term][source] = evidence[term].get(source, 0) + count

        ranked = rank_candidate_terms(aggregate_counts, min_frequency=min_frequency)

        items: list[dict] = []
        for item in ranked:
            term = item["term"]
            files = sorted(evidence.get(term, {}).items(), key=lambda pair: (-pair[1], pair[0]))
            items.append(
                {
                    "term": term,
                    "count": item["count"],
                    "score": item["score"],
                    "status": "pending_review",
                    "sources": [{"file": file_path, "count": count} for file_path, count in files],
                }
            )

        payload = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "count": len(items),
            "min_frequency": min_frequency,
            "items": items,
        }
        save_json(output_path, payload)
        return payload

    def run_folder(
        self,
        input_folder: str | Path,
        output_folder: str | Path,
        glossary_source_path: str | Path,
        include_glob: str = "*.tex",
        exclude_globs: list[str] | None = None,
        term_review_output_path: str | Path | None = None,
        term_min_frequency: int = 2,
        dry_run: bool = False,
        stale_report_output_path: str | Path | None = None,
    ) -> dict:
        """Run translation pipeline for all matched LaTeX files in a folder."""

        input_root = Path(input_folder)
        output_root = Path(output_folder)
        output_root.mkdir(parents=True, exist_ok=True)

        glossary_before_count = len(load_json(self.glossary_path, default={}))

        sync_summary = self.sync_glossary_from_source(glossary_source_path)
        files = self.discover_latex_files(
            input_folder=input_root,
            include_glob=include_glob,
            exclude_globs=exclude_globs,
            glossary_source_path=glossary_source_path,
        )

        processed: list[dict] = []
        translated_documents: list[dict] = []
        failures: list[dict] = []
        folder_start = perf_counter()

        glossary_rollout_summary: dict | None = None
        if not dry_run:
            try:
                glossary_rollout_summary = self.generate_rollout_glossary_outputs(
                    glossary_source_path=glossary_source_path,
                    output_folder=output_root,
                )
            except Exception as exc:  # pragma: no cover - guarded by manifest assertions in tests
                failures.append(
                    {
                        "stage": "glossary_rollout",
                        "input": str(glossary_source_path),
                        "output": str(output_root),
                        "error": str(exc),
                    }
                )

        for file_path in files:
            file_start = perf_counter()
            rel_path = file_path.relative_to(input_root)
            output_path = output_root / rel_path
            expected_output = output_root / rel_path

            try:
                actual_rel = output_path.relative_to(output_root)
            except ValueError as exc:
                failures.append(
                    {
                        "stage": "output_path_validation",
                        "input": str(file_path),
                        "output": str(output_path),
                        "error": str(exc),
                    }
                )
                continue

            if actual_rel != rel_path or output_path != expected_output:
                failures.append(
                    {
                        "stage": "output_path_validation",
                        "input": str(file_path),
                        "output": str(output_path),
                        "error": "Output path does not mirror input relative path.",
                    }
                )
                continue

            if dry_run:
                source_text = self.load_latex(file_path)
                estimated_chunks = len(self.split_into_chunks_by_headings(source_text))
                file_duration = perf_counter() - file_start
                processed.append(
                    {
                        "input": str(file_path),
                        "output": str(output_path),
                        "written": False,
                        "dry_run": True,
                        "chunks": estimated_chunks,
                        "suggested_terms": 0,
                        "duration_seconds": round(file_duration, 6),
                    }
                )
                translated_documents.append({"source": str(file_path), "text": source_text})
            else:
                try:
                    result = self.run(file_path, glossary_source_path=glossary_source_path)
                    file_duration = perf_counter() - file_start
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    write_text(output_path, result.translated_text)

                    processed.append(
                        {
                            "input": str(file_path),
                            "output": str(output_path),
                            "written": True,
                            "dry_run": False,
                            "chunks": result.chunk_count,
                            "suggested_terms": len(result.suggested_terms),
                            "duration_seconds": round(file_duration, 6),
                        }
                    )
                    translated_documents.append({"source": str(file_path), "text": result.translated_text})
                except Exception as exc:  # pragma: no cover - asserted through manifest failure reporting tests
                    file_duration = perf_counter() - file_start
                    processed.append(
                        {
                            "input": str(file_path),
                            "output": str(output_path),
                            "written": False,
                            "dry_run": False,
                            "chunks": 0,
                            "suggested_terms": 0,
                            "duration_seconds": round(file_duration, 6),
                            "status": "failed",
                        }
                    )
                    failures.append(
                        {
                            "stage": "file_translation_write",
                            "input": str(file_path),
                            "output": str(output_path),
                            "error": str(exc),
                        }
                    )

        total_duration = perf_counter() - folder_start
        glossary_after_count = len(load_json(self.glossary_path, default={}))

        term_review_summary: dict | None = None
        if term_review_output_path is not None:
            term_review_payload = self.build_term_review_queue(
                translated_documents=translated_documents,
                output_path=term_review_output_path,
                min_frequency=term_min_frequency,
            )
            term_review_summary = {
                "output": str(term_review_output_path),
                "count": term_review_payload["count"],
                "min_frequency": term_review_payload["min_frequency"],
            }

        stale_report_summary: dict | None = None
        if stale_report_output_path is not None:
            stale_report_payload = self.build_stale_glossary_report(
                stale_terms=sync_summary.get("stale_terms", []),
                output_path=stale_report_output_path,
            )
            stale_report_summary = {
                "output": str(stale_report_output_path),
                "count": stale_report_payload["count"],
            }

        return {
            "input_folder": str(input_root),
            "output_folder": str(output_root),
            "dry_run": dry_run,
            "glossary_source": str(glossary_source_path),
            "glossary_rollout": glossary_rollout_summary,
            "glossary_sync": sync_summary,
            "glossary_conflicts": self._summarize_glossary_conflicts(),
            "stale_glossary": {
                "count": sync_summary.get("stale_terms_count", 0),
                "type_counts": sync_summary.get("stale_type_counts", {}),
                "terms_preview": sync_summary.get("stale_terms", [])[:25],
                "report": stale_report_summary,
            },
            "processed_count": len(processed),
            "failed_count": len(failures),
            "success_count": max(0, len(processed) - len(failures)),
            "failures": failures,
            "duration_seconds": round(total_duration, 6),
            "glossary_delta": {
                "before_count": glossary_before_count,
                "after_count": glossary_after_count,
                "added": max(0, glossary_after_count - glossary_before_count),
            },
            "term_review": term_review_summary,
            "processed": processed,
        }
