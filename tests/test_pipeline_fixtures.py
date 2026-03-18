"""Integration checks against immutable fixture files in testLatex/."""

from __future__ import annotations

import json
from pathlib import Path
import unittest
from unittest.mock import patch

import translator.pipeline as pipeline_module
from translator.glossary import GlossaryManager
from translator.pipeline import GlossarySourceValidationError, PlaceholderIntegrityError, TranslationPipeline
from translator.translator import MockLLMClient
from translator.utils import load_json


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "testLatex"


class PipelineFixtureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.glossary_file = ROOT / "tests" / "_tmp_glossary_fixture.json"
        self.review_queue_file = ROOT / "tests" / "_tmp_glossary_review_queue.json"
        self.review_queue_input_file = ROOT / "tests" / "_tmp_review_input.json"
        self.term_review_file = ROOT / "tests" / "_tmp_term_review_queue.json"
        self.stale_report_file = ROOT / "tests" / "_tmp_stale_glossary_report.json"
        self.output_folder = ROOT / "tests" / "_tmp_output_folder"
        if self.glossary_file.exists():
            self.glossary_file.unlink()
        if self.review_queue_file.exists():
            self.review_queue_file.unlink()
        if self.review_queue_input_file.exists():
            self.review_queue_input_file.unlink()
        if self.term_review_file.exists():
            self.term_review_file.unlink()
        if self.stale_report_file.exists():
            self.stale_report_file.unlink()
        if self.output_folder.exists():
            for item in sorted(self.output_folder.rglob("*"), reverse=True):
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    item.rmdir()
            self.output_folder.rmdir()
        self.pipeline = TranslationPipeline(llm_client=MockLLMClient(), glossary_path=self.glossary_file)

    def tearDown(self) -> None:
        if self.glossary_file.exists():
            self.glossary_file.unlink()
        if self.review_queue_file.exists():
            self.review_queue_file.unlink()
        if self.review_queue_input_file.exists():
            self.review_queue_input_file.unlink()
        if self.term_review_file.exists():
            self.term_review_file.unlink()
        if self.stale_report_file.exists():
            self.stale_report_file.unlink()
        if self.output_folder.exists():
            for item in sorted(self.output_folder.rglob("*"), reverse=True):
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    item.rmdir()
            self.output_folder.rmdir()

    def test_glossary_extraction_from_glossarentries(self) -> None:
        content = (FIXTURES / "glossarentries.tex").read_text(encoding="utf-8")
        glossary = GlossaryManager().extract_glossary(content)

        self.assertIn("WBG", glossary)
        self.assertEqual(glossary["WBG"].type, "acronym")
        self.assertIn("BNC", glossary)

    def test_split_into_chunks_by_headings_from_theorie(self) -> None:
        theorie = (FIXTURES / "Theorie.tex").read_text(encoding="utf-8")
        chunks = self.pipeline.split_into_chunks_by_headings(theorie)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(any("\\chapter{" in chunk for chunk in chunks))
        self.assertEqual(theorie, "".join(chunks))

    def test_pipeline_round_trip_on_theorie_with_external_glossary(self) -> None:
        result = self.pipeline.run(
            FIXTURES / "Theorie.tex",
            glossary_source_path=FIXTURES / "glossarentries.tex",
        )

        original = (FIXTURES / "Theorie.tex").read_text(encoding="utf-8")
        self.assertEqual(result.translated_text, original)
        self.assertIn("WBG", result.glossary)
        self.assertGreater(result.chunk_count, 1)

    def test_pipeline_round_trip_on_versuch_with_external_glossary(self) -> None:
        result = self.pipeline.run(
            FIXTURES / "Versuch.tex",
            glossary_source_path=FIXTURES / "glossarentries.tex",
        )

        original = (FIXTURES / "Versuch.tex").read_text(encoding="utf-8")
        self.assertEqual(result.translated_text, original)
        self.assertIn("IGBT", result.glossary)
        self.assertGreater(result.chunk_count, 0)

    def test_placeholder_integrity_error_on_modified_tokens(self) -> None:
        class BrokenPlaceholderClient:
            def translate_text(self, text: str, glossary_context: dict) -> str:
                _ = glossary_context
                # Simulate LLM corruption: one placeholder token is modified.
                return text.replace("__CMD", "__CMX", 1)

        broken_pipeline = TranslationPipeline(
            llm_client=BrokenPlaceholderClient(),
            glossary_path=self.glossary_file,
        )

        with self.assertRaises(PlaceholderIntegrityError):
            broken_pipeline.run(
                FIXTURES / "Theorie.tex",
                glossary_source_path=FIXTURES / "glossarentries.tex",
            )

    def test_placeholder_integrity_retry_can_recover(self) -> None:
        class RetryOnceClient:
            def __init__(self) -> None:
                self.calls = 0

            def translate_text(self, text: str, glossary_context: dict) -> str:
                _ = glossary_context
                self.calls += 1
                if self.calls == 1:
                    return text.replace("__CMD", "__BROKEN", 1)
                return text

        recover_pipeline = TranslationPipeline(
            llm_client=RetryOnceClient(),
            glossary_path=self.glossary_file,
        )
        result = recover_pipeline.run(
            FIXTURES / "Versuch.tex",
            glossary_source_path=FIXTURES / "glossarentries.tex",
        )
        original = (FIXTURES / "Versuch.tex").read_text(encoding="utf-8")
        self.assertEqual(result.translated_text, original)

    def test_build_glossary_review_queue_from_fixture(self) -> None:
        queue = self.pipeline.build_glossary_review_queue(
            FIXTURES / "glossarentries.tex",
            output_path=self.review_queue_file,
        )

        self.assertTrue(self.review_queue_file.exists())
        payload = load_json(self.review_queue_file)
        self.assertEqual(payload["count"], len(payload["items"]))
        self.assertGreater(payload["count"], 0)
        self.assertIn("WBG", {item["term"] for item in payload["items"]})
        self.assertEqual("pending_review", payload["items"][0]["status"])
        self.assertEqual(queue["count"], payload["count"])

    def test_apply_glossary_review_queue_approved_only(self) -> None:
        input_payload = {
            "items": [
                {
                    "term": "WBG",
                    "type": "acronym",
                    "abbreviation": "WBG",
                    "de": "wide-bandgap",
                    "en_suggested": "wide-bandgap",
                    "status": "approved",
                },
                {
                    "term": "IGBT",
                    "type": "acronym",
                    "abbreviation": "IGBT",
                    "de": "Insulated Gate Bipolar Transistor",
                    "en_suggested": "Insulated Gate Bipolar Transistor",
                    "status": "pending_review",
                },
            ]
        }
        self.review_queue_input_file.write_text(
            json.dumps(input_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        outcome = self.pipeline.apply_glossary_review_queue(self.review_queue_input_file, approved_only=True)
        glossary_after = load_json(self.glossary_file)

        self.assertEqual(1, outcome["applied"])
        self.assertIn("WBG", glossary_after)
        self.assertEqual("wide-bandgap", glossary_after["WBG"]["en"])
        self.assertNotIn("IGBT", glossary_after)

    def test_fallback_chunk_split_preserves_content_for_large_sections(self) -> None:
        small_chunk_pipeline = TranslationPipeline(
            llm_client=MockLLMClient(),
            glossary_path=self.glossary_file,
            max_chunk_chars=120,
        )
        large_text = "\\section{A}\\n" + ("Absatz eins.\\n\\n" * 40)

        chunks = small_chunk_pipeline.split_into_chunks_by_headings(large_text)

        self.assertGreater(len(chunks), 1)
        self.assertEqual(large_text, "".join(chunks))

    def test_token_budget_split_preserves_content(self) -> None:
        token_limited_pipeline = TranslationPipeline(
            llm_client=MockLLMClient(),
            glossary_path=self.glossary_file,
            max_chunk_chars=10000,
            max_chunk_tokens=20,
        )
        large_text = "\\section{Token}\\n" + ("Dieser Text ist lang genug fuer token budget splitting. " * 20)

        chunks = token_limited_pipeline.split_into_chunks_by_headings(large_text)

        self.assertGreater(len(chunks), 1)
        self.assertEqual(large_text, "".join(chunks))

    def test_build_glossary_review_queue_marks_conflicts(self) -> None:
        initial = {
            "WBG": {"de": "wide-bandgap", "en": "wide band gap semiconductor", "type": "acronym", "abbreviation": "WBG"}
        }
        self.glossary_file.write_text(json.dumps(initial, ensure_ascii=False, indent=2), encoding="utf-8")

        payload = self.pipeline.build_glossary_review_queue(
            FIXTURES / "glossarentries.tex",
            output_path=self.review_queue_file,
        )

        wbg_items = [item for item in payload["items"] if item["term"] == "WBG"]
        self.assertEqual(1, len(wbg_items))
        self.assertEqual("conflict_review", wbg_items[0]["status"])
        self.assertGreaterEqual(payload["conflict_count"], 1)

    def test_sync_glossary_from_source_preserves_existing_english(self) -> None:
        initial = {
            "WBG": {"de": "old", "en": "wide-bandgap", "type": "acronym", "abbreviation": "WBG"},
            "LEGACY_TERM": {"de": "Altbegriff", "en": "legacy term", "type": "suggested", "abbreviation": ""},
        }
        self.glossary_file.write_text(json.dumps(initial, ensure_ascii=False, indent=2), encoding="utf-8")

        summary = self.pipeline.sync_glossary_from_source(FIXTURES / "glossarentries.tex")
        synced = load_json(self.glossary_file)

        self.assertGreater(summary["source_terms"], 0)
        self.assertEqual("wide-bandgap", synced["WBG"]["en"])
        self.assertIn("LEGACY_TERM", synced)
        self.assertIn("LEGACY_TERM", summary["stale_terms"])
        self.assertGreaterEqual(summary["stale_terms_count"], 1)

    def test_run_does_not_auto_commit_suggested_terms(self) -> None:
        result = self.pipeline.run(
            FIXTURES / "Versuch.tex",
            glossary_source_path=FIXTURES / "glossarentries.tex",
        )
        stored_glossary = load_json(self.glossary_file)

        self.assertGreater(len(result.suggested_terms), 0)
        for term in result.suggested_terms[:10]:
            self.assertNotEqual("suggested", stored_glossary.get(term, {}).get("type"))

    def test_run_folder_processes_fixture_files(self) -> None:
        manifest = self.pipeline.run_folder(
            input_folder=FIXTURES,
            output_folder=self.output_folder,
            glossary_source_path=FIXTURES / "glossarentries.tex",
            term_review_output_path=self.term_review_file,
            term_min_frequency=2,
            stale_report_output_path=self.stale_report_file,
        )

        output_theorie = self.output_folder / "Theorie.tex"
        output_versuch = self.output_folder / "Versuch.tex"
        output_glossary_latex = self.output_folder / "translated_glossarentries.tex"
        output_glossary_json = self.output_folder / "translated_glossary.json"

        self.assertTrue(output_theorie.exists())
        self.assertTrue(output_versuch.exists())
        self.assertTrue(output_glossary_latex.exists())
        self.assertTrue(output_glossary_json.exists())
        self.assertFalse(manifest["dry_run"])
        self.assertGreaterEqual(manifest["processed_count"], 2)
        self.assertIn("duration_seconds", manifest)
        self.assertIn("glossary_delta", manifest)
        self.assertIn("glossary_conflicts", manifest)
        self.assertIn("missing_en_count", manifest["glossary_conflicts"])
        self.assertIn("stale_glossary", manifest)
        self.assertIn("glossary_rollout", manifest)
        self.assertIn("term_review", manifest)
        self.assertIn("duration_seconds", manifest["processed"][0])
        self.assertTrue(manifest["processed"][0]["written"])
        self.assertTrue(manifest["glossary_rollout"]["auto_applied"])
        self.assertIn("outputs", manifest["glossary_rollout"])
        self.assertIn("latex", manifest["glossary_rollout"]["outputs"])
        self.assertIn("json", manifest["glossary_rollout"]["outputs"])
        self.assertTrue(self.term_review_file.exists())
        self.assertTrue(self.stale_report_file.exists())

        term_payload = load_json(self.term_review_file)
        if term_payload["count"] > 0:
            first = term_payload["items"][0]
            self.assertIn("sources", first)
            self.assertGreaterEqual(len(first["sources"]), 1)
            self.assertIn("file", first["sources"][0])
            self.assertIn("count", first["sources"][0])
        self.assertEqual(
            (FIXTURES / "Theorie.tex").read_text(encoding="utf-8"),
            output_theorie.read_text(encoding="utf-8"),
        )

    def test_run_folder_dry_run_builds_manifest_without_writing_outputs(self) -> None:
        manifest = self.pipeline.run_folder(
            input_folder=FIXTURES,
            output_folder=self.output_folder,
            glossary_source_path=FIXTURES / "glossarentries.tex",
            term_review_output_path=self.term_review_file,
            term_min_frequency=2,
            dry_run=True,
            stale_report_output_path=self.stale_report_file,
        )

        self.assertTrue(manifest["dry_run"])
        self.assertGreaterEqual(manifest["processed_count"], 2)
        self.assertIn("stale_glossary", manifest)
        self.assertIsNotNone(manifest["stale_glossary"]["report"])
        self.assertIsNone(manifest["glossary_rollout"])
        self.assertTrue(self.term_review_file.exists())
        self.assertTrue(self.stale_report_file.exists())

        for item in manifest["processed"]:
            self.assertTrue(item["dry_run"])
            self.assertFalse(item["written"])
            self.assertFalse(Path(item["output"]).exists())

    def test_empty_glossary_source_is_rejected(self) -> None:
        empty_glossary_source = ROOT / "tests" / "_tmp_empty_glossary.tex"
        empty_glossary_source.write_text("% no glossary definitions\n", encoding="utf-8")

        try:
            with self.assertRaises(GlossarySourceValidationError):
                self.pipeline.run_folder(
                    input_folder=FIXTURES,
                    output_folder=self.output_folder,
                    glossary_source_path=empty_glossary_source,
                )
        finally:
            if empty_glossary_source.exists():
                empty_glossary_source.unlink()

    def test_run_folder_reports_partial_write_failures(self) -> None:
        original_write_text = pipeline_module.write_text

        def fail_once(path: str | Path, text: str) -> None:
            if str(path).endswith("Versuch.tex"):
                raise OSError("simulated write failure")
            original_write_text(path, text)

        with patch("translator.pipeline.write_text", side_effect=fail_once):
            manifest = self.pipeline.run_folder(
                input_folder=FIXTURES,
                output_folder=self.output_folder,
                glossary_source_path=FIXTURES / "glossarentries.tex",
            )

        self.assertGreaterEqual(manifest["failed_count"], 1)
        self.assertGreaterEqual(len(manifest["failures"]), 1)
        self.assertGreaterEqual(manifest["processed_count"], manifest["success_count"])
        self.assertTrue(any(item.get("status") == "failed" for item in manifest["processed"]))


if __name__ == "__main__":
    unittest.main()
