"""Integration checks against immutable fixture files in testLatex/."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from translator.glossary import GlossaryManager
from translator.pipeline import PlaceholderIntegrityError, TranslationPipeline
from translator.translator import MockLLMClient
from translator.utils import load_json


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "testLatex"


class PipelineFixtureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.glossary_file = ROOT / "tests" / "_tmp_glossary_fixture.json"
        self.review_queue_file = ROOT / "tests" / "_tmp_glossary_review_queue.json"
        self.review_queue_input_file = ROOT / "tests" / "_tmp_review_input.json"
        if self.glossary_file.exists():
            self.glossary_file.unlink()
        if self.review_queue_file.exists():
            self.review_queue_file.unlink()
        if self.review_queue_input_file.exists():
            self.review_queue_input_file.unlink()
        self.pipeline = TranslationPipeline(llm_client=MockLLMClient(), glossary_path=self.glossary_file)

    def tearDown(self) -> None:
        if self.glossary_file.exists():
            self.glossary_file.unlink()
        if self.review_queue_file.exists():
            self.review_queue_file.unlink()
        if self.review_queue_input_file.exists():
            self.review_queue_input_file.unlink()

    def test_glossary_extraction_from_glossarentries(self) -> None:
        content = (FIXTURES / "glossarentries.tex").read_text(encoding="utf-8")
        glossary = GlossaryManager().extract_glossary(content)

        self.assertIn("WBG", glossary)
        self.assertEqual(glossary["WBG"].type, "acronym")
        self.assertIn("BNC", glossary)

    def test_pipeline_round_trip_on_theorie_with_external_glossary(self) -> None:
        result = self.pipeline.run(
            FIXTURES / "Theorie.tex",
            glossary_source_path=FIXTURES / "glossarentries.tex",
        )

        original = (FIXTURES / "Theorie.tex").read_text(encoding="utf-8")
        self.assertEqual(result.translated_text, original)
        self.assertIn("WBG", result.glossary)

    def test_pipeline_round_trip_on_versuch_with_external_glossary(self) -> None:
        result = self.pipeline.run(
            FIXTURES / "Versuch.tex",
            glossary_source_path=FIXTURES / "glossarentries.tex",
        )

        original = (FIXTURES / "Versuch.tex").read_text(encoding="utf-8")
        self.assertEqual(result.translated_text, original)
        self.assertIn("IGBT", result.glossary)

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


if __name__ == "__main__":
    unittest.main()
