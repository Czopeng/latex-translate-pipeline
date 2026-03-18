"""CLI entry point for the translation pipeline skeleton."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import TranslationPipeline
from .translator import GeminiLLMClient, MockLLMClient
from .utils import write_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LaTeX German->English pipeline (architecture skeleton)")
    parser.add_argument("input", type=Path, help="Path to source LaTeX file")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path for translated LaTeX text",
    )
    parser.add_argument(
        "--glossary",
        type=Path,
        default=Path("glossary.json"),
        help="Path to glossary JSON store",
    )
    parser.add_argument(
        "--glossary-source",
        type=Path,
        default=None,
        help="Optional LaTeX file containing glossary definitions (e.g. glossarentries.tex)",
    )
    parser.add_argument(
        "--provider",
        choices=["mock", "gemini"],
        default="mock",
        help="Translation provider backend",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.0-flash",
        help="Model name when using the gemini provider",
    )
    parser.add_argument(
        "--build-glossary-review-queue",
        action="store_true",
        help="Generate glossary_review_queue.json from a glossary LaTeX source file",
    )
    parser.add_argument(
        "--review-output",
        type=Path,
        default=Path("glossary_review_queue.json"),
        help="Output path for glossary review queue JSON",
    )
    parser.add_argument(
        "--apply-glossary-review-queue",
        action="store_true",
        help="Apply approved glossary suggestions from a review queue JSON file",
    )
    parser.add_argument(
        "--review-input",
        type=Path,
        default=Path("glossary_review_queue.json"),
        help="Input review queue JSON path used with --apply-glossary-review-queue",
    )
    parser.add_argument(
        "--include-pending",
        action="store_true",
        help="Apply all queue entries regardless of status (default applies only approved)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    llm_client = MockLLMClient() if args.provider == "mock" else GeminiLLMClient(model=args.model)

    pipeline = TranslationPipeline(llm_client=llm_client, glossary_path=args.glossary)

    if args.build_glossary_review_queue:
        source_path = args.glossary_source or args.input
        queue = pipeline.build_glossary_review_queue(source_path, args.review_output)
        print(f"Glossary review queue generated: {args.review_output}")
        print(f"Queued entries: {queue['count']}")
        return

    if args.apply_glossary_review_queue:
        result = pipeline.apply_glossary_review_queue(
            args.review_input,
            approved_only=not args.include_pending,
        )
        print(f"Applied glossary review queue: {args.review_input}")
        print(f"Applied: {result['applied']}, Skipped: {result['skipped']}")
        return

    result = pipeline.run(args.input, glossary_source_path=args.glossary_source)

    if args.output is not None:
        write_text(args.output, result.translated_text)
    else:
        print(result.translated_text)

    print(f"Suggested glossary additions: {len(result.suggested_terms)}")


if __name__ == "__main__":
    main()
