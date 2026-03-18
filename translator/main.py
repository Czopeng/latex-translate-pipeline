"""CLI entry point for the translation pipeline skeleton."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import TranslationPipeline
from .translator import MockLLMClient
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
    return parser


def main() -> None:
    args = build_parser().parse_args()

    pipeline = TranslationPipeline(llm_client=MockLLMClient(), glossary_path=args.glossary)
    result = pipeline.run(args.input)

    if args.output is not None:
        write_text(args.output, result.translated_text)
    else:
        print(result.translated_text)

    print(f"Suggested glossary additions: {len(result.suggested_terms)}")


if __name__ == "__main__":
    main()
