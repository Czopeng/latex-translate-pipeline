"""CLI entry point for the translation pipeline skeleton."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import TranslationPipeline
from .translator import GeminiLLMClient, MockLLMClient
from .utils import save_json, write_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LaTeX German->English pipeline (architecture skeleton)")
    parser.add_argument("input", type=Path, nargs="?", default=None, help="Path to source LaTeX file")
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
    parser.add_argument(
        "--input-folder",
        type=Path,
        default=None,
        help="Folder containing LaTeX files to translate recursively",
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        default=None,
        help="Output folder for translated files in folder mode",
    )
    parser.add_argument(
        "--include-glob",
        type=str,
        default="*.tex",
        help="Glob for files to include in folder mode",
    )
    parser.add_argument(
        "--exclude-glob",
        action="append",
        default=[],
        help="Glob pattern to exclude in folder mode. Repeatable.",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=None,
        help="Optional JSON path to write folder-mode manifest",
    )
    parser.add_argument(
        "--build-term-review-queue",
        action="store_true",
        help="Generate term_review_queue.json from translated folder outputs",
    )
    parser.add_argument(
        "--term-review-output",
        type=Path,
        default=Path("term_review_queue.json"),
        help="Output path for generated term review queue JSON",
    )
    parser.add_argument(
        "--term-min-frequency",
        type=int,
        default=2,
        help="Minimum corpus frequency for term review suggestions",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Folder mode only: build manifest/queues without writing translated files",
    )
    parser.add_argument(
        "--stale-report-output",
        type=Path,
        default=None,
        help="Optional JSON path to write stale glossary term report in folder mode",
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

    if args.input_folder is not None:
        if args.glossary_source is None:
            raise ValueError("--glossary-source is required in folder mode.")
        if args.output_folder is None:
            raise ValueError("--output-folder is required in folder mode.")

        manifest = pipeline.run_folder(
            input_folder=args.input_folder,
            output_folder=args.output_folder,
            glossary_source_path=args.glossary_source,
            include_glob=args.include_glob,
            exclude_globs=args.exclude_glob,
            term_review_output_path=args.term_review_output if args.build_term_review_queue else None,
            term_min_frequency=args.term_min_frequency,
            dry_run=args.dry_run,
            stale_report_output_path=args.stale_report_output,
        )

        if args.manifest_output is not None:
            save_json(args.manifest_output, manifest)
            print(f"Manifest written: {args.manifest_output}")

        print(f"Folder translation completed. Files processed: {manifest['processed_count']}")
        return

    if args.input is None:
        raise ValueError("Provide either a source file path or --input-folder mode.")

    result = pipeline.run(args.input, glossary_source_path=args.glossary_source)

    if args.output is not None:
        write_text(args.output, result.translated_text)
    else:
        print(result.translated_text)

    print(f"Suggested glossary additions: {len(result.suggested_terms)}")


if __name__ == "__main__":
    main()
