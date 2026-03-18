"""Shared utility helpers for IO and lightweight text parsing."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Tuple


DEFAULT_TERM_STOPWORDS = {
    "table",
    "figure",
    "section",
    "chapter",
    "appendix",
    "equation",
    "result",
    "method",
    "introduction",
    "conclusion",
}


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def write_text(path: str | Path, content: str) -> None:
    Path(path).write_text(content, encoding="utf-8")


def load_json(path: str | Path, default: Any | None = None) -> Any:
    json_path = Path(path)
    if not json_path.exists():
        return {} if default is None else default
    return json.loads(json_path.read_text(encoding="utf-8"))


def save_json(path: str | Path, data: Any) -> None:
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def extract_braced_content(text: str, start_index: int) -> Tuple[str, int]:
    """Extract content from a balanced {...} block starting at start_index."""

    if start_index >= len(text) or text[start_index] != "{":
        raise ValueError("extract_braced_content requires start_index at '{'")

    depth = 0
    content_start = start_index + 1

    for idx in range(start_index, len(text)):
        char = text[idx]
        if char == "{" and (idx == 0 or text[idx - 1] != "\\"):
            depth += 1
        elif char == "}" and (idx == 0 or text[idx - 1] != "\\"):
            depth -= 1
            if depth == 0:
                return text[content_start:idx], idx

    raise ValueError("Unbalanced braces in input text")


def extract_candidate_term_counts(
    text: str,
    existing_terms: set[str] | None = None,
    stopwords: set[str] | None = None,
) -> dict[str, int]:
    """Count candidate technical terms while filtering known/noisy words."""

    existing = {term.lower() for term in (existing_terms or set())}
    blocked = {word.lower() for word in (stopwords or DEFAULT_TERM_STOPWORDS)}

    counts: dict[str, int] = {}
    for match in re.finditer(r"\b[A-Za-z][A-Za-z0-9-]{2,}\b", text):
        token = match.group(0)
        token_lower = token.lower()

        if token_lower in existing or token_lower in blocked:
            continue
        if token.islower():
            continue
        if len(token) < 3:
            continue

        counts[token] = counts.get(token, 0) + 1

    return counts


def rank_candidate_terms(counts: dict[str, int], min_frequency: int = 2) -> list[dict]:
    """Rank candidate terms by frequency and lexical signal."""

    ranked: list[dict] = []
    for term, count in counts.items():
        if count < min_frequency:
            continue

        token_bonus = 1.0 if any(char.isdigit() for char in term) else 0.0
        hyphen_bonus = 0.5 if "-" in term else 0.0
        acronym_bonus = 1.0 if term.isupper() and len(term) >= 3 else 0.0
        score = float(count) + token_bonus + hyphen_bonus + acronym_bonus

        ranked.append({"term": term, "count": count, "score": round(score, 3)})

    return sorted(ranked, key=lambda item: (-item["score"], -item["count"], item["term"]))


def detect_candidate_terms(
    text: str,
    existing_terms: set[str] | None = None,
    min_frequency: int = 1,
    stopwords: set[str] | None = None,
) -> list[str]:
    """Compatibility helper returning only term strings."""

    counts = extract_candidate_term_counts(text, existing_terms=existing_terms, stopwords=stopwords)
    ranked = rank_candidate_terms(counts, min_frequency=min_frequency)
    return [item["term"] for item in ranked]
