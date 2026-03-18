"""Shared utility helpers for IO and lightweight text parsing."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Tuple


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


def detect_candidate_terms(text: str) -> list[str]:
    """Heuristic detector for potential technical terms to review."""

    candidates = re.findall(r"\b[A-Za-z][A-Za-z0-9-]{4,}\b", text)
    unique = sorted({term for term in candidates if not term.islower()})
    return unique
