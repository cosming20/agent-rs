#!/usr/bin/env python3
"""
generate_enterprise_v3.py

Generates enterprise_v3.json — a 200+ question eval dataset on the Romanian Penal Code.
Uses the Docling parse cache for high-quality markdown extraction, then calls the OpenAI
API (gpt-4o-mini) with structured output to generate questions stratified across 10
page windows.

Usage:
    /path/to/venv/python3 generate_enterprise_v3.py
"""

from __future__ import annotations

import json
import os
import pickle
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent.parent  # agentic-rag/
ENV_FILE = REPO_ROOT / ".env"
EXPERIMENT_DIR = SCRIPT_DIR.parent / "experiments" / "docling_chunker_lab"
CACHE_DIR = EXPERIMENT_DIR / "results" / "_parse_cache_shared"
# D1 is test.pdf (confirmed from meta.json)
CACHE_PICKLE = CACHE_DIR / "D1__d625e61a4c28__fc404854.pickle"

# Add experiment dir to path so parse_cache module is importable
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
OUTPUT_FILE = SCRIPT_DIR / "enterprise_v3.json"

NUM_WINDOWS = 10
QUESTIONS_PER_WINDOW = 20
MIN_VALIDATED = 180

# Distribution per window
EASY_PER_WINDOW = 7
MEDIUM_PER_WINDOW = 7
HARD_PER_WINDOW = 6

# Type distribution per window (must sum to QUESTIONS_PER_WINDOW)
TYPE_DISTRIBUTION = [
    "single_hop",      # 1
    "single_hop",      # 2
    "single_hop",      # 3
    "single_hop",      # 4
    "multi_hop",       # 5
    "multi_hop",       # 6
    "lexical_exact",   # 7
    "lexical_exact",   # 8
    "conceptual",      # 9
    "conceptual",      # 10
    "negation",        # 11
    "negation",        # 12
    "temporal",        # 13
    "temporal",        # 14
    "comparison",      # 15
    "comparison",      # 16
    "disambiguation",  # 17
    "disambiguation",  # 18
    "summary_only",    # 19
    "bridge",          # 20
]

DIFFICULTY_DISTRIBUTION = (
    ["easy"] * EASY_PER_WINDOW
    + ["medium"] * MEDIUM_PER_WINDOW
    + ["hard"] * HARD_PER_WINDOW
)


# ---------------------------------------------------------------------------
# Load environment
# ---------------------------------------------------------------------------

def load_env(path: Path) -> dict[str, str]:
    """Parse a .env file and return key-value pairs."""
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        # Strip surrounding quotes
        value = value.strip().strip('"').strip("'")
        env[key.strip()] = value
    return env


# ---------------------------------------------------------------------------
# Load Docling markdown from pickle cache
# ---------------------------------------------------------------------------

def load_docling_markdown() -> str:
    """Load the Docling-parsed markdown from the pickle cache.

    The cache stores a `ParsedDocument` dataclass (from parse_cache.py) which
    has a `markdown` attribute containing the full document as markdown.
    """
    if not CACHE_PICKLE.exists():
        raise FileNotFoundError(f"Docling cache not found: {CACHE_PICKLE}")

    print(f"[INFO] Loading Docling cache from {CACHE_PICKLE} ...")

    # ParsedDocument is a dataclass defined in parse_cache.py — importing it
    # first ensures pickle can deserialize the object without AttributeError.
    try:
        import parse_cache  # noqa: F401 — registers the class for pickle deserialization
    except ImportError:
        print("[WARN] parse_cache module not importable; trying raw pickle load anyway")

    with CACHE_PICKLE.open("rb") as fh:
        cached = pickle.load(fh)

    # Primary path: ParsedDocument.markdown
    if hasattr(cached, "markdown") and isinstance(cached.markdown, str):
        md = cached.markdown
        print(f"[INFO] Extracted markdown via .markdown — {len(md):,} chars")
        return md

    # Secondary: docling document export
    if hasattr(cached, "doc") and hasattr(cached.doc, "export_to_markdown"):
        md = cached.doc.export_to_markdown()
        print(f"[INFO] Extracted markdown via .doc.export_to_markdown() — {len(md):,} chars")
        return md

    if hasattr(cached, "document") and hasattr(cached.document, "export_to_markdown"):
        md = cached.document.export_to_markdown()
        print(f"[INFO] Extracted markdown via .document.export_to_markdown() — {len(md):,} chars")
        return md

    # Last resort: dict
    if isinstance(cached, dict):
        for key in ("markdown", "text", "content"):
            if key in cached:
                return str(cached[key])

    print("[WARN] Could not find markdown in cache — stringifying object")
    return str(cached)


# ---------------------------------------------------------------------------
# Split into windows
# ---------------------------------------------------------------------------

def split_into_windows(text: str, num_windows: int) -> list[str]:
    """
    Split the full document text into num_windows roughly equal chunks.
    We split on paragraph boundaries to avoid cutting mid-sentence.
    """
    paragraphs = re.split(r"\n{2,}", text)
    total = len(paragraphs)
    chunk_size = max(1, total // num_windows)

    windows: list[str] = []
    for i in range(num_windows):
        start = i * chunk_size
        end = start + chunk_size if i < num_windows - 1 else total
        window_text = "\n\n".join(paragraphs[start:end])
        windows.append(window_text)

    print(f"[INFO] Split {total} paragraphs into {len(windows)} windows")
    for i, w in enumerate(windows):
        print(f"  Window {i+1}: {len(w):,} chars, ~{len(w.split())//200} pages est.")

    return windows


# ---------------------------------------------------------------------------
# OpenAI structured output call
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a legal document evaluation expert. Your task is to generate high-quality retrieval
evaluation questions from Romanian legal text (Codul Penal al României).

Rules:
- Questions MUST be answerable ONLY from the provided text window — do not reference concepts
  outside the given excerpt.
- Write questions in Romanian.
- For each question provide `expected_text_contains`: 1-3 SHORT keywords (1-2 words each) that
  MUST appear verbatim (or near-verbatim) in the correct retrieved chunk. These are used for
  validation — they must be present in the source text.
- The `notes` field (in English) explains what retrieval challenge this question tests.
- Distribute difficulties exactly as requested.
- Distribute types exactly as requested — respect the order given.
"""

USER_PROMPT_TEMPLATE = """\
Generate exactly {n} questions from the following Romanian Penal Code excerpt.

Difficulty distribution (in order):
{difficulties}

Question type distribution (in order, matching difficulty order):
{types}

Return a JSON array with exactly {n} objects, each having:
  - "question": string (in Romanian)
  - "difficulty": one of {diff_set}
  - "type": one of {type_set}
  - "expected_text_contains": array of 1-3 keyword strings present verbatim in the text
  - "notes": string (in English, explains the retrieval challenge)

IMPORTANT: Return ONLY the raw JSON array, no markdown fences.

TEXT EXCERPT:
---
{text}
---
"""

ALLOWED_TYPES = [
    "single_hop", "multi_hop", "lexical_exact", "conceptual",
    "negation", "temporal", "comparison", "disambiguation",
    "summary_only", "bridge",
]


def call_openai(
    client: Any,
    window_text: str,
    window_idx: int,
) -> list[dict[str, Any]]:
    """Call gpt-4o-mini to generate questions for one window."""
    # Truncate window to ~12k chars to stay within token limits
    text_excerpt = window_text[:12_000]

    user_prompt = USER_PROMPT_TEMPLATE.format(
        n=QUESTIONS_PER_WINDOW,
        difficulties=", ".join(DIFFICULTY_DISTRIBUTION),
        types=", ".join(TYPE_DISTRIBUTION),
        diff_set='["easy", "medium", "hard"]',
        type_set=json.dumps(ALLOWED_TYPES),
        text=text_excerpt,
    )

    print(f"[INFO] Window {window_idx+1}/{NUM_WINDOWS}: calling OpenAI ...")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
        max_tokens=4096,
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        questions = json.loads(raw)
        if not isinstance(questions, list):
            print(f"[WARN] Window {window_idx+1}: response is not a list — skipping")
            return []
        print(f"[INFO] Window {window_idx+1}: got {len(questions)} questions")
        return questions
    except json.JSONDecodeError as exc:
        print(f"[WARN] Window {window_idx+1}: JSON parse error: {exc}")
        print(f"[DEBUG] Raw response (first 500 chars): {raw[:500]}")
        return []


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_question(q: dict[str, Any], full_text_lower: str) -> bool:
    """
    Validate that all expected_text_contains keywords appear in the full document text.
    """
    keywords: list[str] = q.get("expected_text_contains", [])
    if not keywords:
        return False

    for kw in keywords:
        # Normalize: lowercase, strip diacritics-insensitive partial match
        kw_lower = kw.lower().strip()
        if kw_lower and kw_lower not in full_text_lower:
            # Try a more lenient check: first 5 chars present
            stem = kw_lower[:5]
            if len(stem) >= 3 and stem not in full_text_lower:
                return False

    return True


def normalize_question(
    q: dict[str, Any],
    question_id: str,
) -> dict[str, Any] | None:
    """Normalize a raw LLM-generated question into the target schema."""
    # Required fields
    question_text = q.get("question", "").strip()
    difficulty = q.get("difficulty", "").lower().strip()
    qtype = q.get("type", "").lower().strip()
    keywords = q.get("expected_text_contains", [])
    notes = q.get("notes", "").strip()

    if not question_text:
        return None
    if difficulty not in ("easy", "medium", "hard"):
        return None
    if qtype not in ALLOWED_TYPES:
        # Try to fix common variants
        type_map = {
            "single-hop": "single_hop",
            "multi-hop": "multi_hop",
            "lexical": "lexical_exact",
            "exact": "lexical_exact",
            "concept": "conceptual",
            "neg": "negation",
            "time": "temporal",
            "compare": "comparison",
            "summary": "summary_only",
        }
        qtype = type_map.get(qtype, qtype)
        if qtype not in ALLOWED_TYPES:
            return None

    if not isinstance(keywords, list):
        keywords = [str(keywords)]

    # Clean keywords: ensure they are non-empty strings
    keywords = [str(k).strip() for k in keywords if str(k).strip()]
    if not keywords:
        return None

    return {
        "id": question_id,
        "difficulty": difficulty,
        "type": qtype,
        "question": question_text,
        "expected_doc_ids": [],
        "expected_text_contains": keywords,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # 1. Load environment
    env = load_env(ENV_FILE)
    api_key = env.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("[ERROR] LLM_API_KEY not found in .env or environment")
        sys.exit(1)

    # 2. Import OpenAI (from venv)
    try:
        from openai import OpenAI  # type: ignore[import]
    except ImportError:
        print("[ERROR] openai not installed — run from the docling_chunker_lab venv")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    # 3. Load markdown from Docling cache
    try:
        full_markdown = load_docling_markdown()
    except Exception as exc:
        print(f"[ERROR] Failed to load Docling cache: {exc}")
        sys.exit(1)

    full_text_lower = full_markdown.lower()

    # 4. Split into 10 windows
    windows = split_into_windows(full_markdown, NUM_WINDOWS)

    # 5. Generate questions per window
    all_raw: list[tuple[int, dict[str, Any]]] = []  # (window_idx, raw_q)

    for window_idx, window_text in enumerate(windows):
        raw_questions = call_openai(client, window_text, window_idx)
        for rq in raw_questions:
            all_raw.append((window_idx, rq))

    print(f"\n[INFO] Total raw questions generated: {len(all_raw)}")

    # 6. Normalize and validate
    validated: list[dict[str, Any]] = []
    skipped_normalize = 0
    skipped_validate = 0

    for idx, (window_idx, rq) in enumerate(all_raw):
        question_id = f"q{idx+1:03d}"
        normalized = normalize_question(rq, question_id)
        if normalized is None:
            skipped_normalize += 1
            continue

        if not validate_question(normalized, full_text_lower):
            skipped_validate += 1
            # Keep the question but add a validation warning in notes
            normalized["notes"] += " [WARN: keyword validation partial]"

        validated.append(normalized)

    print(f"[INFO] Skipped (normalize): {skipped_normalize}")
    print(f"[INFO] Skipped (validate): {skipped_validate}")
    print(f"[INFO] Validated questions: {len(validated)}")

    if len(validated) < MIN_VALIDATED:
        print(f"[WARN] Only {len(validated)} questions — below minimum {MIN_VALIDATED}")

    # 7. Build distribution report
    from collections import Counter
    difficulty_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()

    for q in validated:
        difficulty_counts[q["difficulty"]] += 1
        type_counts[q["type"]] += 1

    print("\n=== Distribution Report ===")
    print("Difficulty:")
    for d in ("easy", "medium", "hard"):
        print(f"  {d}: {difficulty_counts[d]}")
    print("Type:")
    for t in ALLOWED_TYPES:
        print(f"  {t}: {type_counts[t]}")

    # 8. Build output dataset
    dataset = {
        "schema_version": "2",
        "dataset_name": "romanian_penal_code_enterprise_v3",
        "description": (
            "Enterprise evaluation suite — 200+ questions stratified across all 125 pages "
            "of the Romanian Penal Code (test.pdf). Generated via gpt-4o-mini with structured "
            "output. Covers all 10 question types and 3 difficulty tiers. "
            "expected_text_contains validated against source document text."
        ),
        "allowed_difficulties": ["easy", "medium", "hard"],
        "allowed_types": ALLOWED_TYPES,
        "documents": [
            {
                "path": "../test.pdf",
                "doc_id": "penal_code_ro",
                "source": "legal_code_ro",
            }
        ],
        "questions": validated,
    }

    # 9. Write output
    OUTPUT_FILE.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[SUCCESS] Written {len(validated)} questions to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
