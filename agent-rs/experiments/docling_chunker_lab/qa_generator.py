"""Synthetic Q&A generation with verbatim-span validation.

Design (per plan v2):

1. Split each document's markdown export into 4 equal token windows.
2. For each window, ask the QA model to produce 5 Q&A pairs (so 20 per doc,
   100 total across 5 docs).
3. Restrict question types to `factoid | lookup | exact_match` — single-span
   answers are scorable.
4. Post-hoc validate every span: the `answer_span` must be a *verbatim*
   substring of the canonical parsed text after normalization.
5. If validation fails, regenerate the offending item up to 2 times.
6. Drop pairs still failing after 2 retries.  Abort the run if any doc ends
   up with fewer than 15 validated pairs (sanity floor).

Normalization (Codex round 2):
    - Unicode NFKC
    - lowercase
    - strip diacritics
    - collapse whitespace (including non-breaking spaces and soft hyphens)
    - strip markdown table pipes and leading/trailing whitespace

Retrieval uses the same normalization so the canonical match is symmetric.
"""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

LOG = logging.getLogger("docling_lab.qa_generator")

N_WINDOWS_PER_DOC = 4
QA_PER_WINDOW = 5
MIN_VALIDATED_PER_DOC = 15
MAX_REGEN_ATTEMPTS = 2

SYSTEM_PROMPT = """\
You generate high-quality synthetic Q&A pairs for retrieval evaluation.

Rules:
- Produce exactly the requested number of pairs.
- Each `answer_span` MUST be a VERBATIM substring of the provided window.
  Do NOT paraphrase. Do NOT add or remove words. An automated checker will
  reject paraphrased answers.
- Prefer concrete, short spans (under 25 words). Aim for the minimal
  contiguous text that answers the question.
- Use question_type values only from: factoid, lookup, exact_match.
  * factoid  -- "what/when/who/how many" questions with a single verbatim answer
  * lookup   -- questions where the user needs to find a specific value or name
  * exact_match -- questions that probe for a specific phrase or clause
- Mix difficulties: 2 easy, 2 medium, 1 hard per window when possible.
- Match the language of the source window (Romanian questions for Romanian
  windows, English for English windows). If the window is mixed, use the
  dominant language.
"""

USER_PROMPT_TEMPLATE = """\
Source document: {doc_id}
Window: {window_index} of {n_windows}

Generate {count} synthetic Q&A pairs following the system rules.

--- BEGIN WINDOW ---
{window_text}
--- END WINDOW ---
"""


# ----------------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------------


@dataclass
class QAPair:
    qa_id: str
    doc_id: str
    window_index: int
    question: str
    answer_span: str
    normalized_span: str
    question_type: str
    difficulty: str


# ----------------------------------------------------------------------------
# Normalization
# ----------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Canonical string form used for verbatim span matching."""
    # 1. NFKC unicode normalization
    t = unicodedata.normalize("NFKC", text)
    # 2. Strip diacritics
    t = "".join(ch for ch in unicodedata.normalize("NFD", t) if unicodedata.category(ch) != "Mn")
    # 3. Lowercase
    t = t.lower()
    # 4. Normalize soft hyphen and non-breaking space explicitly
    t = t.replace("\u00ad", "").replace("\u00a0", " ")
    # 5. Strip markdown table pipes (retrieved chunks may have |, source may not)
    t = t.replace("|", " ")
    # 6. Collapse whitespace
    t = _WS_RE.sub(" ", t).strip()
    return t


def span_in_canonical(span: str, canonical: str) -> bool:
    """True iff the normalized span is a substring of the normalized canonical."""
    norm_span = normalize(span)
    if not norm_span:
        return False
    norm_canon = normalize(canonical)
    return norm_span in norm_canon


# ----------------------------------------------------------------------------
# Window splitting
# ----------------------------------------------------------------------------


def split_into_windows(markdown: str, n_windows: int) -> list[str]:
    """Split a markdown document into `n_windows` approximately equal slices.

    Uses cl100k_base token counts so windows are balanced for the embedding
    model.  We cut on paragraph boundaries where possible so we don't split
    mid-sentence in Q&A context.
    """
    import tiktoken

    enc = tiktoken.encoding_for_model("text-embedding-3-large")
    paragraphs = re.split(r"\n\s*\n", markdown)
    paragraphs = [p for p in paragraphs if p.strip()]
    if not paragraphs:
        return [markdown] if markdown.strip() else []

    para_token_counts = [len(enc.encode(p, disallowed_special="all")) for p in paragraphs]
    total_tokens = sum(para_token_counts)
    target = max(1, total_tokens // n_windows)

    windows: list[list[str]] = [[]]
    running = 0
    for para, pcount in zip(paragraphs, para_token_counts):
        windows[-1].append(para)
        running += pcount
        if running >= target and len(windows) < n_windows:
            windows.append([])
            running = 0
    return ["\n\n".join(w).strip() for w in windows if w]


# ----------------------------------------------------------------------------
# Generation loop
# ----------------------------------------------------------------------------


def generate_for_doc(
    *,
    doc_id: str,
    canonical_markdown: str,
    qa_model: str,
    openai_client: Any,
) -> list[QAPair]:
    """Generate `N_WINDOWS_PER_DOC * QA_PER_WINDOW` validated pairs for a doc."""
    windows = split_into_windows(canonical_markdown, N_WINDOWS_PER_DOC)
    LOG.info("doc=%s split into %d windows", doc_id, len(windows))

    validated: list[QAPair] = []
    for window_index, window_text in enumerate(windows):
        pairs = _generate_window(
            doc_id=doc_id,
            window_index=window_index,
            window_text=window_text,
            canonical_markdown=canonical_markdown,
            count=QA_PER_WINDOW,
            qa_model=qa_model,
            openai_client=openai_client,
        )
        validated.extend(pairs)

    if len(validated) < MIN_VALIDATED_PER_DOC:
        raise SystemExit(
            f"doc {doc_id}: only {len(validated)} validated pairs "
            f"(minimum {MIN_VALIDATED_PER_DOC}). Aborting run."
        )
    LOG.info("doc=%s produced %d validated pairs", doc_id, len(validated))
    return validated


def _generate_window(
    *,
    doc_id: str,
    window_index: int,
    window_text: str,
    canonical_markdown: str,
    count: int,
    qa_model: str,
    openai_client: Any,
) -> list[QAPair]:
    remaining = count
    attempt = 0
    validated: list[QAPair] = []

    while remaining > 0 and attempt <= MAX_REGEN_ATTEMPTS:
        attempt += 1
        LOG.debug(
            "qa gen doc=%s window=%d attempt=%d need=%d",
            doc_id,
            window_index,
            attempt,
            remaining,
        )
        raw_pairs = _call_qa_model(
            openai_client=openai_client,
            qa_model=qa_model,
            doc_id=doc_id,
            window_index=window_index,
            n_windows=N_WINDOWS_PER_DOC,
            window_text=window_text,
            count=remaining,
        )
        if not raw_pairs:
            LOG.warning(
                "qa gen returned zero pairs on attempt %d for %s/%d",
                attempt,
                doc_id,
                window_index,
            )
            continue

        for raw in raw_pairs:
            question = str(raw.get("question", "")).strip()
            span = str(raw.get("answer_span", "")).strip()
            qtype = str(raw.get("question_type", "factoid")).strip()
            diff = str(raw.get("difficulty", "medium")).strip()
            if not question or not span:
                continue
            if qtype not in {"factoid", "lookup", "exact_match"}:
                continue
            if not span_in_canonical(span, canonical_markdown):
                LOG.debug(
                    "dropped non-verbatim span in doc=%s window=%d: %r",
                    doc_id,
                    window_index,
                    span[:80],
                )
                continue

            qa_id = f"{doc_id}_w{window_index}_q{len(validated):02d}"
            validated.append(
                QAPair(
                    qa_id=qa_id,
                    doc_id=doc_id,
                    window_index=window_index,
                    question=question,
                    answer_span=span,
                    normalized_span=normalize(span),
                    question_type=qtype,
                    difficulty=diff,
                )
            )

        remaining = count - len(validated)

    return validated[:count]


def _call_qa_model(
    *,
    openai_client: Any,
    qa_model: str,
    doc_id: str,
    window_index: int,
    n_windows: int,
    window_text: str,
    count: int,
) -> list[dict[str, Any]]:
    """Make one Q&A gen call with structured JSON schema output."""
    schema = {
        "name": "qa_pairs",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "pairs": {
                    "type": "array",
                    "minItems": count,
                    "maxItems": count,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "question": {"type": "string"},
                            "answer_span": {"type": "string"},
                            "question_type": {
                                "type": "string",
                                "enum": ["factoid", "lookup", "exact_match"],
                            },
                            "difficulty": {
                                "type": "string",
                                "enum": ["easy", "medium", "hard"],
                            },
                        },
                        "required": [
                            "question",
                            "answer_span",
                            "question_type",
                            "difficulty",
                        ],
                    },
                }
            },
            "required": ["pairs"],
        },
    }

    user_prompt = USER_PROMPT_TEMPLATE.format(
        doc_id=doc_id,
        window_index=window_index,
        n_windows=n_windows,
        window_text=window_text,
        count=count,
    )

    try:
        completion = openai_client.chat.completions.create(
            model=qa_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_schema", "json_schema": schema},
            temperature=0.3,
        )
    except Exception as exc:  # noqa: BLE001 -- surface any API error
        LOG.warning("qa gen API call failed for %s/%d: %s", doc_id, window_index, exc)
        return []

    content = completion.choices[0].message.content or ""
    try:
        payload = json.loads(content)
        pairs = payload.get("pairs", [])
        if isinstance(pairs, list):
            return pairs
    except json.JSONDecodeError as exc:
        LOG.warning("qa gen JSON parse failed for %s/%d: %s", doc_id, window_index, exc)
    return []


# ----------------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------------


def save_qa_pairs(pairs: list[QAPair], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps([asdict(p) for p in pairs], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    LOG.info("wrote %d qa pairs to %s", len(pairs), out_path)
