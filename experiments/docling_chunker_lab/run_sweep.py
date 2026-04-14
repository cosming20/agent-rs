"""End-to-end driver for the Docling chunker feedback loop.

Workflow:

1. Validate environment (OPENAI_API_KEY, Qdrant reachable, NLTK resources,
   QA model id, tiktoken smoke test on all parsed docs).
2. Parse all fixtures once (parse_cache).
3. Generate stratified Q&A per doc (qa_generator).  Freeze to disk.
4. For each config in CONFIGS, chunk every doc, compute intrinsic metrics,
   embed + upsert into a per-run Qdrant collection, score retrieval.
5. Compute paired bootstrap CI on MRR@5 deltas.
6. Write hard_gate_report.json, retrieval_scores, bootstrap_ci, summary.md,
   winner.md.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from pathlib import Path

from configs import (
    CONFIGS,
    ensure_nltk_resources,
    resolve_qa_model,
    tokenizer_smoke_test,
)
from chunker_runner import dump_chunks, run_chunker
from intrinsic_metrics import IntrinsicMetrics, compute as compute_intrinsics
from parse_cache import ParseCache, ParsedDocument
from qa_generator import QAPair, generate_for_doc, save_qa_pairs
from retrieval_eval import (
    ConfigScore,
    paired_bootstrap,
    save_bootstrap,
    save_config_scores,
    score_config,
)

LOG = logging.getLogger("docling_lab.run_sweep")

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent
FIXTURES_DIR = REPO_ROOT / "fixtures"
RESULTS_DIR = REPO_ROOT / "results"

DOC_MANIFEST = [
    ("D1", "D1_test.pdf"),
    ("D2", "D2_apd2_websocket.pdf"),
    ("D3", "D3_lucrarea4_so.pdf"),
    ("D4", "D4_automobilus_cart.pdf"),
    ("D5", "D5_invoice.pdf"),
]


def _log_setup() -> None:
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level_name, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _env_or_die(var: str) -> str:
    val = os.environ.get(var)
    if not val:
        raise SystemExit(f"environment variable {var} is required")
    return val


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main() -> int:
    _log_setup()
    LOG.info("docling chunker feedback loop starting")

    api_key = _env_or_die("OPENAI_API_KEY")
    qdrant_host = os.environ.get("QDRANT_HOST", "localhost")
    qdrant_http_port = int(os.environ.get("QDRANT_HTTP_PORT", "1051"))
    qdrant_api_key = os.environ.get("QDRANT_API_KEY", "changeme-qdrant-key")
    embed_model = os.environ.get(
        "OPENAI_EMBED_MODEL", "text-embedding-3-large"
    )

    import openai

    openai_client = openai.OpenAI(api_key=api_key)
    qa_model = resolve_qa_model()
    ensure_nltk_resources()

    run_id = str(uuid.uuid4())
    run_id8 = run_id[:8]
    run_dir = RESULTS_DIR / f"run_{run_id8}"
    run_dir.mkdir(parents=True, exist_ok=True)
    LOG.info("run_id=%s (short=%s) -> %s", run_id, run_id8, run_dir)

    # Parse cache is shared across runs so reruns don't pay the Docling
    # parse cost again for the same fixtures + pipeline options.
    cache = ParseCache(cache_dir=RESULTS_DIR / "_parse_cache_shared")

    # ------------------------------------------------------------------
    # Step 1: parse all docs
    # ------------------------------------------------------------------
    parsed: dict[str, ParsedDocument] = {}
    markdowns: dict[str, str] = {}
    source_table_counts: dict[str, int] = {}

    for doc_id, filename in DOC_MANIFEST:
        pdf_path = FIXTURES_DIR / filename
        if not pdf_path.exists():
            LOG.error("missing fixture: %s", pdf_path)
            return 1
        p = cache.parse_or_load(doc_id, pdf_path)
        parsed[doc_id] = p
        markdowns[doc_id] = p.markdown
        source_table_counts[doc_id] = _count_tables(p.doc)

    tokenizer_smoke_test(markdowns)

    # ------------------------------------------------------------------
    # Step 2: Q&A generation
    #
    # Q&A pairs are shared across reruns (keyed by corpus, not run_id) so
    # we do not pay the OpenAI QA-gen cost repeatedly for the same docs.
    # The per-run dir gets a symlink so `results/run_X/qa_pairs.json`
    # still resolves.
    # ------------------------------------------------------------------
    shared_qa_path = RESULTS_DIR / "_qa_pairs_shared.json"
    qa_path = run_dir / "qa_pairs.json"
    all_qa: list[QAPair] = []
    if shared_qa_path.exists():
        LOG.info("reusing shared qa_pairs from %s", shared_qa_path)
        raw = json.loads(shared_qa_path.read_text(encoding="utf-8"))
        all_qa = [QAPair(**item) for item in raw]
    else:
        for doc_id, p in parsed.items():
            pairs = generate_for_doc(
                doc_id=doc_id,
                canonical_markdown=p.markdown,
                qa_model=qa_model,
                openai_client=openai_client,
            )
            all_qa.extend(pairs)
        save_qa_pairs(all_qa, shared_qa_path)

    # Mirror into the run dir for traceability (hardlink to avoid a copy).
    if not qa_path.exists():
        try:
            os.link(shared_qa_path, qa_path)
        except OSError:
            save_qa_pairs(all_qa, qa_path)

    LOG.info("total validated qa pairs: %d", len(all_qa))

    # ------------------------------------------------------------------
    # Step 3: chunking + intrinsic metrics + retrieval eval per config
    # ------------------------------------------------------------------
    all_intrinsics: dict[str, dict[str, IntrinsicMetrics]] = {}
    all_scores: list[ConfigScore] = []

    for cfg in CONFIGS:
        LOG.info("=" * 60)
        LOG.info("config %s — %s", cfg.cfg_id, cfg.rationale)

        cfg_chunking = []
        for doc_id, p in parsed.items():
            try:
                result = run_chunker(p, cfg_id=cfg.cfg_id)
            except Exception as exc:  # noqa: BLE001 -- log and skip config
                LOG.error(
                    "config %s failed on doc %s: %s", cfg.cfg_id, doc_id, exc
                )
                cfg_chunking = []
                break
            out_path = run_dir / "per_doc" / doc_id / f"{cfg.cfg_id}.jsonl"
            dump_chunks(result, out_path)

            intrinsics = compute_intrinsics(
                result, source_table_count=source_table_counts[doc_id]
            )
            all_intrinsics.setdefault(cfg.cfg_id, {})[doc_id] = intrinsics
            cfg_chunking.append(result)

        if not cfg_chunking:
            LOG.warning("config %s produced no chunking results; skipping retrieval eval", cfg.cfg_id)
            continue

        # Retrieval scoring for this config.
        score = score_config(
            qdrant_host=qdrant_host,
            qdrant_http_port=qdrant_http_port,
            qdrant_api_key=qdrant_api_key,
            openai_client=openai_client,
            embed_model=embed_model,
            run_id8=run_id8,
            cfg_chunking_results=cfg_chunking,
            qa_pairs=all_qa,
        )
        all_scores.append(score)

    # ------------------------------------------------------------------
    # Step 4: hard gate report
    # ------------------------------------------------------------------
    hard_gate_report = _build_hard_gate_report(all_intrinsics)
    (run_dir / "hard_gate_report.json").write_text(
        json.dumps(hard_gate_report, indent=2), encoding="utf-8"
    )

    disqualified = {
        cfg_id for cfg_id, entry in hard_gate_report.items() if entry["failed"]
    }
    LOG.info("disqualified configs: %s", sorted(disqualified))

    # ------------------------------------------------------------------
    # Step 5: persistence + bootstrap + winner selection
    # ------------------------------------------------------------------
    save_config_scores(all_scores, run_dir / "retrieval_scores.json")

    bootstrap_results = paired_bootstrap(all_scores)
    save_bootstrap(bootstrap_results, run_dir / "bootstrap_ci.csv")

    eligible_scores = [s for s in all_scores if s.cfg_id not in disqualified]
    winner = _pick_winner(eligible_scores, bootstrap_results)

    summary_md = _render_summary(
        all_scores=all_scores,
        disqualified=disqualified,
        hard_gate_report=hard_gate_report,
        winner=winner,
    )
    (run_dir / "summary.md").write_text(summary_md, encoding="utf-8")

    winner_md = _render_winner(
        winner=winner,
        all_scores=all_scores,
        bootstrap_results=bootstrap_results,
        disqualified=disqualified,
    )
    (run_dir / "winner.md").write_text(winner_md, encoding="utf-8")

    LOG.info("sweep complete. winner=%s", winner["cfg_id"] if winner else "NONE")
    return 0


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _count_tables(doclingdoc) -> int:
    tables = getattr(doclingdoc, "tables", None)
    if tables is None:
        return 0
    try:
        return len(tables)
    except TypeError:
        return 0


def _build_hard_gate_report(
    intrinsics: dict[str, dict[str, IntrinsicMetrics]],
) -> dict[str, dict]:
    report: dict[str, dict] = {}
    for cfg_id, by_doc in intrinsics.items():
        failures = []
        for doc_id, m in by_doc.items():
            for g in m.gates:
                if not g.passed:
                    failures.append(
                        {
                            "doc_id": doc_id,
                            "gate": g.name,
                            "value": g.value,
                            "detail": g.detail,
                        }
                    )
        report[cfg_id] = {
            "failed": bool(failures),
            "failures": failures,
        }
    return report


def _pick_winner(
    eligible: list[ConfigScore],
    bootstrap: list,
) -> dict | None:
    if not eligible:
        return None
    ordered = sorted(eligible, key=lambda s: s.mean_mrr_at_5, reverse=True)
    top = ordered[0]
    runner_up = ordered[1] if len(ordered) > 1 else None

    # Check whether top MRR is statistically distinguishable from runner-up.
    statistically_clear = True
    if runner_up is not None:
        match = next(
            (
                b
                for b in bootstrap
                if {b.cfg_a, b.cfg_b} == {top.cfg_id, runner_up.cfg_id}
            ),
            None,
        )
        if match is not None and match.ci_crosses_zero:
            statistically_clear = False

    return {
        "cfg_id": top.cfg_id,
        "mean_mrr": top.mean_mrr_at_5,
        "mean_hit_at_1": top.mean_hit_at_1,
        "mean_hit_at_5": top.mean_hit_at_5,
        "n_questions": top.n_questions,
        "statistically_clear": statistically_clear,
        "runner_up": runner_up.cfg_id if runner_up else None,
    }


def _render_summary(
    *,
    all_scores: list[ConfigScore],
    disqualified: set[str],
    hard_gate_report: dict[str, dict],
    winner: dict | None,
) -> str:
    lines = ["# Docling Chunker Sweep Summary", ""]
    if winner:
        clear = "yes" if winner["statistically_clear"] else "no (overlaps runner-up)"
        lines.append(
            f"**Winner:** `{winner['cfg_id']}` — MRR@5={winner['mean_mrr']:.3f}, "
            f"Hit@5={winner['mean_hit_at_5']:.3f} (clear? {clear})"
        )
    else:
        lines.append("**No winner** — every config was disqualified by a hard gate.")
    lines.append("")

    lines.append("## Retrieval scores")
    lines.append("")
    lines.append("| cfg | MRR@5 | Hit@1 | Hit@5 | n | disqualified |")
    lines.append("|-----|-------|-------|-------|---|--------------|")
    ordered = sorted(all_scores, key=lambda s: s.mean_mrr_at_5, reverse=True)
    for s in ordered:
        dq = "yes" if s.cfg_id in disqualified else "no"
        lines.append(
            f"| {s.cfg_id} | {s.mean_mrr_at_5:.3f} | {s.mean_hit_at_1:.3f} | "
            f"{s.mean_hit_at_5:.3f} | {s.n_questions} | {dq} |"
        )
    lines.append("")

    lines.append("## Hard gate failures")
    for cfg_id, entry in hard_gate_report.items():
        if not entry["failed"]:
            continue
        lines.append(f"### {cfg_id}")
        for f in entry["failures"]:
            lines.append(f"- doc {f['doc_id']}: {f['gate']} — {f['detail']}")
        lines.append("")
    return "\n".join(lines) + "\n"


def _render_winner(
    *,
    winner: dict | None,
    all_scores: list[ConfigScore],
    bootstrap_results: list,
    disqualified: set[str],
) -> str:
    if not winner:
        return "# No winner\n\nAll configurations were disqualified.\n"

    lines = [
        "# Winner",
        "",
        f"**cfg_id:** `{winner['cfg_id']}`",
        f"**Mean MRR@5:** {winner['mean_mrr']:.4f}",
        f"**Mean Hit@1:** {winner['mean_hit_at_1']:.4f}",
        f"**Mean Hit@5:** {winner['mean_hit_at_5']:.4f}",
        f"**n questions:** {winner['n_questions']}",
        f"**Statistically clear over runner-up:** {winner['statistically_clear']}",
        f"**Runner-up:** `{winner['runner_up']}`" if winner["runner_up"] else "",
        "",
    ]

    lines.append("## Deltas vs production baseline (C7) and legacy baseline (C9)")
    by_id = {s.cfg_id: s for s in all_scores}
    winner_score = by_id.get(winner["cfg_id"])
    if winner_score is not None:
        for baseline_id in ("C7", "C9"):
            base = by_id.get(baseline_id)
            if base is None:
                continue
            lines.append(
                f"- vs {baseline_id}: "
                f"ΔMRR@5={winner_score.mean_mrr_at_5 - base.mean_mrr_at_5:+.4f}, "
                f"ΔHit@5={winner_score.mean_hit_at_5 - base.mean_hit_at_5:+.4f}"
            )
    lines.append("")
    lines.append("## Pairwise bootstrap CI vs other configs (MRR@5)")
    lines.append("")
    lines.append("| vs | ΔMRR@5 | 95% CI | crosses zero |")
    lines.append("|----|--------|--------|--------------|")
    for b in bootstrap_results:
        if winner["cfg_id"] not in (b.cfg_a, b.cfg_b):
            continue
        if b.cfg_a == winner["cfg_id"]:
            other, delta, lo, hi = b.cfg_b, b.mean_delta_mrr, b.ci_low, b.ci_high
        else:
            # Flip sign so delta is always "winner - other".
            other, delta, lo, hi = b.cfg_a, -b.mean_delta_mrr, -b.ci_high, -b.ci_low
        lines.append(
            f"| {other} | {delta:+.4f} | [{lo:+.4f}, {hi:+.4f}] | "
            f"{'yes' if b.ci_crosses_zero else 'no'} |"
        )
    lines.append("")

    if disqualified:
        lines.append("## Disqualified configs (hard gate failures)")
        lines.append("")
        for cfg_id in sorted(disqualified):
            lines.append(f"- `{cfg_id}`")
        lines.append("")

    lines.append("## Integration recommendation")
    lines.append("")
    lines.append(
        "Use the winner config as the default for the Python Docling sidecar "
        "that will be called from agent-rs via IPC. Export the exact "
        "constructor arguments as YAML so the Rust side can pass them "
        "verbatim when spawning the sidecar."
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
