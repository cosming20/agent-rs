# Docling Chunker Feedback Loop

Standalone feedback-loop experiment to pick the best universal Docling
HybridChunker configuration for our Rust agent-rs retrieval stack.

**This experiment is fully isolated from the main `agent-rs` pipeline.**
Nothing under `crates/` or `src/` is touched. Qdrant collections use a
per-run UUID prefix so they cannot collide with `eval_harness*`.

## What it does

1. Parses 5 diverse PDFs through Docling once with production-quality options
   (`TableFormerMode.ACCURATE`, `do_table_structure=True`, `do_ocr=False`).
2. Runs 10 chunker configurations against each cached `DoclingDocument`:
   - 7 variants of `HybridChunker` with `OpenAITokenizer(cl100k_base)` at
     different token budgets (256 / 512 / 768 / 1024 / 1500) and different
     switches (merge_peers, markdown table serialization,
     repeat_table_header, omit_header_on_overflow).
   - `HierarchicalChunker` baseline.
   - Legacy NLTK TextTiling baseline reproducing the old Python behaviour.
3. Generates 20 synthetic Q&A pairs per document (100 total) with
   `gpt-5.4-mini-2026-03-17`, stratified across 4 equal-token windows per
   doc to avoid prefix bias. Each `answer_span` is post-hoc validated as a
   verbatim match in the canonical parsed text (normalized). Failures are
   regenerated up to 2x.
4. Embeds every chunk with `text-embedding-3-large` (3072 dims) into a
   scoped Qdrant collection (`docling_lab_{run_id}_{cfg}`), runs all
   questions against each config, and computes Hit@1 / Hit@5 / MRR@5.
5. Picks the winner purely by empirical retrieval score (MRR@5 primary,
   Hit@5 secondary) with **paired bootstrap 95% CI**. Intrinsic metrics
   (token uniformity, boundary quality, metadata coverage) act only as
   **hard gates** that disqualify configs from the winner pool.

## Run it

```bash
cd agent-rs/experiments/docling_chunker_lab
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# Ensure OPENAI_API_KEY is set.
export OPENAI_API_KEY=$OPENAI_API_KEY
# Optional override: export OPENAI_QA_MODEL=gpt-5-mini

# Ensure Qdrant is running on localhost:1051
# (already up via agentic-ra-qdrant docker container).

python run_sweep.py
```

Results land in `results/run_<uuid>/`:
- `per_doc/<doc>/<cfg>.jsonl` — one line per chunk with raw/embed/contextualized text + metadata
- `qa_pairs.json` — 100 validated synthetic Q&A
- `retrieval_scores.csv` — per-config × per-doc metrics
- `bootstrap_ci.csv` — pairwise MRR@5 CIs
- `hard_gate_report.json` — which configs were disqualified and why
- `summary.md` — ranking table
- `winner.md` — recommended config + integration notes for the Rust sidecar

## What lives where

| File                  | Purpose                                                    |
| --------------------- | ---------------------------------------------------------- |
| `configs.py`          | Chunker factories + startup validators (NLTK, model ID)    |
| `parse_cache.py`      | `DoclingDocument` pickle cache with version metadata       |
| `chunker_runner.py`   | Apply chunker to cached doc → chunks with 3 text views     |
| `intrinsic_metrics.py`| Per-chunk + per-config diagnostic stats + hard gates       |
| `qa_generator.py`     | Stratified Q&A gen + post-hoc verbatim validation loop     |
| `retrieval_eval.py`   | Embed + Qdrant upsert + score + paired bootstrap CI        |
| `run_sweep.py`        | End-to-end orchestrator                                    |

## Scope

- PDF text-native only. Scan-OCR and native-HTML chunking are not evaluated
  here.
- 5 PDF corpus: legal/general (D1), academic code+prose (D2), academic tech
  prose (D3), e-commerce HTML→PDF with tables (D4), heavy invoice table (D5).
- Winner config will later be wrapped as a Python sidecar called from
  agent-rs via IPC; that wrapping is out of scope for this experiment.
