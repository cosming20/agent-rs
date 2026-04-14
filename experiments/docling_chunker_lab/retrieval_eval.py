"""Retrieval evaluation per chunker configuration.

For each config:

1. Create an isolated Qdrant collection `docling_lab_{run_id8}_{cfg_id}`
   with vectors of size 3072 and cosine distance (text-embedding-3-large).
2. Embed every chunk's `embed_text` with OpenAI in batches of 100.
3. Upsert with a payload that carries `doc_id`, `chunk_id`, `cfg_id`,
   `raw_text`, `embed_text`, and `doc_item_refs`.
4. For each validated QA pair, embed the question, retrieve top-5.  A hit
   requires BOTH `payload.doc_id == qa.doc_id` AND the normalized answer
   span to be a substring of the normalized retrieved `raw_text`.
5. Compute per-question metrics, then aggregate per config.
6. After all configs run, compute paired bootstrap 95% CIs on `MRR@5`
   deltas between every config pair.

The scoring loop is deterministic: no LLM judge, no heuristic weighting.
"""
from __future__ import annotations

import json
import logging
import math
import random
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from chunker_runner import ChunkingResult
from qa_generator import QAPair, normalize

LOG = logging.getLogger("docling_lab.retrieval_eval")

DENSE_DIM = 3072
TOP_K = 5
EMBED_BATCH = 100
# Qdrant REST has a 32 MiB body limit.  Each point is ~12 KB vector +
# up to ~6 KB payload text => ~18 KB.  128 points per batch leaves a
# comfortable margin below the cap even for chunks that carry a full
# 1500-token `raw_text`.
UPSERT_BATCH = 64
EMBED_RETRY_ATTEMPTS = 3
EMBED_RETRY_BACKOFF_S = 2.0


# ----------------------------------------------------------------------------
# Output shapes
# ----------------------------------------------------------------------------


@dataclass
class PerQuestionResult:
    qa_id: str
    doc_id: str
    cfg_id: str
    hit_at_1: int
    hit_at_5: int
    mrr_at_5: float
    first_hit_rank: int  # 1..5, or 0 if miss
    question_type: str
    difficulty: str


@dataclass
class ConfigScore:
    cfg_id: str
    mean_hit_at_1: float
    mean_hit_at_5: float
    mean_mrr_at_5: float
    n_questions: int
    per_question: list[PerQuestionResult] = field(default_factory=list)
    by_difficulty: dict[str, dict[str, float]] = field(default_factory=dict)
    by_question_type: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class PairedBootstrapResult:
    cfg_a: str
    cfg_b: str
    mean_delta_mrr: float
    ci_low: float
    ci_high: float
    n_resamples: int
    ci_crosses_zero: bool


# ----------------------------------------------------------------------------
# Qdrant helpers
# ----------------------------------------------------------------------------


def _qdrant_client(host: str, port: int, api_key: str) -> Any:
    from qdrant_client import QdrantClient

    # Passing api_key to QdrantClient flips https=True by default.  Our
    # local Qdrant (port 1051) is HTTP-only, so force https=False and
    # disable gRPC -- this client talks REST.
    kwargs: dict[str, Any] = {
        "host": host,
        "port": port,
        "https": False,
        "prefer_grpc": False,
    }
    if api_key:
        kwargs["api_key"] = api_key
    return QdrantClient(**kwargs)


def _create_collection(client: Any, name: str) -> None:
    from qdrant_client.http.models import Distance, VectorParams

    client.recreate_collection(
        collection_name=name,
        vectors_config=VectorParams(size=DENSE_DIM, distance=Distance.COSINE),
    )
    LOG.info("created collection %s (dim=%d cosine)", name, DENSE_DIM)


def _delete_collection(client: Any, name: str) -> None:
    try:
        client.delete_collection(collection_name=name)
        LOG.debug("deleted collection %s", name)
    except Exception as exc:  # noqa: BLE001 -- best effort
        LOG.warning("failed to delete collection %s: %s", name, exc)


# ----------------------------------------------------------------------------
# OpenAI embedding
# ----------------------------------------------------------------------------


def _embed_batch(
    openai_client: Any,
    embed_model: str,
    texts: list[str],
) -> list[list[float]]:
    last_exc: Exception | None = None
    for attempt in range(1, EMBED_RETRY_ATTEMPTS + 1):
        try:
            response = openai_client.embeddings.create(
                model=embed_model,
                input=texts,
                dimensions=DENSE_DIM,
            )
            return [d.embedding for d in response.data]
        except Exception as exc:  # noqa: BLE001 -- retry on any transient error
            last_exc = exc
            delay = EMBED_RETRY_BACKOFF_S * attempt
            LOG.warning(
                "embed batch failed on attempt %d (%s); sleeping %.1fs",
                attempt,
                exc,
                delay,
            )
            time.sleep(delay)
    raise RuntimeError(f"embed batch failed after {EMBED_RETRY_ATTEMPTS} attempts: {last_exc}")


def _embed_all_chunks(
    openai_client: Any,
    embed_model: str,
    chunks: list[Any],
) -> list[list[float]]:
    out: list[list[float]] = []
    for start in range(0, len(chunks), EMBED_BATCH):
        batch = chunks[start : start + EMBED_BATCH]
        texts = [c.embed_text for c in batch]
        out.extend(_embed_batch(openai_client, embed_model, texts))
    return out


# ----------------------------------------------------------------------------
# Upsert + query
# ----------------------------------------------------------------------------


def _upsert_chunks(
    client: Any,
    collection: str,
    chunks: list[Any],
    vectors: list[list[float]],
) -> None:
    """Upsert points in UPSERT_BATCH-sized chunks to stay under Qdrant's
    32 MiB REST body limit.  A single batch of 587 points exceeds this
    for large-token chunkers like C7 (1500 token budget) on D1."""
    from qdrant_client.http.models import PointStruct

    points = [
        PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_OID, c.chunk_id)),
            vector=vec,
            payload={
                "doc_id": c.doc_id,
                "chunk_id": c.chunk_id,
                "cfg_id": c.cfg_id,
                "raw_text": c.raw_text,
                "headings": c.headings,
                "doc_item_refs": c.doc_item_refs,
            },
        )
        for c, vec in zip(chunks, vectors)
    ]

    for start in range(0, len(points), UPSERT_BATCH):
        batch = points[start : start + UPSERT_BATCH]
        client.upsert(collection_name=collection, points=batch, wait=True)
        LOG.debug(
            "upserted batch %d/%d points to %s",
            start + len(batch),
            len(points),
            collection,
        )


def _query_top_k(
    client: Any,
    collection: str,
    query_vec: list[float],
) -> list[dict[str, Any]]:
    """Run a top-K dense query.

    qdrant-client 1.17 removed the legacy `.search()` method.  Use
    `.query_points()` which returns a `QueryResponse` exposing `.points`.
    """
    response = client.query_points(
        collection_name=collection,
        query=query_vec,
        limit=TOP_K,
        with_payload=True,
    )
    points = response.points
    return [
        {
            "rank": i + 1,
            "score": p.score,
            "payload": p.payload or {},
        }
        for i, p in enumerate(points)
    ]


# ----------------------------------------------------------------------------
# Scoring per config
# ----------------------------------------------------------------------------


def score_config(
    *,
    qdrant_host: str,
    qdrant_http_port: int,
    qdrant_api_key: str,
    openai_client: Any,
    embed_model: str,
    run_id8: str,
    cfg_chunking_results: list[ChunkingResult],
    qa_pairs: list[QAPair],
) -> ConfigScore:
    """Embed, upsert, and score retrieval for one config across all docs."""
    if not cfg_chunking_results:
        raise ValueError("score_config got empty chunking result list")
    cfg_id = cfg_chunking_results[0].cfg_id
    collection = f"docling_lab_{run_id8}_{cfg_id.lower()}"

    client = _qdrant_client(qdrant_host, qdrant_http_port, qdrant_api_key)
    _create_collection(client, collection)

    # Embed + upsert every chunk from every doc for this config.
    for result in cfg_chunking_results:
        if not result.chunks:
            LOG.warning("config %s doc %s: zero chunks", cfg_id, result.doc_id)
            continue
        LOG.info(
            "embed + upsert: cfg=%s doc=%s chunks=%d",
            cfg_id,
            result.doc_id,
            len(result.chunks),
        )
        vectors = _embed_all_chunks(openai_client, embed_model, result.chunks)
        _upsert_chunks(client, collection, result.chunks, vectors)

    # Pre-embed all questions in one batch.
    question_texts = [qa.question for qa in qa_pairs]
    q_vectors = _embed_all_chunks_from_texts(openai_client, embed_model, question_texts)

    # Run retrieval and score.
    per_q: list[PerQuestionResult] = []
    for qa, q_vec in zip(qa_pairs, q_vectors):
        hits = _query_top_k(client, collection, q_vec)
        first_rank = 0
        for h in hits:
            payload = h["payload"]
            if payload.get("doc_id") != qa.doc_id:
                continue
            retrieved = str(payload.get("raw_text", ""))
            if qa.normalized_span in normalize(retrieved):
                first_rank = h["rank"]
                break
        per_q.append(
            PerQuestionResult(
                qa_id=qa.qa_id,
                doc_id=qa.doc_id,
                cfg_id=cfg_id,
                hit_at_1=1 if first_rank == 1 else 0,
                hit_at_5=1 if first_rank > 0 else 0,
                mrr_at_5=(1.0 / first_rank) if first_rank > 0 else 0.0,
                first_hit_rank=first_rank,
                question_type=qa.question_type,
                difficulty=qa.difficulty,
            )
        )

    _delete_collection(client, collection)

    score = _aggregate(per_q, cfg_id=cfg_id)
    LOG.info(
        "config %s: Hit@1=%.3f Hit@5=%.3f MRR@5=%.3f (n=%d)",
        cfg_id,
        score.mean_hit_at_1,
        score.mean_hit_at_5,
        score.mean_mrr_at_5,
        score.n_questions,
    )
    return score


def _embed_all_chunks_from_texts(
    openai_client: Any, embed_model: str, texts: list[str]
) -> list[list[float]]:
    out: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH):
        batch = texts[start : start + EMBED_BATCH]
        out.extend(_embed_batch(openai_client, embed_model, batch))
    return out


def _aggregate(per_q: list[PerQuestionResult], *, cfg_id: str) -> ConfigScore:
    if not per_q:
        return ConfigScore(
            cfg_id=cfg_id,
            mean_hit_at_1=0.0,
            mean_hit_at_5=0.0,
            mean_mrr_at_5=0.0,
            n_questions=0,
        )
    n = len(per_q)
    mean_h1 = sum(r.hit_at_1 for r in per_q) / n
    mean_h5 = sum(r.hit_at_5 for r in per_q) / n
    mean_mrr = sum(r.mrr_at_5 for r in per_q) / n

    # By difficulty
    by_diff: dict[str, dict[str, float]] = {}
    for level in {r.difficulty for r in per_q}:
        subset = [r for r in per_q if r.difficulty == level]
        by_diff[level] = _subset_stats(subset)

    # By question type
    by_type: dict[str, dict[str, float]] = {}
    for qtype in {r.question_type for r in per_q}:
        subset = [r for r in per_q if r.question_type == qtype]
        by_type[qtype] = _subset_stats(subset)

    return ConfigScore(
        cfg_id=cfg_id,
        mean_hit_at_1=round(mean_h1, 4),
        mean_hit_at_5=round(mean_h5, 4),
        mean_mrr_at_5=round(mean_mrr, 4),
        n_questions=n,
        per_question=per_q,
        by_difficulty=by_diff,
        by_question_type=by_type,
    )


def _subset_stats(subset: list[PerQuestionResult]) -> dict[str, float]:
    if not subset:
        return {"n": 0, "hit@1": 0.0, "hit@5": 0.0, "mrr@5": 0.0}
    n = len(subset)
    return {
        "n": n,
        "hit@1": round(sum(r.hit_at_1 for r in subset) / n, 4),
        "hit@5": round(sum(r.hit_at_5 for r in subset) / n, 4),
        "mrr@5": round(sum(r.mrr_at_5 for r in subset) / n, 4),
    }


# ----------------------------------------------------------------------------
# Paired bootstrap CI
# ----------------------------------------------------------------------------


def paired_bootstrap(
    scores: list[ConfigScore],
    *,
    n_resamples: int = 5000,
    seed: int = 42,
) -> list[PairedBootstrapResult]:
    """Compute paired bootstrap 95% CI on MRR@5 deltas between all pairs.

    Each pair (A, B) is scored by resampling QA_ids with replacement and
    computing `mean(MRR_A) - mean(MRR_B)` on each resample.  The 2.5th and
    97.5th percentiles form the 95% CI.
    """
    rng = random.Random(seed)

    # Index per-question results by (cfg_id, qa_id) so pairing is deterministic.
    per_qa: dict[str, dict[str, float]] = {}
    for score in scores:
        for r in score.per_question:
            per_qa.setdefault(r.qa_id, {})[score.cfg_id] = r.mrr_at_5
    qa_ids = sorted(per_qa.keys())
    if not qa_ids:
        return []

    out: list[PairedBootstrapResult] = []
    cfgs = sorted({s.cfg_id for s in scores})
    for i, a in enumerate(cfgs):
        for b in cfgs[i + 1 :]:
            deltas = []
            for _ in range(n_resamples):
                sample = [rng.choice(qa_ids) for _ in qa_ids]
                mean_a = sum(per_qa[q].get(a, 0.0) for q in sample) / len(sample)
                mean_b = sum(per_qa[q].get(b, 0.0) for q in sample) / len(sample)
                deltas.append(mean_a - mean_b)
            deltas.sort()
            lo = deltas[int(math.floor(0.025 * n_resamples))]
            hi = deltas[int(math.floor(0.975 * n_resamples))]
            mean_delta = sum(deltas) / len(deltas)
            out.append(
                PairedBootstrapResult(
                    cfg_a=a,
                    cfg_b=b,
                    mean_delta_mrr=round(mean_delta, 4),
                    ci_low=round(lo, 4),
                    ci_high=round(hi, 4),
                    n_resamples=n_resamples,
                    ci_crosses_zero=lo <= 0 <= hi,
                )
            )
    return out


# ----------------------------------------------------------------------------
# Persistence helpers
# ----------------------------------------------------------------------------


def save_config_scores(scores: list[ConfigScore], out_path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = []
    for s in scores:
        d = asdict(s)
        d["per_question"] = [asdict(r) for r in s.per_question]
        serializable.append(d)
    out_path.write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LOG.info("wrote %d config scores to %s", len(scores), out_path)


def save_bootstrap(results: list[PairedBootstrapResult], out_path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["cfg_a,cfg_b,mean_delta_mrr,ci_low,ci_high,ci_crosses_zero,n_resamples\n"]
    for r in results:
        lines.append(
            f"{r.cfg_a},{r.cfg_b},{r.mean_delta_mrr},{r.ci_low},{r.ci_high},"
            f"{str(r.ci_crosses_zero).lower()},{r.n_resamples}\n"
        )
    out_path.write_text("".join(lines), encoding="utf-8")
    LOG.info("wrote bootstrap CIs to %s", out_path)
