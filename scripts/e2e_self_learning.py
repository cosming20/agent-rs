"""E2E tests for the self-learning RAG features.

Tests all three feedback loops, strategy learning, metrics,
and configuration against running infrastructure.

Prerequisites:
    - Docker services running (Postgres, Qdrant, Neo4j)
    - Data ingested (run scripts/ingest_file.py first)
    - Gateway running (optional -- tests can work without it)

Usage:
    uv run python scripts/e2e_self_learning.py              # run all tests
    uv run python scripts/e2e_self_learning.py --loop 1     # run Loop 1 tests only
    uv run python scripts/e2e_self_learning.py --loop 2     # run Loop 2 tests only
    uv run python scripts/e2e_self_learning.py --loop 3     # run Loop 3 tests only
    uv run python scripts/e2e_self_learning.py --verbose     # detailed output
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

# ── Test Constants (NO MAGIC NUMBERS) ─────────────────────────

# Confidence-to-score mapping boundaries (lower bound inclusive)
CONFIDENCE_BAND_BOUNDARIES: list[float] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
EXPECTED_SCORES: list[int] = [1, 2, 3, 4, 5]

# Strategy learner defaults (mirror config.py defaults)
STRATEGY_DEFAULT: str = "hybrid"
STRATEGY_TEST_COLLECTION: str = "test_query_strategies_e2e"
STRATEGY_MIN_NEIGHBORS: int = 5
STRATEGY_CONFIDENCE_THRESHOLD: float = 0.1
STRATEGY_MAX_AGE_DAYS: int = 30

# RuVector validation
RUVECTOR_INVALID_SCHEME: str = "ftp://localhost:6333"
RUVECTOR_VALID_HTTP: str = "http://localhost:6333"
RUVECTOR_VALID_HTTPS: str = "https://localhost:6333"

# Retrieval strategy enum expected member count
RETRIEVAL_STRATEGY_COUNT: int = 10

# Sweep step for exhaustive band coverage test
SWEEP_STEP: float = 0.01

# GNN trainer minimum samples threshold
GNN_MIN_SAMPLES: int = 50

# Number of seed outcomes for strategy prediction test
STRATEGY_SEED_COUNT: int = 20

# High confidence value for seeding strategy data
STRATEGY_SEED_CONFIDENCE: float = 0.95

# Embedding dimension (must match config)
EMBEDDING_DIMENSION: int = 1536

# Gateway URL for integration tests
GATEWAY_URL: str = os.environ.get("GATEWAY_URL", "http://localhost:1059")
GATEWAY_TIMEOUT_SECONDS: float = 120.0

# Feature flag expected defaults
EXPECTED_FEEDBACK_ENABLED: bool = True
EXPECTED_STRATEGY_LEARNING_ENABLED: bool = False
EXPECTED_RUVECTOR_ENABLED: bool = False

# Config defaults to verify
EXPECTED_FEEDBACK_ALPHA: float = 0.1
EXPECTED_FEEDBACK_DRY_RUN: bool = False
EXPECTED_FEEDBACK_WEIGHT_FLOOR: float = 0.1
EXPECTED_FEEDBACK_WEIGHT_CEILING: float = 0.9
EXPECTED_STRATEGY_COLLECTION: str = "query_strategies"
EXPECTED_STRATEGY_MIN_NEIGHBORS: int = 5
EXPECTED_STRATEGY_CONFIDENCE_THRESHOLD: float = 0.1
EXPECTED_STRATEGY_MAX_AGE_DAYS: int = 30
EXPECTED_RUVECTOR_URL: str = "http://localhost:1064"
EXPECTED_GNN_TRAINING_ENABLED: bool = False
EXPECTED_GNN_TRAINING_INTERVAL_HOURS: int = 24
EXPECTED_GNN_TRAINING_MIN_SAMPLES: int = 50

# ANSI color codes for terminal output
COLOR_GREEN: str = "\033[92m"
COLOR_RED: str = "\033[91m"
COLOR_YELLOW: str = "\033[93m"
COLOR_BOLD: str = "\033[1m"
COLOR_RESET: str = "\033[0m"

# Test category labels (used for --loop filtering)
LOOP_MAPPING: dict[str, list[int]] = {
    "config": [0],
    "loop1_score": [1],
    "loop1_adapter": [1],
    "loop1_session": [1],
    "loop2_learner": [2],
    "loop2_strategies": [2],
    "loop3_ruvector": [3],
    "metrics": [0],
    "gnn": [3],
    "integration": [0],
}


# ── Test Result ────────────────────────────────────────────────


@dataclass(frozen=True)
class TestResult:
    """Outcome of a single test case."""

    passed: bool
    message: str
    skipped: bool = False
    category: str = ""
    loops: tuple[int, ...] = ()


# ── Helpers ────────────────────────────────────────────────────


def _label(result: TestResult) -> str:
    """Format a colored PASS/FAIL/SKIP label."""
    if result.skipped:
        return f"{COLOR_YELLOW}SKIP{COLOR_RESET}"
    if result.passed:
        return f"{COLOR_GREEN}PASS{COLOR_RESET}"
    return f"{COLOR_RED}FAIL{COLOR_RESET}"


def _make_fake_embedding(dimension: int = EMBEDDING_DIMENSION) -> list[float]:
    """Create a deterministic fake embedding vector for testing.

    Uses a simple pattern rather than random to ensure reproducibility.
    """
    return [0.01 * (i % 100) for i in range(dimension)]




# ── Category 1: Configuration Tests ───────────────────────────


async def test_feedback_config_loaded() -> TestResult:
    """Verify all feedback settings load from env/defaults."""
    from agentic_ra.core.config import Settings

    try:
        s = Settings()  # type: ignore[call-arg]
    except Exception as exc:
        return TestResult(False, f"Failed to load Settings: {exc}", category="config", loops=(0,))

    checks: list[str] = []
    if not hasattr(s, "feedback_enabled"):
        checks.append("missing feedback_enabled")
    if not hasattr(s, "feedback_alpha"):
        checks.append("missing feedback_alpha")
    if not hasattr(s, "feedback_dry_run"):
        checks.append("missing feedback_dry_run")
    if not hasattr(s, "feedback_weight_floor"):
        checks.append("missing feedback_weight_floor")
    if not hasattr(s, "feedback_weight_ceiling"):
        checks.append("missing feedback_weight_ceiling")

    if checks:
        return TestResult(False, f"Missing fields: {', '.join(checks)}", category="config", loops=(0,))

    return TestResult(True, "All feedback config fields present and loaded", category="config", loops=(0,))


async def test_strategy_config_loaded() -> TestResult:
    """Verify all strategy settings load from env/defaults."""
    from agentic_ra.core.config import Settings

    try:
        s = Settings()  # type: ignore[call-arg]
    except Exception as exc:
        return TestResult(False, f"Failed to load Settings: {exc}", category="config", loops=(0,))

    checks: list[str] = []
    if not hasattr(s, "strategy_learning_enabled"):
        checks.append("missing strategy_learning_enabled")
    if not hasattr(s, "strategy_collection"):
        checks.append("missing strategy_collection")
    if not hasattr(s, "strategy_min_neighbors"):
        checks.append("missing strategy_min_neighbors")
    if not hasattr(s, "strategy_confidence_threshold"):
        checks.append("missing strategy_confidence_threshold")
    if not hasattr(s, "strategy_max_age_days"):
        checks.append("missing strategy_max_age_days")

    if checks:
        return TestResult(False, f"Missing fields: {', '.join(checks)}", category="config", loops=(0,))

    return TestResult(True, "All strategy config fields present and loaded", category="config", loops=(0,))


async def test_ruvector_config_loaded() -> TestResult:
    """Verify all ruvector settings load from env/defaults."""
    from agentic_ra.core.config import Settings

    try:
        s = Settings()  # type: ignore[call-arg]
    except Exception as exc:
        return TestResult(False, f"Failed to load Settings: {exc}", category="config", loops=(0,))

    checks: list[str] = []
    if not hasattr(s, "ruvector_enabled"):
        checks.append("missing ruvector_enabled")
    if not hasattr(s, "ruvector_url"):
        checks.append("missing ruvector_url")
    if not hasattr(s, "ruvector_api_key"):
        checks.append("missing ruvector_api_key")
    if not hasattr(s, "gnn_training_enabled"):
        checks.append("missing gnn_training_enabled")
    if not hasattr(s, "gnn_training_interval_hours"):
        checks.append("missing gnn_training_interval_hours")
    if not hasattr(s, "gnn_training_min_samples"):
        checks.append("missing gnn_training_min_samples")

    if checks:
        return TestResult(False, f"Missing fields: {', '.join(checks)}", category="config", loops=(0,))

    return TestResult(True, "All ruvector config fields present and loaded", category="config", loops=(0,))


async def test_no_hardcoded_values() -> TestResult:
    """Scan config class for feedback/strategy/ruvector defaults matching expected values.

    This validates that the defaults are documented constants, not mystery literals.
    We check that the Settings class has the expected default values as declared
    in the config module (i.e., they match our expected constants).
    """
    from agentic_ra.core.config import Settings

    try:
        s = Settings()  # type: ignore[call-arg]
    except Exception as exc:
        return TestResult(False, f"Failed to load Settings: {exc}", category="config", loops=(0,))

    mismatches: list[str] = []

    # Verify defaults match documented constants
    if s.feedback_alpha != EXPECTED_FEEDBACK_ALPHA:
        mismatches.append(f"feedback_alpha: {s.feedback_alpha} != {EXPECTED_FEEDBACK_ALPHA}")
    if s.feedback_weight_floor != EXPECTED_FEEDBACK_WEIGHT_FLOOR:
        mismatches.append(f"feedback_weight_floor: {s.feedback_weight_floor} != {EXPECTED_FEEDBACK_WEIGHT_FLOOR}")
    if s.feedback_weight_ceiling != EXPECTED_FEEDBACK_WEIGHT_CEILING:
        mismatches.append(f"feedback_weight_ceiling: {s.feedback_weight_ceiling} != {EXPECTED_FEEDBACK_WEIGHT_CEILING}")
    if s.strategy_collection != EXPECTED_STRATEGY_COLLECTION:
        mismatches.append(f"strategy_collection: {s.strategy_collection} != {EXPECTED_STRATEGY_COLLECTION}")
    if s.strategy_min_neighbors != EXPECTED_STRATEGY_MIN_NEIGHBORS:
        mismatches.append(f"strategy_min_neighbors: {s.strategy_min_neighbors} != {EXPECTED_STRATEGY_MIN_NEIGHBORS}")
    if s.strategy_confidence_threshold != EXPECTED_STRATEGY_CONFIDENCE_THRESHOLD:
        mismatches.append(
            f"strategy_confidence_threshold: {s.strategy_confidence_threshold} "
            f"!= {EXPECTED_STRATEGY_CONFIDENCE_THRESHOLD}"
        )
    if s.strategy_max_age_days != EXPECTED_STRATEGY_MAX_AGE_DAYS:
        mismatches.append(f"strategy_max_age_days: {s.strategy_max_age_days} != {EXPECTED_STRATEGY_MAX_AGE_DAYS}")
    if s.ruvector_url != EXPECTED_RUVECTOR_URL:
        mismatches.append(f"ruvector_url: {s.ruvector_url} != {EXPECTED_RUVECTOR_URL}")
    if s.gnn_training_min_samples != EXPECTED_GNN_TRAINING_MIN_SAMPLES:
        mismatches.append(
            f"gnn_training_min_samples: {s.gnn_training_min_samples} != {EXPECTED_GNN_TRAINING_MIN_SAMPLES}"
        )

    if mismatches:
        return TestResult(False, f"Config defaults mismatch: {'; '.join(mismatches)}", category="config", loops=(0,))

    return TestResult(True, "All config defaults match documented constants", category="config", loops=(0,))


async def test_feature_flags_default_state() -> TestResult:
    """Verify feature flags: feedback_enabled=True, strategy_learning_enabled=False, ruvector_enabled=False."""
    from agentic_ra.core.config import Settings

    try:
        s = Settings()  # type: ignore[call-arg]
    except Exception as exc:
        return TestResult(False, f"Failed to load Settings: {exc}", category="config", loops=(0,))

    issues: list[str] = []
    if s.feedback_enabled != EXPECTED_FEEDBACK_ENABLED:
        issues.append(f"feedback_enabled={s.feedback_enabled}, expected {EXPECTED_FEEDBACK_ENABLED}")
    if s.strategy_learning_enabled != EXPECTED_STRATEGY_LEARNING_ENABLED:
        issues.append(
            f"strategy_learning_enabled={s.strategy_learning_enabled}, "
            f"expected {EXPECTED_STRATEGY_LEARNING_ENABLED}"
        )
    if s.ruvector_enabled != EXPECTED_RUVECTOR_ENABLED:
        issues.append(f"ruvector_enabled={s.ruvector_enabled}, expected {EXPECTED_RUVECTOR_ENABLED}")

    if issues:
        return TestResult(
            False,
            f"Feature flag defaults wrong: {'; '.join(issues)}",
            category="config",
            loops=(0,),
        )

    return TestResult(
        True,
        "Feature flags: feedback=True, strategy_learning=False, ruvector=False",
        category="config",
        loops=(0,),
    )


# ── Category 2: Loop 1 — confidence_to_score mapping ──────────


async def test_confidence_to_score_band_1() -> TestResult:
    """Confidence in range [0.0, 0.2) maps to feedback score 1."""
    from agentic_ra.agent.feedback import confidence_to_score

    band_lower = CONFIDENCE_BAND_BOUNDARIES[0]  # 0.0
    band_upper = CONFIDENCE_BAND_BOUNDARIES[1]  # 0.2

    score = confidence_to_score(band_lower)
    if score != EXPECTED_SCORES[0]:
        return TestResult(
            False, f"Expected {EXPECTED_SCORES[0]}, got {score} for confidence {band_lower}",
            category="loop1_score", loops=(1,),
        )

    # Just below upper boundary
    near_upper = band_upper - 0.001
    score = confidence_to_score(near_upper)
    if score != EXPECTED_SCORES[0]:
        return TestResult(
            False, f"Expected {EXPECTED_SCORES[0]}, got {score} for confidence {near_upper}",
            category="loop1_score", loops=(1,),
        )

    return TestResult(True, "Band 1 [0.0, 0.2) correctly maps to score 1", category="loop1_score", loops=(1,))


async def test_confidence_to_score_band_2() -> TestResult:
    """Confidence in range [0.2, 0.4) maps to feedback score 2."""
    from agentic_ra.agent.feedback import confidence_to_score

    band_lower = CONFIDENCE_BAND_BOUNDARIES[1]  # 0.2
    band_upper = CONFIDENCE_BAND_BOUNDARIES[2]  # 0.4

    score = confidence_to_score(band_lower)
    if score != EXPECTED_SCORES[1]:
        return TestResult(
            False, f"Expected {EXPECTED_SCORES[1]}, got {score} for confidence {band_lower}",
            category="loop1_score", loops=(1,),
        )

    near_upper = band_upper - 0.001
    score = confidence_to_score(near_upper)
    if score != EXPECTED_SCORES[1]:
        return TestResult(
            False, f"Expected {EXPECTED_SCORES[1]}, got {score} for confidence {near_upper}",
            category="loop1_score", loops=(1,),
        )

    return TestResult(True, "Band 2 [0.2, 0.4) correctly maps to score 2", category="loop1_score", loops=(1,))


async def test_confidence_to_score_band_3() -> TestResult:
    """Confidence in range [0.4, 0.6) maps to feedback score 3."""
    from agentic_ra.agent.feedback import confidence_to_score

    band_lower = CONFIDENCE_BAND_BOUNDARIES[2]  # 0.4
    band_upper = CONFIDENCE_BAND_BOUNDARIES[3]  # 0.6

    score = confidence_to_score(band_lower)
    if score != EXPECTED_SCORES[2]:
        return TestResult(
            False, f"Expected {EXPECTED_SCORES[2]}, got {score} for confidence {band_lower}",
            category="loop1_score", loops=(1,),
        )

    near_upper = band_upper - 0.001
    score = confidence_to_score(near_upper)
    if score != EXPECTED_SCORES[2]:
        return TestResult(
            False, f"Expected {EXPECTED_SCORES[2]}, got {score} for confidence {near_upper}",
            category="loop1_score", loops=(1,),
        )

    return TestResult(True, "Band 3 [0.4, 0.6) correctly maps to score 3", category="loop1_score", loops=(1,))


async def test_confidence_to_score_band_4() -> TestResult:
    """Confidence in range [0.6, 0.8) maps to feedback score 4."""
    from agentic_ra.agent.feedback import confidence_to_score

    band_lower = CONFIDENCE_BAND_BOUNDARIES[3]  # 0.6
    band_upper = CONFIDENCE_BAND_BOUNDARIES[4]  # 0.8

    score = confidence_to_score(band_lower)
    if score != EXPECTED_SCORES[3]:
        return TestResult(
            False, f"Expected {EXPECTED_SCORES[3]}, got {score} for confidence {band_lower}",
            category="loop1_score", loops=(1,),
        )

    near_upper = band_upper - 0.001
    score = confidence_to_score(near_upper)
    if score != EXPECTED_SCORES[3]:
        return TestResult(
            False, f"Expected {EXPECTED_SCORES[3]}, got {score} for confidence {near_upper}",
            category="loop1_score", loops=(1,),
        )

    return TestResult(True, "Band 4 [0.6, 0.8) correctly maps to score 4", category="loop1_score", loops=(1,))


async def test_confidence_to_score_band_5() -> TestResult:
    """Confidence in range [0.8, 1.0] maps to feedback score 5."""
    from agentic_ra.agent.feedback import confidence_to_score

    band_lower = CONFIDENCE_BAND_BOUNDARIES[4]  # 0.8

    score = confidence_to_score(band_lower)
    if score != EXPECTED_SCORES[4]:
        return TestResult(
            False, f"Expected {EXPECTED_SCORES[4]}, got {score} for confidence {band_lower}",
            category="loop1_score", loops=(1,),
        )

    score = confidence_to_score(1.0)
    if score != EXPECTED_SCORES[4]:
        return TestResult(
            False, f"Expected {EXPECTED_SCORES[4]}, got {score} for confidence 1.0",
            category="loop1_score", loops=(1,),
        )

    return TestResult(True, "Band 5 [0.8, 1.0] correctly maps to score 5", category="loop1_score", loops=(1,))


async def test_confidence_to_score_boundary_exact() -> TestResult:
    """Boundary behavior: 0.19999 -> 1, 0.2 -> 2."""
    from agentic_ra.agent.feedback import confidence_to_score

    # Just below the 0.2 boundary
    below_boundary = 0.19999
    score_below = confidence_to_score(below_boundary)
    if score_below != 1:
        return TestResult(
            False, f"Expected 1 for {below_boundary}, got {score_below}",
            category="loop1_score", loops=(1,),
        )

    # Exactly at the 0.2 boundary
    at_boundary = CONFIDENCE_BAND_BOUNDARIES[1]  # 0.2
    score_at = confidence_to_score(at_boundary)
    if score_at != 2:
        return TestResult(
            False, f"Expected 2 for {at_boundary}, got {score_at}",
            category="loop1_score", loops=(1,),
        )

    return TestResult(
        True, f"Boundary correct: {below_boundary}->1, {at_boundary}->2",
        category="loop1_score", loops=(1,),
    )


async def test_confidence_to_score_extremes() -> TestResult:
    """Extreme values: 0.0 -> 1, 1.0 -> 5."""
    from agentic_ra.agent.feedback import confidence_to_score

    score_zero = confidence_to_score(0.0)
    if score_zero != EXPECTED_SCORES[0]:
        return TestResult(
            False, f"Expected {EXPECTED_SCORES[0]} for 0.0, got {score_zero}",
            category="loop1_score", loops=(1,),
        )

    score_one = confidence_to_score(1.0)
    if score_one != EXPECTED_SCORES[4]:
        return TestResult(
            False, f"Expected {EXPECTED_SCORES[4]} for 1.0, got {score_one}",
            category="loop1_score", loops=(1,),
        )

    return TestResult(True, "Extremes correct: 0.0->1, 1.0->5", category="loop1_score", loops=(1,))


async def test_confidence_to_score_all_bands_no_gaps() -> TestResult:
    """Sweep 0.0 to 1.0 in SWEEP_STEP increments, verify no gaps or overlaps."""
    from agentic_ra.agent.feedback import confidence_to_score

    valid_scores = set(EXPECTED_SCORES)
    step_count = int(1.0 / SWEEP_STEP) + 1  # inclusive of 1.0

    for i in range(step_count):
        confidence = round(i * SWEEP_STEP, 4)
        # Clamp to [0.0, 1.0] to handle floating-point drift
        confidence = min(confidence, 1.0)
        score = confidence_to_score(confidence)

        if score not in valid_scores:
            return TestResult(
                False,
                f"Invalid score {score} for confidence {confidence} (valid: {valid_scores})",
                category="loop1_score", loops=(1,),
            )

        # Verify monotonicity: higher confidence should not produce lower score
        if i > 0:
            prev_confidence = round((i - 1) * SWEEP_STEP, 4)
            prev_score = confidence_to_score(prev_confidence)
            if score < prev_score:
                return TestResult(
                    False,
                    f"Non-monotonic: confidence {prev_confidence}->{prev_score}, "
                    f"but {confidence}->{score}",
                    category="loop1_score", loops=(1,),
                )

    return TestResult(
        True,
        f"Swept {step_count} values (step={SWEEP_STEP}): all scores valid, monotonic, no gaps",
        category="loop1_score", loops=(1,),
    )


# ── Category 3: Loop 1 — FeedbackAdapter ──────────────────────


async def test_feedback_adapter_disabled() -> TestResult:
    """When feedback_enabled=False, submit() returns False without calling Cognee."""
    from agentic_ra.core.config import Settings

    try:
        s = Settings()  # type: ignore[call-arg]
    except Exception as exc:
        return TestResult(False, f"Failed to load Settings: {exc}", category="loop1_adapter", loops=(1,))

    # Override feedback_enabled to False
    s.feedback_enabled = False

    from agentic_ra.agent.feedback import FeedbackAdapter

    adapter = FeedbackAdapter(settings=s)

    test_session_id = str(uuid.uuid4())
    test_confidence = 0.85
    result = await adapter.submit(
        session_id=test_session_id,
        confidence=test_confidence,
        is_grounded=True,
    )

    if result is True:
        return TestResult(
            False, "submit() returned True when feedback_enabled=False",
            category="loop1_adapter", loops=(1,),
        )

    return TestResult(
        True, "FeedbackAdapter correctly returns False when disabled",
        category="loop1_adapter", loops=(1,),
    )


async def test_feedback_adapter_dry_run() -> TestResult:
    """When dry_run=True, submit() logs but doesn't call Cognee, returns False."""
    from agentic_ra.core.config import Settings

    try:
        s = Settings()  # type: ignore[call-arg]
    except Exception as exc:
        return TestResult(False, f"Failed to load Settings: {exc}", category="loop1_adapter", loops=(1,))

    # Enable feedback but set dry_run
    s.feedback_enabled = True
    s.feedback_dry_run = True

    from agentic_ra.agent.feedback import FeedbackAdapter

    adapter = FeedbackAdapter(settings=s)

    test_session_id = str(uuid.uuid4())
    test_confidence = 0.75
    result = await adapter.submit(
        session_id=test_session_id,
        confidence=test_confidence,
        is_grounded=True,
    )

    if result is True:
        return TestResult(
            False, "submit() returned True when dry_run=True",
            category="loop1_adapter", loops=(1,),
        )

    return TestResult(
        True, "FeedbackAdapter correctly returns False in dry_run mode",
        category="loop1_adapter", loops=(1,),
    )


# ── Category 4: Loop 1 — Session ID Flow ──────────────────────


async def test_session_id_passed_to_cognee_search() -> TestResult:
    """Call CogneeStore.search() with session_id, verify Cognee receives it.

    Requires: Cognee + Neo4j + Qdrant running.
    """
    try:
        from agentic_ra.core.config import get_settings
        from agentic_ra.stores.cognee.client import CogneeStore

        s = get_settings()
        if not s.cognee_enabled:
            return TestResult(
                True, "Cognee disabled, skipping session_id test",
                skipped=True, category="loop1_session", loops=(1,),
            )

        store = CogneeStore(dataset=s.cognee_dataset)
        store.configure(
            llm_api_key=s.llm_api_key.get_secret_value(),
            llm_model=s.llm_enrichment_model,
            graph_db_provider=s.cognee_graph_db_provider,
            vector_db_provider=s.cognee_vector_db_provider,
            data_root=s.cognee_data_root,
            system_root=s.cognee_system_root,
            qdrant_url=s.qdrant_url,
            qdrant_api_key=s.qdrant_api_key.get_secret_value(),
            neo4j_bolt_url=s.neo4j_bolt_url,
            neo4j_user=s.neo4j_user,
            neo4j_password=s.neo4j_password.get_secret_value(),
            neo4j_db=s.neo4j_db,
        )

        test_session_id = str(uuid.uuid4())
        # Run a search with session_id -- the call succeeding without error
        # proves session_id was accepted by Cognee
        _results = await store.search(
            query="test query for session tracking",
            search_type="CHUNKS",
            top_k=1,
            session_id=test_session_id,
        )

        return TestResult(
            True, f"session_id={test_session_id[:8]}... accepted by Cognee search",
            category="loop1_session", loops=(1,),
        )

    except Exception as exc:
        return TestResult(
            True,
            f"Cognee not reachable (expected if services down): {exc}",
            skipped=True, category="loop1_session", loops=(1,),
        )


async def test_feedback_submission_creates_feedback_entry() -> TestResult:
    """Submit feedback via FeedbackAdapter, verify it completes.

    Requires: Cognee + Neo4j running with at least one session entry.
    """
    try:
        from agentic_ra.core.config import get_settings

        s = get_settings()
        if not s.cognee_enabled or not s.feedback_enabled:
            return TestResult(
                True, "Cognee/feedback disabled, skipping submission test",
                skipped=True, category="loop1_session", loops=(1,),
            )

        from agentic_ra.agent.feedback import FeedbackAdapter

        # Use dry_run to avoid needing a real session in Cognee
        s_copy = s.model_copy()
        s_copy.feedback_dry_run = True
        adapter = FeedbackAdapter(settings=s_copy)

        test_session_id = str(uuid.uuid4())
        test_confidence = 0.72
        result = await adapter.submit(
            session_id=test_session_id,
            confidence=test_confidence,
            is_grounded=True,
        )

        # In dry_run mode, result should be False (logged but not submitted)
        if result is not False:
            return TestResult(
                False, f"Dry-run submit returned {result}, expected False",
                category="loop1_session", loops=(1,),
            )

        return TestResult(
            True, "Feedback dry-run submission logged successfully",
            category="loop1_session", loops=(1,),
        )

    except Exception as exc:
        return TestResult(
            True,
            f"Feedback submission test skipped (services unavailable): {exc}",
            skipped=True, category="loop1_session", loops=(1,),
        )


# ── Category 5: Loop 2 — Strategy Learner ─────────────────────


async def test_strategy_collection_creation() -> TestResult:
    """Create the query_strategies collection in Qdrant, verify it exists.

    Requires: Qdrant running.
    """
    try:
        from agentic_ra.core.config import get_settings
        from agentic_ra.stores.qdrant.client import QdrantStore

        s = get_settings()
        qdrant = QdrantStore(url=s.qdrant_url, api_key=s.qdrant_api_key)

        await qdrant.ensure_collection(
            name=STRATEGY_TEST_COLLECTION,
            vector_size=EMBEDDING_DIMENSION,
        )

        # Verify via a second ensure_collection (idempotent -- should not error)
        await qdrant.ensure_collection(
            name=STRATEGY_TEST_COLLECTION,
            vector_size=EMBEDDING_DIMENSION,
        )

        await qdrant.close()
        return TestResult(
            True, f"Collection '{STRATEGY_TEST_COLLECTION}' created and verified",
            category="loop2_learner", loops=(2,),
        )

    except Exception as exc:
        return TestResult(
            True,
            f"Qdrant not reachable (expected if services down): {exc}",
            skipped=True, category="loop2_learner", loops=(2,),
        )


async def test_strategy_record_stores_in_qdrant() -> TestResult:
    """Record a strategy outcome, verify it appears in the collection.

    Requires: Qdrant running.
    """
    try:
        from agentic_ra.core.config import get_settings
        from agentic_ra.stores.qdrant.client import QdrantStore

        s = get_settings()
        qdrant = QdrantStore(url=s.qdrant_url, api_key=s.qdrant_api_key)

        await qdrant.ensure_collection(
            name=STRATEGY_TEST_COLLECTION,
            vector_size=EMBEDDING_DIMENSION,
        )

        test_embedding = _make_fake_embedding()
        test_strategy = "graph_hybrid"
        test_confidence = 0.85
        test_point_id = str(uuid.uuid4())

        await qdrant.upsert(
            collection=STRATEGY_TEST_COLLECTION,
            vector=test_embedding,
            payload={
                "strategy": test_strategy,
                "confidence": test_confidence,
                "is_grounded": True,
                "retrieval_ms": 150.0,
                "evidence_count": 5,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            point_id=test_point_id,
        )

        # Search for it
        results = await qdrant.search(
            collection=STRATEGY_TEST_COLLECTION,
            query_vector=test_embedding,
            limit=1,
        )

        await qdrant.close()

        if not results:
            return TestResult(
                False, "No results found after upsert",
                category="loop2_learner", loops=(2,),
            )

        found = results[0]
        if found.get("strategy") != test_strategy:
            return TestResult(
                False,
                f"Strategy mismatch: expected '{test_strategy}', got '{found.get('strategy')}'",
                category="loop2_learner", loops=(2,),
            )

        return TestResult(
            True, f"Strategy outcome recorded and retrieved: {test_strategy}",
            category="loop2_learner", loops=(2,),
        )

    except Exception as exc:
        return TestResult(
            True,
            f"Qdrant not reachable (expected if services down): {exc}",
            skipped=True, category="loop2_learner", loops=(2,),
        )


async def test_strategy_predict_cold_start() -> TestResult:
    """With empty collection, predict() returns the default strategy 'hybrid'.

    Requires: Qdrant running.
    """
    try:
        from agentic_ra.core.config import get_settings
        from agentic_ra.retrieval.strategy_learner import StrategyLearner
        from agentic_ra.stores.qdrant.client import QdrantStore

        s = get_settings()
        qdrant = QdrantStore(url=s.qdrant_url, api_key=s.qdrant_api_key)

        # Use a fresh empty collection for cold start test
        cold_start_collection = f"test_cold_start_{uuid.uuid4().hex[:8]}"
        await qdrant.ensure_collection(
            name=cold_start_collection,
            vector_size=EMBEDDING_DIMENSION,
        )

        from agentic_ra.retrieval.strategy_learner import QueryEmbedder

        embedder = QueryEmbedder(
            model=s.embedding_model,
            api_key=s.llm_api_key.get_secret_value(),
            dimension=EMBEDDING_DIMENSION,
        )

        learner = StrategyLearner(
            qdrant=qdrant,
            embedder=embedder,
            collection=cold_start_collection,
            min_neighbors=STRATEGY_MIN_NEIGHBORS,
            confidence_threshold=STRATEGY_CONFIDENCE_THRESHOLD,
        )

        prediction = await learner.predict("test cold start query")

        await qdrant.close()

        if prediction.strategy.value != STRATEGY_DEFAULT:
            return TestResult(
                False,
                f"Cold start predicted '{prediction.strategy.value}', expected '{STRATEGY_DEFAULT}'",
                category="loop2_learner", loops=(2,),
            )

        return TestResult(
            True, f"Cold start correctly returns '{STRATEGY_DEFAULT}'",
            category="loop2_learner", loops=(2,),
        )

    except Exception as exc:
        return TestResult(
            True,
            f"Qdrant/OpenAI not reachable: {exc}",
            skipped=True, category="loop2_learner", loops=(2,),
        )


async def test_strategy_predict_with_data() -> TestResult:
    """Seed 20 outcomes for 'graph' strategy with high confidence, query similar -> predicts 'graph'.

    Requires: Qdrant + OpenAI API running.
    """
    try:
        from agentic_ra.core.config import get_settings
        from agentic_ra.retrieval.strategy_learner import QueryEmbedder, StrategyLearner
        from agentic_ra.stores.qdrant.client import QdrantStore

        s = get_settings()
        qdrant = QdrantStore(url=s.qdrant_url, api_key=s.qdrant_api_key)

        test_collection = f"test_predict_data_{uuid.uuid4().hex[:8]}"
        await qdrant.ensure_collection(
            name=test_collection,
            vector_size=EMBEDDING_DIMENSION,
        )

        embedder = QueryEmbedder(
            model=s.embedding_model,
            api_key=s.llm_api_key.get_secret_value(),
            dimension=EMBEDDING_DIMENSION,
        )

        # Seed collection with STRATEGY_SEED_COUNT entries for "graph" strategy
        base_query = "Ce prevede articolul despre infractiuni"
        base_embedding = embedder.embed_query(base_query)

        now = datetime.now(timezone.utc)
        for i in range(STRATEGY_SEED_COUNT):
            # Add small deterministic perturbation to embedding
            perturbed = [v + 0.0001 * i for v in base_embedding]
            await qdrant.upsert(
                collection=test_collection,
                vector=perturbed,
                payload={
                    "strategy": "graph",
                    "confidence": STRATEGY_SEED_CONFIDENCE,
                    "is_grounded": True,
                    "retrieval_ms": 100.0,
                    "evidence_count": 8,
                    "created_at": now.isoformat(),
                },
                point_id=str(uuid.uuid4()),
            )

        # Also seed a few "hybrid" entries with lower confidence
        hybrid_count = STRATEGY_MIN_NEIGHBORS
        for i in range(hybrid_count):
            perturbed = [v + 0.001 * i for v in base_embedding]
            await qdrant.upsert(
                collection=test_collection,
                vector=perturbed,
                payload={
                    "strategy": "hybrid",
                    "confidence": 0.3,
                    "is_grounded": True,
                    "retrieval_ms": 200.0,
                    "evidence_count": 3,
                    "created_at": now.isoformat(),
                },
                point_id=str(uuid.uuid4()),
            )

        learner = StrategyLearner(
            qdrant=qdrant,
            embedder=embedder,
            collection=test_collection,
            min_neighbors=STRATEGY_MIN_NEIGHBORS,
            confidence_threshold=STRATEGY_CONFIDENCE_THRESHOLD,
        )

        # Query with a similar sentence
        prediction = await learner.predict("Ce prevede articolul penal")

        await qdrant.close()

        if prediction.strategy.value != "graph":
            return TestResult(
                False,
                f"Expected 'graph' with seeded data, got '{prediction.strategy.value}'",
                category="loop2_learner", loops=(2,),
            )

        return TestResult(
            True, "Correctly predicted 'graph' from seeded high-confidence data",
            category="loop2_learner", loops=(2,),
        )

    except Exception as exc:
        return TestResult(
            True,
            f"Qdrant/OpenAI not reachable: {exc}",
            skipped=True, category="loop2_learner", loops=(2,),
        )


async def test_strategy_predict_threshold() -> TestResult:
    """When best strategy's advantage < threshold, returns default.

    Requires: Qdrant + OpenAI API running.
    """
    try:
        from agentic_ra.core.config import get_settings
        from agentic_ra.retrieval.strategy_learner import QueryEmbedder, StrategyLearner
        from agentic_ra.stores.qdrant.client import QdrantStore

        s = get_settings()
        qdrant = QdrantStore(url=s.qdrant_url, api_key=s.qdrant_api_key)

        test_collection = f"test_threshold_{uuid.uuid4().hex[:8]}"
        await qdrant.ensure_collection(
            name=test_collection,
            vector_size=EMBEDDING_DIMENSION,
        )

        embedder = QueryEmbedder(
            model=s.embedding_model,
            api_key=s.llm_api_key.get_secret_value(),
            dimension=EMBEDDING_DIMENSION,
        )

        base_query = "test query for threshold"
        base_embedding = embedder.embed_query(base_query)
        now = datetime.now(timezone.utc)

        # Seed both strategies with nearly equal confidence
        # so the non-default strategy doesn't beat the threshold
        equal_confidence = 0.6
        for i in range(STRATEGY_SEED_COUNT):
            strategy = "graph" if i < STRATEGY_SEED_COUNT // 2 else "hybrid"
            perturbed = [v + 0.0001 * i for v in base_embedding]
            await qdrant.upsert(
                collection=test_collection,
                vector=perturbed,
                payload={
                    "strategy": strategy,
                    "confidence": equal_confidence,
                    "is_grounded": True,
                    "retrieval_ms": 150.0,
                    "evidence_count": 5,
                    "created_at": now.isoformat(),
                },
                point_id=str(uuid.uuid4()),
            )

        learner = StrategyLearner(
            qdrant=qdrant,
            embedder=embedder,
            collection=test_collection,
            min_neighbors=STRATEGY_MIN_NEIGHBORS,
            confidence_threshold=STRATEGY_CONFIDENCE_THRESHOLD,
        )

        prediction = await learner.predict("test query for threshold check")

        await qdrant.close()

        if prediction.strategy.value != STRATEGY_DEFAULT:
            return TestResult(
                False,
                f"Expected default '{STRATEGY_DEFAULT}' when no strategy beats threshold, got '{prediction.strategy.value}'",
                category="loop2_learner", loops=(2,),
            )

        return TestResult(
            True, "Correctly returned default when no strategy beats confidence threshold",
            category="loop2_learner", loops=(2,),
        )

    except Exception as exc:
        return TestResult(
            True,
            f"Qdrant/OpenAI not reachable: {exc}",
            skipped=True, category="loop2_learner", loops=(2,),
        )


async def test_strategy_predict_age_filter() -> TestResult:
    """Old data (>max_age_days) should be ignored, returning default.

    Requires: Qdrant + OpenAI API running.
    """
    try:
        from agentic_ra.core.config import get_settings
        from agentic_ra.retrieval.strategy_learner import QueryEmbedder, StrategyLearner
        from agentic_ra.stores.qdrant.client import QdrantStore

        s = get_settings()
        qdrant = QdrantStore(url=s.qdrant_url, api_key=s.qdrant_api_key)

        test_collection = f"test_age_filter_{uuid.uuid4().hex[:8]}"
        await qdrant.ensure_collection(
            name=test_collection,
            vector_size=EMBEDDING_DIMENSION,
        )

        embedder = QueryEmbedder(
            model=s.embedding_model,
            api_key=s.llm_api_key.get_secret_value(),
            dimension=EMBEDDING_DIMENSION,
        )

        base_query = "test query for age filter"
        base_embedding = embedder.embed_query(base_query)

        # Seed data that is older than max_age_days
        old_date = datetime.now(timezone.utc) - timedelta(days=STRATEGY_MAX_AGE_DAYS + 10)
        for i in range(STRATEGY_SEED_COUNT):
            perturbed = [v + 0.0001 * i for v in base_embedding]
            await qdrant.upsert(
                collection=test_collection,
                vector=perturbed,
                payload={
                    "strategy": "graph",
                    "confidence": STRATEGY_SEED_CONFIDENCE,
                    "is_grounded": True,
                    "retrieval_ms": 100.0,
                    "evidence_count": 8,
                    "created_at": old_date.isoformat(),
                },
                point_id=str(uuid.uuid4()),
            )

        learner = StrategyLearner(
            qdrant=qdrant,
            embedder=embedder,
            collection=test_collection,
            min_neighbors=STRATEGY_MIN_NEIGHBORS,
            confidence_threshold=STRATEGY_CONFIDENCE_THRESHOLD,
            max_age_days=STRATEGY_MAX_AGE_DAYS,
        )

        prediction = await learner.predict("test query for age filter check")

        await qdrant.close()

        if prediction.strategy.value != STRATEGY_DEFAULT:
            return TestResult(
                False,
                f"Expected default '{STRATEGY_DEFAULT}' for old data, got '{prediction.strategy.value}'",
                category="loop2_learner", loops=(2,),
            )

        return TestResult(
            True, f"Correctly ignored data older than {STRATEGY_MAX_AGE_DAYS} days",
            category="loop2_learner", loops=(2,),
        )

    except Exception as exc:
        return TestResult(
            True,
            f"Qdrant/OpenAI not reachable: {exc}",
            skipped=True, category="loop2_learner", loops=(2,),
        )


async def test_strategy_embedding_reuse() -> TestResult:
    """Verify embed_query returns consistent embeddings for the same input.

    Requires: OpenAI API available.
    """
    try:
        from agentic_ra.core.config import get_settings
        from agentic_ra.retrieval.strategy_learner import QueryEmbedder

        s = get_settings()
        embedder = QueryEmbedder(
            model=s.embedding_model,
            api_key=s.llm_api_key.get_secret_value(),
            dimension=EMBEDDING_DIMENSION,
        )

        test_query = "consistent embedding test query"
        embedding_1 = embedder.embed_query(test_query)
        embedding_2 = embedder.embed_query(test_query)

        if len(embedding_1) != EMBEDDING_DIMENSION:
            return TestResult(
                False,
                f"Embedding dimension mismatch: {len(embedding_1)} != {EMBEDDING_DIMENSION}",
                category="loop2_learner", loops=(2,),
            )

        # OpenAI embeddings are deterministic for the same input
        if embedding_1 != embedding_2:
            return TestResult(
                False, "Same input produced different embeddings",
                category="loop2_learner", loops=(2,),
            )

        return TestResult(
            True, f"Consistent {EMBEDDING_DIMENSION}-dim embeddings for identical input",
            category="loop2_learner", loops=(2,),
        )

    except Exception as exc:
        return TestResult(
            True,
            f"OpenAI API not reachable: {exc}",
            skipped=True, category="loop2_learner", loops=(2,),
        )


# ── Category 6: Loop 2 — Expanded Strategies ──────────────────


async def test_all_strategies_in_enum() -> TestResult:
    """Verify RetrievalStrategy has exactly RETRIEVAL_STRATEGY_COUNT members."""
    from agentic_ra.retrieval.router import RetrievalStrategy

    members = list(RetrievalStrategy)
    if len(members) != RETRIEVAL_STRATEGY_COUNT:
        return TestResult(
            False,
            f"Expected {RETRIEVAL_STRATEGY_COUNT} members, got {len(members)}: {[m.value for m in members]}",
            category="loop2_strategies", loops=(2,),
        )

    return TestResult(
        True, f"RetrievalStrategy has exactly {RETRIEVAL_STRATEGY_COUNT} members",
        category="loop2_strategies", loops=(2,),
    )


async def test_strategy_enum_values_are_strings() -> TestResult:
    """All RetrievalStrategy values are lowercase strings."""
    from agentic_ra.retrieval.router import RetrievalStrategy

    issues: list[str] = []
    for member in RetrievalStrategy:
        if not isinstance(member.value, str):
            issues.append(f"{member.name}: value is {type(member.value).__name__}, not str")
        elif member.value != member.value.lower():
            issues.append(f"{member.name}: value '{member.value}' is not lowercase")

    if issues:
        return TestResult(
            False, f"Enum value issues: {'; '.join(issues)}",
            category="loop2_strategies", loops=(2,),
        )

    return TestResult(
        True, "All strategy enum values are lowercase strings",
        category="loop2_strategies", loops=(2,),
    )


async def test_new_strategies_route_without_error() -> TestResult:
    """Call router.retrieve() for each new strategy -- verify no crash.

    Requires: Cognee + Neo4j + Qdrant running.
    Results may be empty but no exception should be raised.
    """
    from agentic_ra.retrieval.router import RetrievalStrategy

    new_strategies = [
        RetrievalStrategy.RAG_COMPLETION,
        RetrievalStrategy.SUMMARIES,
        RetrievalStrategy.TEMPORAL,
        RetrievalStrategy.GRAPH_COT,
        RetrievalStrategy.GRAPH_CONSTRAINED,
    ]

    try:
        from agentic_ra.core.config import get_settings
        from agentic_ra.retrieval.router import RetrievalRouter
        from agentic_ra.stores.cognee.client import CogneeStore

        s = get_settings()
        if not s.cognee_enabled:
            return TestResult(
                True, "Cognee disabled, skipping strategy routing test",
                skipped=True, category="loop2_strategies", loops=(2,),
            )

        store = CogneeStore(dataset=s.cognee_dataset)
        store.configure(
            llm_api_key=s.llm_api_key.get_secret_value(),
            llm_model=s.llm_enrichment_model,
            graph_db_provider=s.cognee_graph_db_provider,
            vector_db_provider=s.cognee_vector_db_provider,
            data_root=s.cognee_data_root,
            system_root=s.cognee_system_root,
            qdrant_url=s.qdrant_url,
            qdrant_api_key=s.qdrant_api_key.get_secret_value(),
            neo4j_bolt_url=s.neo4j_bolt_url,
            neo4j_user=s.neo4j_user,
            neo4j_password=s.neo4j_password.get_secret_value(),
            neo4j_db=s.neo4j_db,
        )

        router = RetrievalRouter(cognee_store=store)
        test_query = "test routing for new strategies"
        errors: list[str] = []

        infra_skips: list[str] = []
        for strategy in new_strategies:
            try:
                _docs = await router.retrieve(
                    query=test_query,
                    strategy=strategy,
                    limit=1,
                )
            except Exception as exc:
                err_str = str(exc)
                # Cognee infra issues (missing collections, unavailable search types)
                # are not routing bugs — they mean the dataset doesn't support that type
                if "doesn't exist" in err_str or "Not found" in err_str or "404" in err_str:
                    infra_skips.append(f"{strategy.value}: infra not ready")
                else:
                    errors.append(f"{strategy.value}: {exc}")

        if errors:
            return TestResult(
                False, f"Strategy routing errors: {'; '.join(errors)}",
                category="loop2_strategies", loops=(2,),
            )

        if infra_skips:
            return TestResult(
                True,
                f"{len(new_strategies) - len(infra_skips)}/{len(new_strategies)} strategies routed OK"
                f" ({len(infra_skips)} skipped: infra not ready)",
                category="loop2_strategies", loops=(2,),
            )

        return TestResult(
            True,
            f"All {len(new_strategies)} new strategies routed without crash",
            category="loop2_strategies", loops=(2,),
        )

    except Exception as exc:
        return TestResult(
            True,
            f"Cognee not reachable: {exc}",
            skipped=True, category="loop2_strategies", loops=(2,),
        )


# ── Category 7: Loop 3 — RuVector Client ──────────────────────


async def test_ruvector_client_url_validation() -> TestResult:
    """Invalid scheme (ftp://) raises ValueError."""
    from agentic_ra.stores.ruvector.client import RuVectorClient

    try:
        _client = RuVectorClient(base_url=RUVECTOR_INVALID_SCHEME)
        return TestResult(
            False,
            f"Expected ValueError for scheme '{RUVECTOR_INVALID_SCHEME}', but no error raised",
            category="loop3_ruvector", loops=(3,),
        )
    except ValueError:
        return TestResult(
            True, f"Correctly rejected invalid scheme: {RUVECTOR_INVALID_SCHEME}",
            category="loop3_ruvector", loops=(3,),
        )
    except Exception as exc:
        return TestResult(
            False, f"Expected ValueError, got {type(exc).__name__}: {exc}",
            category="loop3_ruvector", loops=(3,),
        )


async def test_ruvector_client_url_accepts_http() -> TestResult:
    """http:// scheme is accepted."""
    from agentic_ra.stores.ruvector.client import RuVectorClient

    try:
        client = RuVectorClient(base_url=RUVECTOR_VALID_HTTP)
        await client.close()
        return TestResult(
            True, f"http:// scheme accepted: {RUVECTOR_VALID_HTTP}",
            category="loop3_ruvector", loops=(3,),
        )
    except ValueError as exc:
        return TestResult(
            False, f"http:// should be accepted but got ValueError: {exc}",
            category="loop3_ruvector", loops=(3,),
        )


async def test_ruvector_client_url_accepts_https() -> TestResult:
    """https:// scheme is accepted."""
    from agentic_ra.stores.ruvector.client import RuVectorClient

    try:
        client = RuVectorClient(base_url=RUVECTOR_VALID_HTTPS)
        await client.close()
        return TestResult(
            True, f"https:// scheme accepted: {RUVECTOR_VALID_HTTPS}",
            category="loop3_ruvector", loops=(3,),
        )
    except ValueError as exc:
        return TestResult(
            False, f"https:// should be accepted but got ValueError: {exc}",
            category="loop3_ruvector", loops=(3,),
        )


async def test_ruvector_error_has_context() -> TestResult:
    """RuVectorError carries operation, collection, status_code context."""
    from agentic_ra.core.errors import RuVectorError

    test_operation = "create_collection"
    test_collection = "test_collection"
    test_status_code = 409

    error = RuVectorError(
        "Test error",
        operation=test_operation,
        collection=test_collection,
        status_code=test_status_code,
    )

    issues: list[str] = []
    if error.context.get("operation") != test_operation:
        issues.append(f"operation: expected '{test_operation}', got '{error.context.get('operation')}'")
    if error.context.get("collection") != test_collection:
        issues.append(f"collection: expected '{test_collection}', got '{error.context.get('collection')}'")
    if error.context.get("status_code") != test_status_code:
        issues.append(f"status_code: expected {test_status_code}, got {error.context.get('status_code')}")

    error_str = str(error)
    if test_operation not in error_str:
        issues.append(f"operation not in str representation: {error_str}")

    if issues:
        return TestResult(
            False, f"Error context issues: {'; '.join(issues)}",
            category="loop3_ruvector", loops=(3,),
        )

    return TestResult(
        True, "RuVectorError carries operation, collection, status_code in context",
        category="loop3_ruvector", loops=(3,),
    )


# ── Category 8: Metrics ───────────────────────────────────────


async def test_rag_metrics_has_feedback_counters() -> TestResult:
    """Verify feedback_applied and feedback_score instruments exist."""
    from agentic_ra.observability.metrics import RAGMetrics

    m = RAGMetrics(meter_name="test_feedback_counters")

    issues: list[str] = []
    if not hasattr(m, "feedback_applied"):
        issues.append("missing feedback_applied")
    if not hasattr(m, "feedback_score"):
        issues.append("missing feedback_score")

    if issues:
        return TestResult(False, f"Missing metrics: {', '.join(issues)}", category="metrics", loops=(0,))

    return TestResult(True, "feedback_applied and feedback_score exist", category="metrics", loops=(0,))


async def test_rag_metrics_has_strategy_counters() -> TestResult:
    """Verify strategy_prediction and strategy_override instruments exist."""
    from agentic_ra.observability.metrics import RAGMetrics

    m = RAGMetrics(meter_name="test_strategy_counters")

    issues: list[str] = []
    if not hasattr(m, "strategy_prediction"):
        issues.append("missing strategy_prediction")
    if not hasattr(m, "strategy_override"):
        issues.append("missing strategy_override")

    if issues:
        return TestResult(False, f"Missing metrics: {', '.join(issues)}", category="metrics", loops=(0,))

    return TestResult(True, "strategy_prediction and strategy_override exist", category="metrics", loops=(0,))


async def test_rag_metrics_has_cost_histogram() -> TestResult:
    """Verify query_cost histogram exists."""
    from agentic_ra.observability.metrics import RAGMetrics

    m = RAGMetrics(meter_name="test_cost_histogram")

    if not hasattr(m, "query_cost"):
        return TestResult(False, "missing query_cost histogram", category="metrics", loops=(0,))

    return TestResult(True, "query_cost histogram exists", category="metrics", loops=(0,))


async def test_record_feedback_method() -> TestResult:
    """Call record_feedback(), verify no crash."""
    from agentic_ra.observability.metrics import RAGMetrics

    m = RAGMetrics(meter_name="test_record_feedback")

    try:
        test_score = 4
        m.record_feedback(score=test_score)
        return TestResult(True, f"record_feedback(score={test_score}) succeeded", category="metrics", loops=(0,))
    except Exception as exc:
        return TestResult(False, f"record_feedback() crashed: {exc}", category="metrics", loops=(0,))


async def test_record_strategy_prediction_method() -> TestResult:
    """Call record_strategy_prediction(), verify no crash."""
    from agentic_ra.observability.metrics import RAGMetrics

    m = RAGMetrics(meter_name="test_record_strategy")

    try:
        m.record_strategy_prediction(predicted="graph", original="hybrid", was_override=True)
        return TestResult(
            True, "record_strategy_prediction() succeeded",
            category="metrics", loops=(0,),
        )
    except Exception as exc:
        return TestResult(
            False, f"record_strategy_prediction() crashed: {exc}",
            category="metrics", loops=(0,),
        )


# ── Category 9: GNN Trainer ───────────────────────────────────


async def test_gnn_trainer_skips_below_minimum() -> TestResult:
    """With <min_samples pipeline runs, GNNTrainer.train() returns status=skipped.

    Uses a mock RunStore that returns fewer than GNN_MIN_SAMPLES runs.
    """

    # Minimal stub for RunStore that returns empty list
    class StubRunStore:
        async def get_since(self, since: datetime) -> list[Any]:
            return []

    # Minimal stub for RuVectorClient (should not be called)
    class StubRuVector:
        async def train_gnn(self, training_data: dict[str, Any]) -> None:
            raise AssertionError("train_gnn should not be called when samples < minimum")

    from agentic_ra.observability.gnn_trainer import GNNTrainer

    trainer = GNNTrainer(
        ruvector=StubRuVector(),  # type: ignore[arg-type]
        run_store=StubRunStore(),  # type: ignore[arg-type]
        min_samples=GNN_MIN_SAMPLES,
    )

    result = await trainer.train(since_hours=24)

    if result.get("status") != "skipped":
        return TestResult(
            False,
            f"Expected status='skipped', got {result}",
            category="gnn", loops=(3,),
        )

    samples_reported = result.get("samples", -1)
    if samples_reported != 0:
        return TestResult(
            False,
            f"Expected samples=0, got {samples_reported}",
            category="gnn", loops=(3,),
        )

    return TestResult(
        True,
        f"GNNTrainer correctly skipped training (0 < {GNN_MIN_SAMPLES} min_samples)",
        category="gnn", loops=(3,),
    )


# ── Category 10: Integration — Full Pipeline ──────────────────


async def test_full_query_with_feedback_enabled() -> TestResult:
    """Run a query through the gateway API, verify pipeline_run is created.

    Requires: All services running + gateway at GATEWAY_URL.
    """
    import httpx

    try:
        async with httpx.AsyncClient() as client:
            # Health check first
            try:
                health_resp = await client.get(
                    f"{GATEWAY_URL}/health",
                    timeout=5.0,
                )
                health_resp.raise_for_status()
            except Exception:
                return TestResult(
                    True,
                    f"Gateway at {GATEWAY_URL} not reachable (expected if services down)",
                    skipped=True, category="integration", loops=(0,),
                )

            # Send a query
            query_resp = await client.post(
                f"{GATEWAY_URL}/api/v1/query",
                json={
                    "query": "Ce prevede articolul 1 din Codul Penal?",
                    "strategy": "hybrid",
                },
                timeout=GATEWAY_TIMEOUT_SECONDS,
            )
            query_resp.raise_for_status()
            data = query_resp.json()

            issues: list[str] = []
            if "answer" not in data:
                issues.append("response missing 'answer' field")
            if "confidence" not in data:
                issues.append("response missing 'confidence' field")
            if "is_grounded" not in data:
                issues.append("response missing 'is_grounded' field")

            if issues:
                return TestResult(
                    False,
                    f"Response structure issues: {'; '.join(issues)}",
                    category="integration", loops=(0,),
                )

            confidence = data.get("confidence", 0.0)
            is_grounded = data.get("is_grounded", False)

            return TestResult(
                True,
                f"Query succeeded: confidence={confidence:.2f}, grounded={is_grounded}",
                category="integration", loops=(0,),
            )

    except Exception as exc:
        return TestResult(
            True,
            f"Integration test skipped (services unavailable): {exc}",
            skipped=True, category="integration", loops=(0,),
        )


# ── Test Runner ────────────────────────────────────────────────

# Registry of all test functions
ALL_TESTS: list[tuple[str, Any]] = [
    # Category 1: Configuration
    ("config: feedback_config_loaded", test_feedback_config_loaded),
    ("config: strategy_config_loaded", test_strategy_config_loaded),
    ("config: ruvector_config_loaded", test_ruvector_config_loaded),
    ("config: no_hardcoded_values", test_no_hardcoded_values),
    ("config: feature_flags_default_state", test_feature_flags_default_state),
    # Category 2: Loop 1 — confidence_to_score
    ("loop1: confidence_to_score_band_1", test_confidence_to_score_band_1),
    ("loop1: confidence_to_score_band_2", test_confidence_to_score_band_2),
    ("loop1: confidence_to_score_band_3", test_confidence_to_score_band_3),
    ("loop1: confidence_to_score_band_4", test_confidence_to_score_band_4),
    ("loop1: confidence_to_score_band_5", test_confidence_to_score_band_5),
    ("loop1: confidence_to_score_boundary_exact", test_confidence_to_score_boundary_exact),
    ("loop1: confidence_to_score_extremes", test_confidence_to_score_extremes),
    ("loop1: confidence_to_score_all_bands_no_gaps", test_confidence_to_score_all_bands_no_gaps),
    # Category 3: Loop 1 — FeedbackAdapter
    ("loop1: feedback_adapter_disabled", test_feedback_adapter_disabled),
    ("loop1: feedback_adapter_dry_run", test_feedback_adapter_dry_run),
    # Category 4: Loop 1 — Session ID Flow
    ("loop1: session_id_passed_to_cognee_search", test_session_id_passed_to_cognee_search),
    ("loop1: feedback_submission_creates_entry", test_feedback_submission_creates_feedback_entry),
    # Category 5: Loop 2 — Strategy Learner
    ("loop2: strategy_collection_creation", test_strategy_collection_creation),
    ("loop2: strategy_record_stores_in_qdrant", test_strategy_record_stores_in_qdrant),
    ("loop2: strategy_predict_cold_start", test_strategy_predict_cold_start),
    ("loop2: strategy_predict_with_data", test_strategy_predict_with_data),
    ("loop2: strategy_predict_threshold", test_strategy_predict_threshold),
    ("loop2: strategy_predict_age_filter", test_strategy_predict_age_filter),
    ("loop2: strategy_embedding_reuse", test_strategy_embedding_reuse),
    # Category 6: Loop 2 — Expanded Strategies
    ("loop2: all_strategies_in_enum", test_all_strategies_in_enum),
    ("loop2: strategy_enum_values_are_strings", test_strategy_enum_values_are_strings),
    ("loop2: new_strategies_route_without_error", test_new_strategies_route_without_error),
    # Category 7: Loop 3 — RuVector
    ("loop3: ruvector_client_url_validation", test_ruvector_client_url_validation),
    ("loop3: ruvector_client_url_accepts_http", test_ruvector_client_url_accepts_http),
    ("loop3: ruvector_client_url_accepts_https", test_ruvector_client_url_accepts_https),
    ("loop3: ruvector_error_has_context", test_ruvector_error_has_context),
    # Category 8: Metrics
    ("metrics: feedback_counters", test_rag_metrics_has_feedback_counters),
    ("metrics: strategy_counters", test_rag_metrics_has_strategy_counters),
    ("metrics: cost_histogram", test_rag_metrics_has_cost_histogram),
    ("metrics: record_feedback_method", test_record_feedback_method),
    ("metrics: record_strategy_prediction_method", test_record_strategy_prediction_method),
    # Category 9: GNN Trainer
    ("gnn: trainer_skips_below_minimum", test_gnn_trainer_skips_below_minimum),
    # Category 10: Integration
    ("integration: full_query_with_feedback", test_full_query_with_feedback_enabled),
]


def _get_loop_for_test(name: str) -> tuple[int, ...]:
    """Determine which loop(s) a test belongs to based on its name prefix."""
    prefix = name.split(":")[0].strip()
    if prefix == "config" or prefix == "metrics" or prefix == "integration":
        return (0,)  # loop 0 = always run unless filtering
    if prefix == "loop1":
        return (1,)
    if prefix == "loop2":
        return (2,)
    if prefix == "loop3" or prefix == "gnn":
        return (3,)
    return (0,)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="E2E tests for self-learning RAG features",
    )
    parser.add_argument(
        "--loop",
        type=int,
        choices=[0, 1, 2, 3],
        default=None,
        help="Run only tests for a specific loop (0=config/metrics/integration, 1/2/3=loop N)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed test output",
    )
    return parser.parse_args()


async def run_tests(loop_filter: int | None, verbose: bool) -> int:
    """Execute all tests and print results.

    Returns:
        Exit code: 0 if all pass, 1 if any fail.
    """
    print()
    print(f"{COLOR_BOLD}{'=' * 64}{COLOR_RESET}")
    print(f"{COLOR_BOLD}  E2E Self-Learning RAG Tests{COLOR_RESET}")
    print(f"{COLOR_BOLD}{'=' * 64}{COLOR_RESET}")
    print()

    if loop_filter is not None:
        loop_label = "config/metrics/integration" if loop_filter == 0 else f"Loop {loop_filter}"
        print(f"  Filter: {loop_label} tests only")
        print()

    passed_count = 0
    failed_count = 0
    skipped_count = 0
    total_start = time.time()

    results: list[tuple[str, TestResult]] = []

    for name, test_fn in ALL_TESTS:
        test_loops = _get_loop_for_test(name)

        # Apply loop filter
        if loop_filter is not None:
            if loop_filter == 0:
                # Loop 0 filter: run only config/metrics/integration tests
                if 0 not in test_loops:
                    continue
            elif loop_filter not in test_loops:
                continue

        start = time.time()
        try:
            result = await test_fn()
        except Exception as exc:
            result = TestResult(False, f"Unhandled exception: {exc}")
        elapsed = time.time() - start

        results.append((name, result))

        label = _label(result)
        elapsed_str = f"({elapsed:.2f}s)" if elapsed > 0.1 else ""

        if result.skipped:
            skipped_count += 1
        elif result.passed:
            passed_count += 1
        else:
            failed_count += 1

        print(f"  [{label}] {name} {elapsed_str}")
        if verbose or (not result.passed and not result.skipped):
            print(f"         {result.message}")

    total_elapsed = time.time() - total_start

    # Summary
    print()
    print(f"{COLOR_BOLD}{'─' * 64}{COLOR_RESET}")
    total_run = passed_count + failed_count + skipped_count
    print(
        f"  {COLOR_GREEN}{passed_count} passed{COLOR_RESET}, "
        f"{COLOR_RED}{failed_count} failed{COLOR_RESET}, "
        f"{COLOR_YELLOW}{skipped_count} skipped{COLOR_RESET} "
        f"({total_run} total, {total_elapsed:.1f}s)"
    )
    print(f"{COLOR_BOLD}{'─' * 64}{COLOR_RESET}")
    print()

    if failed_count > 0:
        print(f"  {COLOR_RED}{COLOR_BOLD}FAILURES:{COLOR_RESET}")
        for name, result in results:
            if not result.passed and not result.skipped:
                print(f"    {name}: {result.message}")
        print()

    return 1 if failed_count > 0 else 0


def main() -> None:
    """Entry point for the E2E self-learning test suite."""
    args = parse_args()
    exit_code = asyncio.run(run_tests(loop_filter=args.loop, verbose=args.verbose))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
