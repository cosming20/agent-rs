#!/usr/bin/env bash
# tool-routing-smoke.sh
#
# Manual end-to-end smoke test that each query routes to the intended
# tool. Run after any change to the system prompt or tool descriptions
# to catch regressions in the agent's tool-selection decisions.
#
# Each query below is crafted so that exactly one tool should fire on
# the first conversation turn. The script greps the trace output for
# the expected tool invocation and reports PASS / FAIL per query.
#
# Runs `just ask` for each probe, so requires all infra up (`just
# infra-up`, `just obs-up` optional) and an ingested corpus. The
# probes are corpus-agnostic: they exercise the agent's decision
# logic, not the retrieved content. The `search_knowledge` probe
# asks a generic domain question that any ingested corpus should be
# able to route (even if the retrieval comes back empty — we only
# check that the tool fires).
#
# Usage:
#   bash scripts/tool-routing-smoke.sh               # run all probes
#   bash scripts/tool-routing-smoke.sh document      # only probes matching
#   VERBOSE=1 bash scripts/tool-routing-smoke.sh     # stream full logs

set -euo pipefail

FILTER="${1:-}"
VERBOSE="${VERBOSE:-0}"
PASS=0
FAIL=0
FAILED_PROBES=()

# Run one probe: $1 = label, $2 = query, $3 = expected tool name
# (substring match in the ask log output).
probe() {
    local label="$1"
    local query="$2"
    local expected_tool="$3"

    if [[ -n "$FILTER" ]] && [[ "$label" != *"$FILTER"* ]]; then
        return 0
    fi

    echo
    echo "━━━ probe: $label"
    echo "    query:    $query"
    echo "    expected: executing_knowledge search | $expected_tool"

    # Capture the first 5000 lines of `just ask` output; that's plenty
    # for the first conversation turn even on long tool responses.
    local out
    if ! out=$(just ask "$query" 2>&1 | head -n 5000); then
        echo "    result:   FAIL  (just ask exited non-zero)"
        FAIL=$((FAIL + 1))
        FAILED_PROBES+=("$label")
        return 0
    fi

    if [[ "$VERBOSE" == "1" ]]; then
        echo "--- full output ---"
        echo "$out"
        echo "--- end output ---"
    fi

    # Check for the expected tool invocation. Two signals are accepted:
    # 1. The agent-runner emits a `tool: "<name>"` span field when rig
    #    dispatches the call.
    # 2. The tool's own `tracing::instrument` span fires with
    #    `tool = Self::NAME`.
    if echo "$out" | grep -qE "tool[:=] ?\"?${expected_tool}\"?"; then
        echo "    result:   PASS"
        PASS=$((PASS + 1))
    else
        echo "    result:   FAIL  (expected tool ${expected_tool} did not fire)"
        FAIL=$((FAIL + 1))
        FAILED_PROBES+=("$label")
    fi
}

echo "================================================"
echo "  Tool routing smoke test — agent-rs"
echo "================================================"

probe "document_info / list corpus" \
    "What documents are in the knowledge base?" \
    "document_info"

probe "document_info / is doc ingested" \
    "Do you have any PDFs ingested?" \
    "document_info"

probe "search_knowledge / conceptual question" \
    "Explain the core concept discussed in section 2.1 of the ingested material." \
    "search_knowledge"

probe "search_knowledge / keyword identifier" \
    "What does section 2.3 say?" \
    "search_knowledge"

probe "fetch_url / explicit URL in query" \
    "Fetch the content of https://example.com/" \
    "fetch_url"

probe "remember / explicit save request" \
    "Please remember that I prefer concise answers with citations." \
    "remember"

echo
echo "================================================"
echo "  Summary: ${PASS} passed, ${FAIL} failed"
if [[ ${FAIL} -gt 0 ]]; then
    echo "  Failed:"
    for p in "${FAILED_PROBES[@]}"; do
        echo "    - $p"
    done
fi
echo "================================================"

if [[ ${FAIL} -gt 0 ]]; then
    exit 1
fi
