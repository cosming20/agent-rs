# ============================================================
# Agentic RAG — Authorization Policy (Skeleton)
# ============================================================
# Placeholder policy for OPA health check verification.
# Real authorization rules are implemented in Phase 6 (auth/opa).
# ============================================================

package authz

default allow := false

allow if {
	input.method == "GET"
	input.path == ["health"]
}
