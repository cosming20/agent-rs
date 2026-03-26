# ============================================================
# Agentic RAG — Authorization Policy
# ============================================================
# OPA evaluates this policy on every request via the auth middleware.
# Default: allow all (dev mode). Tighten for production.
# ============================================================

package authz

# Dev mode: allow everything. Replace with real rules for production.
default allow := true

# --- Production rules (uncomment to enforce) ---

# # Block anonymous users from API endpoints
# deny if {
#     input.user == "anonymous"
#     startswith(input.path, "/api/")
# }

# # Always allow health checks
# allow if {
#     input.method == "GET"
#     input.path == "/health"
# }

# # Allow authenticated users full API access
# allow if {
#     input.user != "anonymous"
#     startswith(input.path, "/api/")
# }

# # Restrict ingestion to admin users
# deny if {
#     input.path == "/api/v1/ingest"
#     input.user != "admin"
# }
