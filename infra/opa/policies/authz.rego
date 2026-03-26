# ============================================================
# Agentic RAG — RBAC Authorization Policy
# ============================================================
# Roles: admin (full access), user (own data only).
# Data isolation is enforced at the tool/dataset level, not here.
# OPA enforces endpoint-level access based on role.
# ============================================================

package authz

default allow := false

# Health and docs are always public
allow if {
    input.method == "GET"
    input.path == "/health"
}
allow if {
    input.method == "GET"
    startswith(input.path, "/docs")
}
allow if {
    input.method == "GET"
    input.path == "/openapi.json"
}

# Admin can do everything
allow if {
    input.role == "admin"
}

# Users can query (data isolation enforced at tool level)
allow if {
    input.role == "user"
    input.method == "POST"
    input.path == "/api/v1/query"
}

# Users can ingest their own documents
allow if {
    input.role == "user"
    input.method == "POST"
    input.path == "/api/v1/ingest"
}
