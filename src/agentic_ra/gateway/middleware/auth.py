"""OPA authorization middleware for FastAPI."""

from __future__ import annotations

import logging

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse

from agentic_ra.auth.opa import OPAClient

logger = logging.getLogger(__name__)

# Paths that skip authorization
_PUBLIC_PATHS = frozenset({"/health", "/docs", "/openapi.json", "/redoc"})


class OPAAuthMiddleware(BaseHTTPMiddleware):
    """Middleware that checks OPA for request authorization.

    Skips public paths (health, docs). For all other requests,
    sends the request context to OPA for a policy decision.

    Args:
        app: The ASGI application.
        opa_client: OPA client for policy checks.
        enabled: Whether authorization is enforced. Disable for dev.
    """

    def __init__(
        self,
        app: object,
        opa_client: OPAClient,
        enabled: bool = True,
    ) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self._opa = opa_client
        self._enabled = enabled

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Check authorization before passing to the route handler."""
        if not self._enabled:
            return await call_next(request)

        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        # Build OPA input from request
        input_data = {
            "method": request.method,
            "path": request.url.path,
            "headers": dict(request.headers),
            "user": request.headers.get("x-user-id", "anonymous"),
            "role": request.headers.get("X-User-Role", "user"),
        }

        try:
            decision = await self._opa.check(input_data)
        except Exception:
            logger.exception("OPA check failed, denying request")
            return JSONResponse(
                status_code=503,
                content={"detail": "Authorization service unavailable"},
            )

        if not decision.allowed:
            return JSONResponse(
                status_code=403,
                content={"detail": "Forbidden", "reason": decision.reason},
            )

        return await call_next(request)
