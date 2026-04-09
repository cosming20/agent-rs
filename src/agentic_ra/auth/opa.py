"""OPA (Open Policy Agent) integration for request authorization."""


import logging
from dataclasses import dataclass
from typing import Any

import httpx

from agentic_ra.core.errors import OPAError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AuthzDecision:
    """Authorization decision from OPA.

    Attributes:
        allowed: Whether the request is authorized.
        reason: Explanation of the decision.
        policy_path: OPA policy path that was evaluated.
    """

    allowed: bool
    reason: str = ""
    policy_path: str = ""


class OPAClient:
    """Async OPA client for policy-based authorization.

    Evaluates authorization decisions against OPA's REST API.

    Args:
        base_url: OPA server base URL (e.g., ``http://localhost:1055``).
        default_policy_path: Default policy path for authorization queries.
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str,
        default_policy_path: str = "v1/data/authz/allow",
        timeout: float = 5.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._default_policy_path = default_policy_path
        self._client = httpx.AsyncClient(timeout=timeout)

    async def check(
        self,
        input_data: dict[str, Any],
        policy_path: str | None = None,
    ) -> AuthzDecision:
        """Evaluate an authorization policy against OPA.

        Args:
            input_data: Input context for the policy evaluation
                (e.g., user, action, resource).
            policy_path: OPA policy path to query. Defaults to constructor value.

        Returns:
            Authorization decision.
        """
        path = policy_path or self._default_policy_path
        url = f"{self._base_url}/{path}"

        try:
            response = await self._client.post(
                url,
                json={"input": input_data},
            )
            response.raise_for_status()
            body = response.json()

            # OPA returns {"result": true/false} for data queries
            result = body.get("result", False)
            if not isinstance(result, bool):
                raise OPAError(
                    "Unexpected OPA result type",
                    result_type=type(result).__name__,
                    operation="check",
                )
            allowed = result

            return AuthzDecision(
                allowed=allowed,
                reason="allowed" if allowed else "denied by policy",
                policy_path=path,
            )

        except httpx.HTTPStatusError as e:
            raise OPAError(
                "OPA policy evaluation failed",
                policy_path=path,
                status_code=e.response.status_code,
                operation="check",
                original_error=str(e),
            ) from e
        except httpx.HTTPError as e:
            raise OPAError(
                "OPA connection failed",
                url=url,
                operation="check",
                original_error=str(e),
            ) from e

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()
