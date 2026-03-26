import asyncio
import socket
from agentic_ra.core.config import get_settings

def check_port(host: str, port: int, timeout: float = 1.0) -> bool:
    """Check if a port is open."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, ConnectionRefusedError):
        return False

async def verify_infrastructure():
    settings = get_settings()
    services = {
        "Qdrant (HTTP)": (settings.qdrant_url.split("//")[1].split(":")[0], settings.qdrant_http_port),
        "OpenSearch": (settings.opensearch_url.split("//")[1].split(":")[0], settings.opensearch_port),
        "PostgreSQL": (settings.postgres_host, settings.postgres_port),
        "OPA": (settings.opa_url.split("//")[1].split(":")[0], settings.opa_port),
        "OTel Collector": (settings.otel_endpoint.split("//")[1].split(":")[0], settings.otel_collector_port),
        "Jaeger UI": (settings.jaeger_ui_url.split("//")[1].split(":")[0], settings.jaeger_ui_port),
    }

    print("--- Infrastructure Health Check ---")
    all_healthy = True
    for name, (host, port) in services.items():
        is_healthy = check_port(host, port)
        status = "✅ HEALTHY" if is_healthy else "❌ UNREACHABLE"
        print(f"{name:15} | {host}:{port:<5} | {status}")
        if not is_healthy:
            all_healthy = False

    if all_healthy:
        print("\nAll infrastructure services are reachable.")
    else:
        print("\nSome infrastructure services are unreachable. Please run 'docker-compose up -d'.")
        exit(1)

if __name__ == "__main__":
    asyncio.run(verify_infrastructure())
