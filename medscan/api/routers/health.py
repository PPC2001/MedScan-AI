"""Health check router."""

import time

from fastapi import APIRouter

from medscan.models.schemas import HealthResponse

router = APIRouter()
_start_time = time.time()


@router.get("/", response_model=HealthResponse, summary="Health Check")
async def health_check() -> HealthResponse:
    """Check API health, service connectivity, and active LLM provider."""
    from medscan.config import get_settings
    from medscan.llm import describe_active_provider

    cfg = get_settings()

    services: dict[str, str] = {
        "llm_provider": describe_active_provider(),
    }

    # Check DB
    try:
        from medscan.db.session import engine
        async with engine.connect() as conn:
            await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        services["postgres"] = "healthy"
    except Exception as e:
        services["postgres"] = f"unhealthy: {e}"

    # Check Redis
    try:
        import redis.asyncio as aioredis
        r = aioredis.from_url(cfg.redis_url, socket_connect_timeout=2)
        await r.ping()
        await r.aclose()
        services["redis"] = "healthy"
    except Exception as e:
        services["redis"] = f"unhealthy: {e}"

    infra_services = {k: v for k, v in services.items() if k != "llm_provider"}
    all_healthy = all(v == "healthy" for v in infra_services.values())

    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        version="0.1.0",
        services=services,
    )
