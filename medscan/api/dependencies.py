"""Shared FastAPI dependencies."""

from typing import Annotated

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from sqlalchemy.ext.asyncio import AsyncSession

from medscan.config import get_settings
from medscan.db.session import get_session

settings = get_settings()

# ---------------------------------------------------------------------------
# Database session dependency
# ---------------------------------------------------------------------------
SessionDep = Annotated[AsyncSession, Depends(get_session)]

# ---------------------------------------------------------------------------
# API key authentication
# ---------------------------------------------------------------------------
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    api_key: str | None = Security(api_key_header),
) -> str:
    """Validate X-API-Key header."""
    if api_key is None or api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Pass X-API-Key header.",
        )
    return api_key


AuthDep = Annotated[str, Depends(verify_api_key)]
