"""
FastAPI application factory with lifespan, middleware, and router registration.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from medscan.api.routers import documents, health, patients, query
from medscan.config import get_settings
from medscan.db.session import close_db, init_db

logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: startup and shutdown logic."""
    # Startup
    logger.info("MedScan AI starting up...")
    settings.ensure_upload_dir()
    await init_db()
    logger.info("Database initialized. Upload dir: %s", settings.upload_dir)

    yield

    # Shutdown
    logger.info("MedScan AI shutting down...")
    await close_db()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="MedScan AI",
        description=(
            "Multimodal Clinical Document Intelligence Pipeline\n\n"
            "Ingest messy medical documents (PDFs, images, handwritten notes) and "
            "query patient intelligence using multi-agent AI reasoning."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ---------------------------------------------------------------------------
    # CORS
    # ---------------------------------------------------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if not settings.is_production else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---------------------------------------------------------------------------
    # Global exception handler
    # ---------------------------------------------------------------------------
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "error": str(exc)},
        )

    # ---------------------------------------------------------------------------
    # Routers
    # ---------------------------------------------------------------------------
    app.include_router(health.router, prefix="/health", tags=["Health"])
    app.include_router(documents.router, prefix="/documents", tags=["Documents"])
    app.include_router(patients.router, prefix="/patients", tags=["Patients"])
    app.include_router(query.router, prefix="/query", tags=["Clinical Query"])

    return app


app = create_app()


def run() -> None:
    """Entry point for uvicorn (used by pyproject.toml script)."""
    import uvicorn
    uvicorn.run(
        "medscan.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.app_env == "development",
        log_level=settings.log_level.lower(),
    )
