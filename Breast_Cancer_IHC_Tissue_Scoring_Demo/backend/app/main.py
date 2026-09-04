"""FastAPI application factory and ASGI entry point.

Run locally with::

    uvicorn app.main:app --reload
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.services import maintenance_service
from app.schemas.common import ErrorDetail
from app.services.qc_service import qc_service
from app.services.tile_service import tile_service

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Configure logging on startup and log a clean shutdown."""
    configure_logging(settings.debug)
    settings.ensure_dirs()

    # Housekeeping that needs no judgement: caches whose slide is gone, and upload parts
    # already reassembled. Both are unreachable by construction, so sweeping them cannot
    # destroy anything anyone could still ask for. Everything that *would* need a
    # judgement stays an explicit act behind /maintenance/cleanup.
    #
    # This is the honest answer to "clear up when the browser closes": a backend cannot
    # tell a closed tab from a refresh, so wiping on that signal would throw away half an
    # hour of step 8 because somebody pressed F5.
    if settings.cleanup_orphans_on_start:
        maintenance_service.sweep_orphans_on_start()
    logger.info(
        "%s v%s starting (%s)",
        settings.app_name,
        settings.app_version,
        settings.environment,
    )
    yield
    tile_service.close_all()  # release open slide file handles
    qc_service.shutdown()  # drop the cached QC models and their device memory
    logger.info("%s shutting down", settings.app_name)


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Read-only API behind the Breast Cancer IHC Tissue Scoring demo. It serves the "
            "16-step pipeline catalogue and runs the first five steps for real against an "
            "uploaded slide: reading its pyramid, GrandQC quality control, the tissue "
            "mask, white calibration, and optical density."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.backend_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix=settings.api_v1_prefix)

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
        """Return a normalised error body instead of a bare stack trace."""
        logger.exception("Unhandled error", exc_info=exc)
        payload = ErrorDetail(code="internal_error", message="An unexpected error occurred.")
        return JSONResponse(status_code=500, content=payload.model_dump(by_alias=True))

    @app.get("/", tags=["meta"], summary="Service root")
    async def root() -> dict[str, str]:
        """Point callers at the docs and the versioned API."""
        return {
            "service": settings.app_name,
            "version": settings.app_version,
            "docs": "/docs",
            "api": settings.api_v1_prefix,
        }

    return app


app = create_app()
