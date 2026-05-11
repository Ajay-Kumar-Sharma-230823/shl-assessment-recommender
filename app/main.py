"""
SHL Assessment Recommendation System — Main Application
========================================================
FastAPI application with:
- Two endpoints: GET /health, POST /chat
- Lifespan context manager for startup/shutdown
- CORS middleware
- Exception handlers
- Request logging middleware
- OpenAPI documentation
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.models.config import get_settings
from app.routes.chat import router
from app.services.startup import shutdown, startup
from app.utils.logging_config import configure_logging

# ---- Configure logging first ----
settings = get_settings()
configure_logging(log_level=settings.log_level, env=settings.app_env)
logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"


# ============================================================
# Lifespan Context Manager
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    # Startup
    await startup()
    yield
    # Shutdown
    await shutdown()


# ============================================================
# FastAPI Application
# ============================================================
app = FastAPI(
    title="SHL Assessment Recommendation System",
    description=(
        "Conversational AI agent that helps recruiters and hiring managers "
        "discover the correct SHL assessments through natural conversation. "
        "All recommendations are grounded in the official SHL product catalog."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
    contact={
        "name": "SHL Assessment Advisor",
        "url": "https://www.shl.com/solutions/products/product-catalog/",
    },
)


# ============================================================
# CORS Middleware
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ============================================================
# Request Logging Middleware
# ============================================================
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests with timing."""
    start_time = time.time()
    method = request.method
    path = request.url.path

    logger.info(f"→ {method} {path}")

    try:
        response = await call_next(request)
        elapsed = time.time() - start_time
        logger.info(f"← {method} {path} — {response.status_code} ({elapsed:.3f}s)")
        return response
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"← {method} {path} — ERROR: {e} ({elapsed:.3f}s)")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )


# ============================================================
# Exception Handlers
# ============================================================
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "detail": f"Endpoint not found: {request.url.path}. "
            "Available endpoints: GET /health, POST /chat"
        },
    )


@app.exception_handler(405)
async def method_not_allowed_handler(request: Request, exc):
    return JSONResponse(
        status_code=405,
        content={
            "detail": f"Method {request.method} not allowed for {request.url.path}"
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again."},
    )


# ============================================================
# Include Routes
# ============================================================
app.include_router(router, prefix="")


# ============================================================
# Root endpoint
# ============================================================
@app.get("/", include_in_schema=False)
async def root():
    """Serve the recruiter-facing assessment advisor workspace."""
    return FileResponse(STATIC_DIR / "index.html")


# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_env == "development",
        log_level=settings.log_level.lower(),
        workers=1,
    )
