from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

from app.api import documents, health, ledger_rules
from app.config import settings
from app.integrations.ocr.nemotron_client import check_ollama_health

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Make sure storage/ directory exists on startup
    import os
    os.makedirs("storage", exist_ok=True)
    # Perform health ping check on startup
    await check_ollama_health()
    yield


app = FastAPI(
    title="Tally Prime Integration Service",
    description="AI-powered pipeline for automated voucher ingestion and Tally Prime posting.",
    version="0.1.0",
    lifespan=lifespan,
    # Disable public Swagger/ReDoc docs — this is an internal tool, not a public API
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


# Global exception handler — never leak internal tracebacks to the client
@app.exception_handler(Exception)
async def _global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(
        "Unhandled exception on %s %s: %s",
        request.method, request.url.path, exc, exc_info=True
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred. Please try again."},
    )

# CORS — restrict to localhost only (this is a local desktop tool)
# If you ever deploy to a server, add that domain here explicitly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:8000",
        "http://127.0.0.1",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "Accept"],
)

# Register API Routers
from app.api.documents import router as documents_router
from app.api.ledger_rules import router as ledger_rules_router
from app.api.companies import router as companies_router

app.include_router(health.router)
app.include_router(documents_router, prefix="/api")
app.include_router(ledger_rules_router, prefix="/api")
app.include_router(companies_router, prefix="/api")


from fastapi.responses import HTMLResponse


@app.get("/", response_class=HTMLResponse)
async def root():
    """
    Serves the premium single-page application dashboard for automated voucher extraction.
    """
    try:
        with open("app/static/index.html", "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return HTMLResponse(
            content=f"<h3>Error loading index.html: {str(e)}</h3>",
            status_code=500
        )
