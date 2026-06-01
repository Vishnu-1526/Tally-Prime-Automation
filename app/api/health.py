from fastapi import APIRouter
from sqlalchemy import text

from app.database import get_db_session

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check():
    """
    Standard liveness probe to verify the web service is running.
    """
    return {
        "status": "healthy",
        "service": "Tally Prime AI API",
        "version": "0.1.0"
    }


@router.get("/health/db")
async def db_health_check():
    """
    Readiness probe that checks database connection health.
    """
    try:
        async with get_db_session() as session:
            await session.execute(text("SELECT 1"))
        return {
            "status": "healthy",
            "database": "connected"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": "error",
            "detail": str(e)
        }
    # Wait, the get_db_session context manager commits automatically, but SELECT 1 is read-only.
