"""Health check endpoints — Episode 25."""
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    status:  str
    version: str


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(status="ok", version="0.1.0")


@router.get("/ready")
async def readiness():
    """Readiness probe — checks DB connectivity."""
    try:
        from rag.ingestion.indexer import VectorIndex
        idx = VectorIndex()
        n   = idx.count()
        return {"status": "ready", "chunks_indexed": n}
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail=f"DB not ready: {e}")
