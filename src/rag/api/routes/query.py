"""
Query endpoints — Episode 25
================================
Streaming SSE endpoint + synchronous endpoint.
Auth via Bearer token. Rate limiting via slowapi.
"""
from __future__ import annotations
import json, logging, time
from typing import AsyncIterator
from fastapi import APIRouter, Depends, HTTPException, Security
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from rag.config.settings import get_settings

logger = logging.getLogger(__name__)
router  = APIRouter()
bearer  = HTTPBearer(auto_error=False)


class QueryRequest(BaseModel):
    question: str        = Field(..., min_length=3, max_length=2000)
    filters:  dict       = Field(default_factory=dict)
    stream:   bool       = Field(default=True)
    use_agent: bool      = Field(default=False, description="Use LangGraph agent (Phase 3)")


class QueryResponse(BaseModel):
    answer:   str
    sources:  list[dict] = Field(default_factory=list)
    metadata: dict       = Field(default_factory=dict)


def verify_api_key(credentials: HTTPAuthorizationCredentials | None = Security(bearer)):
    settings = get_settings()
    expected = settings.api_key.get_secret_value()
    if not credentials or credentials.credentials != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return credentials.credentials


@router.post("/query", response_model=QueryResponse)
async def query_sync(req: QueryRequest,
                     _: str = Depends(verify_api_key)) -> QueryResponse:
    """Synchronous query — returns complete answer."""
    t0 = time.perf_counter()
    if req.use_agent:
        from rag.agent.graph import run_agent
        result  = run_agent(req.question)
        answer  = result.get("answer", "")
        sources = result.get("sources", [])
        meta    = {"elapsed": round(time.perf_counter() - t0, 2),
                   "strategy": result.get("strategy"),
                   "rewrites":  result.get("rewrites", 0)}
    else:
        from rag.retrieval.vector_retriever import VectorRetriever
        from rag.chains.rag_chain import invoke, format_docs
        ret    = VectorRetriever(filters=req.filters)
        docs   = ret.retrieve(req.question)
        answer = invoke(req.question, ret)
        sources = [{"n": i+1,
                    "title":   d.metadata.get("report_title",""),
                    "country": d.metadata.get("country",""),
                    "year":    d.metadata.get("year",""),
                    "page":    d.metadata.get("page_number")}
                   for i, d in enumerate(docs[:5])]
        meta = {"elapsed": round(time.perf_counter() - t0, 2)}
    return QueryResponse(answer=answer, sources=sources, metadata=meta)


@router.post("/query/stream")
async def query_stream(req: QueryRequest,
                       _: str = Depends(verify_api_key)) -> StreamingResponse:
    """Streaming SSE endpoint — tokens arrive in real time."""
    async def event_stream() -> AsyncIterator[str]:
        try:
            if req.use_agent:
                from rag.agent.graph import stream_agent
                for node, update in stream_agent(req.question):
                    data = json.dumps({"node": node, "update": str(update)[:200]})
                    yield f"data: {data}\n\n"
                    if "answer" in update:
                        yield f"data: {json.dumps({'token': update['answer']})}\n\n"
            else:
                from rag.retrieval.vector_retriever import VectorRetriever
                from rag.chains.rag_chain import astream
                ret = VectorRetriever(filters=req.filters)
                async for token in astream(req.question, ret):
                    yield f"data: {json.dumps({'token': token})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error("Stream error: %s", e)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})
