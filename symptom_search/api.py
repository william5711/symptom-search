"""HTTP API: uvicorn symptom_search.api:app

The model and the symptom vectors are loaded once at startup, so a request only
encodes the query. The search endpoint is a plain function, so FastAPI runs it in
a worker thread and a slow search doesn't block the event loop.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Callable, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .__main__ import DISCLAIMER
from .matcher import SymptomMatcher

MAX_TEXT_LENGTH = 2000
MAX_TOP_K = 50


class SearchRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT_LENGTH)
    top_k: int = Field(default=10, ge=1, le=MAX_TOP_K)


class Evidence(BaseModel):
    clause: str
    symptom: str
    similarity: float


class Result(BaseModel):
    condition: str
    score: float
    evidence: List[Evidence]


class SearchResponse(BaseModel):
    results: List[Result]
    disclaimer: str = DISCLAIMER


def _default_matcher() -> SymptomMatcher:
    # Imported here so the tests can build the app without loading torch.
    from .data import load_diseases
    from .embeddings import Encoder

    return SymptomMatcher(load_diseases(), Encoder())


def create_app(build_matcher: Callable[[], SymptomMatcher] = _default_matcher) -> FastAPI:
    matcher: Optional[SymptomMatcher] = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal matcher
        matcher = build_matcher()
        yield

    app = FastAPI(title="Symptom Search", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        if matcher is None:
            raise HTTPException(status_code=503, detail="still loading")
        return {"status": "ok", "conditions": len(matcher.diseases)}

    @app.post("/search", response_model=SearchResponse)
    def search(request: SearchRequest) -> SearchResponse:
        if not request.text.strip():
            raise HTTPException(status_code=422, detail="text is blank")
        matches = matcher.search(request.text, top_k=request.top_k)
        return SearchResponse(
            results=[
                Result(
                    condition=match.disease,
                    score=match.score,
                    evidence=[Evidence(clause=e.clause, symptom=e.symptom, similarity=e.similarity) for e in match.evidence],
                )
                for match in matches
            ]
        )

    return app


app = create_app()
