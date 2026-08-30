from functools import lru_cache
from typing import Literal

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from src.bandit import RecommendationPolicy


app = FastAPI(
    title="FIAP Datathon - Recommendation API",
    version="1.0.0",
)


class RecommendationRequest(BaseModel):
    previous: int = Field(
        ge=0,
        description="Quantidade de contatos anteriores com o cliente.",
    )
    mode: Literal["deterministic", "thompson"] = "deterministic"


class RecommendationResponse(BaseModel):
    context: str
    recommended_channel: str
    mode: str
    posterior_estimates: dict[str, float]
    decision_scores: dict[str, float]


@lru_cache
def get_policy() -> RecommendationPolicy:
    return RecommendationPolicy()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post(
    "/recommend",
    response_model=RecommendationResponse,
)
def recommend(
    request: RecommendationRequest,
    policy: RecommendationPolicy = Depends(get_policy),
):
    return policy.recommend(
        previous=request.previous,
        mode=request.mode,
    )
