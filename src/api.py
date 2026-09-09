from functools import lru_cache
from typing import Literal, Optional

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field, model_validator

from src.bandit import RecommendationPolicy


app = FastAPI(
    title="FIAP Datathon - Recommendation API",
    version="1.0.0",
)


class RecommendationRequest(BaseModel):
    previous: Optional[int] = Field(
        default=None,
        ge=0,
        description=(
            "Quantidade de contatos anteriores com o cliente. "
            "Utilizado para derivar o contexto quando 'context' não é informado."
        ),
    )
    context: Optional[Literal["none", "one", "two_plus"]] = Field(
        default=None,
        description=(
            "Contexto do cliente. Quando informado, tem prioridade sobre 'previous'. "
            "Valores aceitos: 'none', 'one', 'two_plus'."
        ),
    )
    mode: Literal["deterministic", "thompson"] = "deterministic"

    @model_validator(mode="after")
    def check_context_or_previous(self) -> "RecommendationRequest":
        if self.context is None and self.previous is None:
            raise ValueError(
                "Informe 'previous' ou 'context'. Pelo menos um dos campos é obrigatório."
            )
        return self


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
    if request.context is not None:
        return policy.recommend_by_context(
            context=request.context,
            mode=request.mode,
        )

    return policy.recommend(
        previous=request.previous,
        mode=request.mode,
    )
