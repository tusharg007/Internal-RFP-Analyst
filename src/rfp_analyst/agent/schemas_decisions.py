"""Pydantic schemas for structured agent decisions."""

from typing import Literal

from pydantic import BaseModel, Field

from rfp_analyst.retrieval.decisions import RetrievalDecision

__all__ = ["RouteDecision", "EvidenceGrade", "QueryRewrite", "RetrievalDecision"]


class RouteDecision(BaseModel):
    """Choose whether a question needs private knowledge-base retrieval."""

    route: Literal["kb", "direct"] = Field(
        description=(
            "Use 'kb' for questions requiring document retrieval or evidence; "
            "use 'direct' for greetings, thanks, or simple conversation."
        )
    )


class EvidenceGrade(BaseModel):
    """Assess whether retrieved evidence sufficiently answers a question."""

    grade: Literal["good", "weak"] = Field(
        description=(
            "Use 'good' when the evidence is relevant and sufficient to answer the question; "
            "use 'weak' when it is irrelevant, incomplete, or insufficient."
        )
    )


class QueryRewrite(BaseModel):
    """Represent a search-optimized rewrite of a user query."""

    rewritten_query: str = Field(
        description="A search-optimized version of the original question."
    )
