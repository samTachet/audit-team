"""Sorties structurées des rôles (contrats entre nœuds du graphe). Champs supplémentaires interdits."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Finding(_Strict):
    nc_id: str
    regulation: str
    article: str
    article_title: str
    severity: str
    title: str


class AnalysisReport(_Strict):
    dossier_id: str
    client: str | None = None
    findings: list[Finding] = Field(default_factory=list)
    error: str | None = None
    recipient_hint: str | None = None  # ce qu'un agent a « cru lire » dans les données : jamais une autorité
    notes: list[str] = Field(default_factory=list)


class ProposedAction(_Strict):
    tool: Literal["submit_dossier", "delete_draft"]
    args: dict[str, Any]
    proposed_by: str


class Draft(_Strict):
    draft_id: str
    nc_id: str
    text: str


class DraftSet(_Strict):
    drafts: list[Draft] = Field(default_factory=list)
    proposed_action: ProposedAction | None = None
    error: str | None = None


class FeedbackItem(_Strict):
    nc_id: str | None = None
    issue: str


class ReviewVerdict(_Strict):
    verdict: Literal["approve", "request_changes", "block"]
    reasons: list[str] = Field(default_factory=list)
    feedback: list[FeedbackItem] = Field(default_factory=list)


OUTPUT_MODELS: dict[str, type[BaseModel]] = {"analyst": AnalysisReport, "writer": DraftSet, "reviewer": ReviewVerdict}
