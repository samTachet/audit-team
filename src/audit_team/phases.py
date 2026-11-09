"""Exécution d'une phase (un rôle, une tâche) : interface commune et runner direct sans framework (P1).

`PhaseRunner.run(mission_id, role, inputs, brain) -> PhaseResult` :
- chaque appel du « modèle » est précédé de `TeamRuntime.check_llm` et suivi de `charge_llm` (budget par rôle) ;
- chaque tool passe par `TeamRuntime.propose` (gateway) : aucun accès direct aux services ;
- une exception du modèle remonte en `PhaseError` (le graphe clôt alors la mission en `PLANNER_ERROR`).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from acsi_agent.prompt import estimate_tokens
from acsi_agent.team import TeamRuntime

from .doubles import Act, Brain, Final


@dataclass
class PhaseResult:
    output: dict[str, Any] | None
    stopped: bool = False  # la phase a été interrompue par ACSI
    reason: str | None = None  # budget | denials | closed
    outcomes: list[dict[str, Any]] = field(default_factory=list)


class PhaseError(Exception):
    def __init__(self, role: str, cause: BaseException | str):
        super().__init__(f"{role}: {cause}")
        self.role, self.cause = role, cause


class PhaseRunner(Protocol):
    id: str

    def run(self, mission_id: str, role: str, inputs: dict[str, Any], brain: Brain) -> PhaseResult: ...


def usage_for(inputs: dict[str, Any], outcomes: list[dict[str, Any]], decision: Act | Final) -> dict[str, int]:
    """Décompte déterministe (≈ 4 caractères / token) : identique quel que soit le runner, donc rejouable.
    Un LLM réel (P4) fournit son propre décompte via la même interface."""
    seen = json.dumps({"inputs": inputs, "outcomes": outcomes}, ensure_ascii=False, sort_keys=True, default=str)
    out = json.dumps({"tool": decision.tool, "args": decision.args} if isinstance(decision, Act)
                     else decision.output, ensure_ascii=False, sort_keys=True)
    return {"tokens_in": estimate_tokens(seen), "tokens_out": estimate_tokens(out)}


def summary_of(decision: Act | Final) -> dict[str, Any]:
    if isinstance(decision, Act):
        return {"kind": "tool_call", "tool": decision.tool, "args": decision.args}
    return {"kind": "final", "output": decision.output}


def decide(rt: TeamRuntime, mid: str, role: str, brain: Brain, inputs: dict[str, Any],
           outcomes: list[dict[str, Any]]) -> Act | Final | None:
    """Un « appel LLM » gouverné : budget avant, imputation après. None = arrêt imposé par ACSI."""
    if rt.check_llm(mid, role):
        return None
    t0 = time.perf_counter()
    try:
        decision = brain.next(inputs, outcomes)
    except Exception as e:  # noqa: BLE001 — I4 : l'erreur d'un modèle est un résultat, pas un crash
        raise PhaseError(role, e) from e
    lat = (time.perf_counter() - t0) * 1000
    if not isinstance(decision, (Act, Final)):
        raise PhaseError(role, f"sortie invalide : {type(decision).__name__}")
    if rt.charge_llm(mid, role, usage_for(inputs, outcomes, decision), lat, summary_of(decision)):
        return None
    return decision


class DirectPhaseRunner:
    """Runner de référence, sans CrewAI : boucle décision → gateway, bornée par `max_iter`."""
    id = "direct-runner-v1"

    def __init__(self, rt: TeamRuntime, max_iter: int = 30):
        self.rt, self.max_iter = rt, max_iter

    def run(self, mission_id: str, role: str, inputs: dict[str, Any], brain: Brain) -> PhaseResult:
        outcomes: list[dict[str, Any]] = []
        for _ in range(self.max_iter):
            d = decide(self.rt, mission_id, role, brain, inputs, outcomes)
            if d is None:
                return PhaseResult(None, stopped=True, reason="budget", outcomes=outcomes)
            if isinstance(d, Final):
                return PhaseResult(d.output, outcomes=outcomes)
            out = self.rt.propose(mission_id, role, d.tool, d.args)
            outcomes.append({"tool": d.tool, "args": d.args, **out})
            if out.get("code") in {"BUDGET_EXCEEDED", "MISSION_CLOSED"}:
                return PhaseResult(None, stopped=True, reason="budget", outcomes=outcomes)
            if self.rt.denial_limit_reached(mission_id, role):
                return PhaseResult(None, stopped=True, reason="denials", outcomes=outcomes)
        raise PhaseError(role, f"max_iter ({self.max_iter}) atteint sans conclusion")
