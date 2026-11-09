"""Replay d'une mission (EF-10) : les décisions enregistrées de chaque rôle (`agent_output`) et les décisions
humaines (`approval_resolved`) sont rejouées dans la VRAIE gateway, sur un journal temporaire.

Toute divergence de trajectoire (tools, arguments, issues, verdicts, relais, approbations, fin) est signalée
avec `identical: false`. Le replay utilise le runner direct : il éprouve ACSI et le graphe, pas CrewAI.
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from acsi_agent.agent import ApprovalError
from acsi_agent.idempotency import IdempotencyRegistry
from acsi_agent.team import MissionInput, TeamRuntime

from .doubles import Act, BrainFactory, Final
from .service import MissionService


@dataclass
class RecordedBrain:
    decisions: list[dict[str, Any]] = field(default_factory=list)
    id: str = "recorded-brain"
    i: int = 0

    def next(self, inputs, outcomes):
        if self.i >= len(self.decisions):
            raise RuntimeError("replay : plus de sortie enregistrée pour ce rôle")
        d = self.decisions[self.i]
        self.i += 1
        if d["kind"] == "tool_call":
            return Act(d["tool"], dict(d["args"]))
        return Final(d["output"])


def _norm(e: dict[str, Any]) -> tuple | None:
    t, d, a = e["type"], e["data"], e.get("actor")
    if t == "tool_call":
        return (t, a, d["tool"], d["args"], d["status"], (d["error"] or {}).get("code"), d["executed"])
    if t == "review_verdict":
        return (t, a, d["round"], d["verdict"])
    if t == "handoff":
        return (t, d["from"], d["to"], d["reason"])
    if t == "approval_requested":
        return (t, d["tool"], d["args_hash"])
    if t == "approval_resolved":
        return (t, d["decision"])
    if t == "budget_exceeded":
        return (t, d["scope"], d["role"], d["dimension"])
    if t == "agent_error":
        return (t, a, d.get("code"))
    if t == "mission_finished":
        return (t, d["status"], d.get("code"), d["effects"])
    return None


def trajectory(events: list[dict[str, Any]]) -> list[tuple]:
    return [n for n in map(_norm, events) if n is not None]


def replay_mission(rt: TeamRuntime, mid: str) -> dict[str, Any]:
    events = rt.journal.read(mid)
    ok, why = rt.journal.verify(mid)
    base = {"mission_id": mid, "chain_ok": ok, "chain_detail": why}
    if not events or events[0]["type"] != "mission_started":
        return {**base, "replayable": False, "identical": False, "reason": "journal vide ou invalide"}
    if not any(e["type"] == "mission_finished" for e in events):
        return {**base, "replayable": False, "identical": False, "reason": "mission non terminée"}
    if any(e["type"] == "budget_exceeded" and e["data"]["dimension"] == "wall" for e in events):
        return {**base, "replayable": False, "identical": False, "reason": "budget de temps : dépend de l'horloge"}
    if any(e["type"] == "approval_resolved" and e["data"]["decision"] == "expired" for e in events):
        return {**base, "replayable": False, "identical": False, "reason": "expiration : dépend de l'horloge"}
    s = events[0]["data"]
    by_role: dict[str, list[dict[str, Any]]] = {}
    for e in events:
        if e["type"] == "agent_output":
            by_role.setdefault(e["actor"], []).append(e["data"]["output"])
    tmp = Path(tempfile.mkdtemp(prefix="audit-team-replay-"))
    scratch = TeamRuntime(tmp / "journal", rt.tools, rt.policy, rt.crew, hmac_key=b"replay",
                          idempotency=IdempotencyRegistry(tmp / "idempotency.json"))
    svc = MissionService(scratch, "direct", checkpointer=InMemorySaver(),
                         brains=BrainFactory({r: RecordedBrain(list(d)) for r, d in by_role.items()}))
    mi = MissionInput(task=s["task"], dossier_id=s["dossier_id"], recipient=s["recipient"], scopes=list(s["scopes"]),
                      owner=s["owner"], budget=s["budget"]["effective"], faults=s["faults"], doubles=s["doubles"],
                      role_budgets={r: b for r, b in s["role_budgets"].items()}, max_review_rounds=s["max_review_rounds"])
    err = None
    try:
        res = svc.start(mi, mission_id=mid)
        for r in (e["data"] for e in events if e["type"] == "approval_resolved"):
            if res["status"] != "awaiting_approval":
                break
            p = res["pending_approval"]
            res = svc.resolve(mid, p["approval_id"], r["decision"], r["approver"] or "replay", ("audit:approve",),
                              p["args_hash"])
    except (ApprovalError, RuntimeError, ValueError) as e:
        err = f"{type(e).__name__}: {e}"
    a, b = trajectory(events), trajectory(scratch.journal.read(mid))
    diffs = [{"index": i, "recorded": x, "replayed": y} for i, (x, y) in enumerate(zip(a, b)) if x != y]
    if len(a) != len(b):
        diffs.append({"index": min(len(a), len(b)), "recorded_len": len(a), "replayed_len": len(b)})
    return {**base, "replayable": err is None, "identical": err is None and not diffs, "differences": diffs[:10],
            "error": err, "events_compared": len(a)}
