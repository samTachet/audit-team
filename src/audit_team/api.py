"""API HTTP d'audit-team : l'API v1 (`/tasks`, mode mono-agent) + les missions d'équipe.

Lancement : uvicorn audit_team.api:create_app --factory   (un seul worker : imposé au démarrage)

Routes missions (EF-14) :
- POST /missions                                   créer et lancer une mission (jusqu'à la première pause)
- GET  /missions/{id}                              statut, approbation en attente, métriques par rôle
- GET  /missions/{id}/trajectory                   journal complet + vérification de la chaîne
- GET  /missions/{id}/timeline                     frise lisible
- POST /missions/{id}/approvals/{approval_id}      corps : {"decision": "approve"|"reject", "args_hash": "…"}
- POST /missions/{id}/replay                       rejoue la mission dans la vraie gateway (débit limité)
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field

from acsi_agent.agent import ApprovalError, Runtime
from acsi_agent.api import STATUS_BY_APPROVAL_ERROR, Principal
from acsi_agent.api import create_app as create_v1_app
from acsi_agent.ratelimit import RateLimiter
from acsi_agent.team import TeamRuntime
from acsi_agent.types import MISSION_ID_RE

from . import __version__
from .replay import replay_mission
from .service import MissionService, sqlite_checkpointer

Finite = Annotated[float, Field(allow_inf_nan=False)]


class MissionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task: str = Field(min_length=3, max_length=2000)
    dossier_id: str = Field(pattern=r"^D-[0-9]{3}$")
    recipient: str | None = Field(default=None, max_length=254)
    scopes: list[str] | None = None
    budget: dict[str, Finite] | None = None
    role_budgets: dict[str, dict[str, Finite]] | None = None
    max_review_rounds: int | None = Field(default=None, ge=1, le=10)


class MissionApproval(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: str
    args_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


def render_timeline(events: list[dict[str, Any]]) -> str:
    out = []
    for e in events:
        d, t = e["data"], e["type"]
        who = e.get("actor", "?")
        if t == "mission_started":
            line = f"MISSION {e['request_id']} — {d['owner']} : « {d['task']} » ({d['dossier_id']})"
        elif t == "node_entered":
            line = f"  ▸ nœud {d['node']}"
        elif t == "tool_call":
            code = (d["error"] or {}).get("code")
            line = f"    {who:9} {d['tool']} → {d['status']}{f' ({code})' if code else ''}"
        elif t == "agent_output":
            line = f"    {who:9} décision {d['output'].get('kind')}" + (f" {d['output'].get('tool')}" if d['output'].get('tool') else "")
        elif t == "handoff":
            line = f"    relais {d['from']} → {d['to']} ({d['reason']})"
        elif t == "review_verdict":
            line = f"    relecteur : {d['verdict']} (tour {d['round']})"
        elif t == "approval_requested":
            line = f"  ⏸ approbation {d['approval_id']} demandée : {d['tool']} {d['args']} (args_hash {d['args_hash'][:12]}…)"
        elif t == "approval_resolved":
            line = f"  ▶ approbation {d['approval_id']} : {d['decision']} par {d.get('approver')}"
        elif t == "budget_exceeded":
            line = f"  ✖ budget dépassé ({d['scope']}{':' + d['role'] if d.get('role') else ''}, {d['dimension']})"
        elif t == "agent_error":
            line = f"  ! {who} : {d.get('code')}"
        elif t == "mission_finished":
            line = f"FIN {d['status']}{' (' + d['code'] + ')' if d.get('code') else ''} — {d['answer']}"
        else:
            line = f"  {t}"
        out.append(line)
    return "\n".join(out)


def create_app(service: MissionService | None = None, api_keys: dict[str, dict[str, str]] | None = None,
               runtime: Runtime | None = None) -> FastAPI:
    jdir = Path(os.environ.get("ACSI_JOURNAL_DIR", "data/journal"))
    app = create_v1_app(runtime or Runtime(jdir), api_keys)
    app.title, app.version = "audit-team", __version__
    if service is None:
        cp = os.environ.get("AUDIT_TEAM_CHECKPOINTS")
        service = MissionService(TeamRuntime(jdir), os.environ.get("AUDIT_TEAM_RUNNER", "crewai"),
                                 checkpointer=sqlite_checkpointer(Path(cp)) if cp else None)
    app.state.missions = service
    rt = service.rt
    policy, err, principal = app.state.policy, app.state.err, app.state.principal
    approver_scope = policy["approval"]["approver_scope"]
    mission_limiter = RateLimiter(policy["rate_limits"]["tasks_per_minute_per_key"])
    replay_limiter = RateLimiter(2)

    interval = float(os.environ.get("AUDIT_TEAM_SWEEP_INTERVAL_S", "0") or 0)
    if interval > 0:  # EF-18 : balayeur périodique, sans attendre une lecture
        def loop():
            while True:
                rt.sweep()
                app.state.runtime.sweep()
                threading.Event().wait(interval)
        threading.Thread(target=loop, name="sweeper", daemon=True).start()

    def visible(p: Principal, mid: str) -> dict[str, Any]:
        if not MISSION_ID_RE.fullmatch(mid):
            raise err(404, "NOT_FOUND", "mission inconnue")
        try:
            res = rt.public(mid)
        except KeyError:
            raise err(404, "NOT_FOUND", "mission inconnue") from None
        if res["owner"] != p.id and approver_scope not in p.scopes:
            raise err(404, "NOT_FOUND", "mission inconnue")
        return res

    @app.post("/missions", status_code=201)
    def create_mission(body: MissionRequest, p: Principal = Depends(principal)):
        if not mission_limiter.consume(p.id):
            raise err(429, "RATE_LIMITED", "trop de missions par minute pour cette clé")
        default = ["audit:read", "audit:draft"] + (["audit:submit"] if body.recipient else [])
        scopes = body.scopes if body.scopes is not None else default
        if approver_scope in scopes or set(scopes) - set(policy["scopes"]):
            raise err(422, "INVALID_SCOPES", f"portées invalides (`{approver_scope}` n'est pas une portée de mission)")
        if set(scopes) - set(p.scopes):
            raise err(403, "SCOPE_NOT_GRANTED", f"le rôle `{p.role}` ne permet pas : {sorted(set(scopes) - set(p.scopes))}")
        from acsi_agent.team import MissionInput
        mi = MissionInput(task=body.task, dossier_id=body.dossier_id, recipient=body.recipient, scopes=scopes,
                          owner=p.id, budget=body.budget or {}, role_budgets=body.role_budgets or {},
                          max_review_rounds=body.max_review_rounds)
        try:
            return service.start(mi)
        except ValueError as e:
            code = "INVALID_BUDGET" if ("plafond" in str(e) or "budget" in str(e)) else "INVALID_MISSION"
            raise err(422, code, str(e)) from None

    @app.get("/missions/{mission_id}")
    def get_mission(mission_id: str, p: Principal = Depends(principal)):
        return visible(p, mission_id)

    @app.get("/missions/{mission_id}/trajectory")
    def mission_trajectory(mission_id: str, p: Principal = Depends(principal)):
        visible(p, mission_id)
        ok, why = rt.journal.verify(mission_id)
        return {"mission_id": mission_id, "chain_ok": ok, "chain_detail": why, "events": rt.journal.read(mission_id)}

    @app.get("/missions/{mission_id}/timeline", response_class=PlainTextResponse)
    def mission_timeline(mission_id: str, p: Principal = Depends(principal)):
        visible(p, mission_id)
        return render_timeline(rt.journal.read(mission_id))

    @app.post("/missions/{mission_id}/approvals/{approval_id}")
    def approve_mission(mission_id: str, approval_id: str, body: MissionApproval, p: Principal = Depends(principal)):
        visible(p, mission_id)
        try:
            return service.resolve(mission_id, approval_id, body.decision, p.id, p.scopes, body.args_hash)
        except ApprovalError as e:
            raise err(STATUS_BY_APPROVAL_ERROR.get(e.code, 400), e.code, e.message) from None

    @app.post("/missions/{mission_id}/replay")
    def replay(mission_id: str, p: Principal = Depends(principal)):
        visible(p, mission_id)
        if not replay_limiter.consume(p.id):
            raise err(429, "RATE_LIMITED", "replay limité à 2 par minute et par clé")
        return replay_mission(rt, mission_id)

    return app
