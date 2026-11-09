"""Noyau ACSI multi-acteurs (P1) : missions d'équipe, sans aucune dépendance à CrewAI ni LangGraph.

Ce module est la SEULE porte de l'équipe vers les tools. Les frameworks (paquet `audit_team`) n'appellent
que les méthodes publiques ci-dessous ; ils n'importent ni `backends` ni `tools_impl` (test d'architecture).

Contrats (cahier des charges, section 7) :
- `propose(mission_id, actor, tool, args) -> ToolOutcome` : gateway, rôle, budget, journal. Seul point d'entrée
  des tools pour un agent. Les tools irréversibles (`graph_only_tools`) y sont TOUJOURS refusés ;
- `check_llm` / `charge_llm(mission_id, actor, usage)` : imputation de chaque appel LLM au couple (mission, rôle) ;
- `request_approval(mission_id, actor, ProposedAction) -> ApprovalTicket | ToolOutcome` : évalue d'abord
  (un refus ne sollicite jamais d'humain) et exige un `review_verdict` « approve » préalable (O5) ;
- `resolve_approval(..., args_hash_seen)` : portée, approbateur distinct, TTL, usage unique, `args_hash`
  identique à celui gardé en mémoire et à celui relu par l'approbateur ;
- `execute_approved(mission_id, approval_id)` : idempotent (un nœud rejoué ne réexécute pas).

L'état d'une mission est reconstruit depuis son journal (source de vérité) ; `Live` garde en mémoire les
services simulés, la gateway et les arguments approuvables. Mono-processus (ENF-05).
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .agent import ApprovalError, args_hash, default_state_dir, hmac_key_from_env
from .backends import Services, fixtures_hash
from .budget import Budget, Meter, Violation, check_post_planner, check_pre_planner, check_pre_tool, ratios
from .gateway import Decision, Gateway
from .idempotency import IdempotencyRegistry
from .journal import Journal, JournalSealed, JournalTampered
from .ratelimit import RateLimiter
from .specs import SPECS_DIR, ToolSpec, load_policy, load_tools
from .tools_impl import build_impls
from .types import MISSION_ID_RE, CallContext, new_mission_id

TERMINAL = {"completed", "refused", "failed", "budget_exceeded", "expired"}
TEAM_EVENTS = {"mission_started", "node_entered", "agent_output", "agent_error", "handoff", "review_verdict",
               "tool_call", "approval_requested", "approval_resolved", "budget_warning", "budget_exceeded",
               "mission_finished"}
DOSSIER_RE = re.compile(r"^D-[0-9]{3}$")
HUMAN = "human"
RUNTIME = "acsi"


def load_crew(path: Path | None = None) -> dict[str, Any]:
    return json.loads((path or SPECS_DIR / "crew.json").read_text(encoding="utf-8"))


def artifact_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@dataclass
class MissionInput:
    task: str
    dossier_id: str
    recipient: str | None = None
    scopes: list[str] = field(default_factory=lambda: ["audit:read", "audit:draft"])
    owner: str = "anonymous"
    budget: dict[str, Any] = field(default_factory=dict)
    role_budgets: dict[str, dict[str, Any]] = field(default_factory=dict)
    faults: dict[str, Any] = field(default_factory=dict)
    doubles: dict[str, Any] = field(default_factory=dict)  # réglages des doubles déterministes (rejouables)
    max_review_rounds: int | None = None


@dataclass
class RoleMeter:
    llm_calls: int = 0
    tool_calls: int = 0
    tokens: int = 0
    cost: float = 0.0
    active_ms: float = 0.0
    denied: int = 0
    consecutive_denied: int = 0

    def meter(self) -> Meter:
        return Meter(self.llm_calls, self.tool_calls, self.tokens, self.cost, self.active_ms / 1000)


@dataclass
class MissionState:
    mission_id: str
    started: dict[str, Any] = field(default_factory=dict)
    budget: Budget | None = None
    role_budgets: dict[str, Budget] = field(default_factory=dict)
    roles: dict[str, RoleMeter] = field(default_factory=dict)
    total: RoleMeter = field(default_factory=RoleMeter)
    call_counts: dict[str, int] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    pending: dict[str, Any] | None = None
    pending_ts: float = 0.0
    resolutions: dict[str, str] = field(default_factory=dict)
    executed: dict[str, dict[str, Any]] = field(default_factory=dict)
    verdicts: list[dict[str, Any]] = field(default_factory=list)
    warned: set[str] = field(default_factory=set)
    stop: dict[str, Any] | None = None
    finished: dict[str, Any] | None = None
    human_wait_s: float = 0.0
    seq: int = 0

    def role(self, actor: str) -> RoleMeter:
        return self.roles.setdefault(actor, RoleMeter())

    def apply(self, ev: dict[str, Any]) -> None:
        t, d, actor = ev["type"], ev["data"], ev.get("actor", RUNTIME)
        self.seq = ev["seq"]
        if t == "mission_started":
            self.started = d
            self.budget = Budget(**d["budget"]["effective"])
            self.role_budgets = {r: Budget(**b) for r, b in d["role_budgets"].items()}
        elif t == "agent_output":
            u = d["usage"]
            for m in (self.role(actor), self.total):
                m.llm_calls += 1
                m.tokens += u["tokens_in"] + u["tokens_out"]
                m.cost += u["cost_usd"]
                m.active_ms += d["latency_ms"]
        elif t == "tool_call":
            self.tool_calls.append({**d, "actor": actor})
            if d["executed"]:
                for m in (self.role(actor), self.total):
                    m.tool_calls += 1
                    m.active_ms += d["latency_ms"]
            if d["gateway"]["verdict"] in {"allow", "needs_approval"}:
                self.call_counts[d["tool"]] = self.call_counts.get(d["tool"], 0) + 1
            if d["status"] == "denied":
                self.role(actor).denied += 1
                self.role(actor).consecutive_denied += 1
            elif d["status"] == "ok":
                self.role(actor).consecutive_denied = 0
            if d.get("approval_id") and d["executed"]:
                self.executed[d["approval_id"]] = d
        elif t == "approval_requested":
            self.pending, self.pending_ts = d, ev["ts"]
        elif t == "approval_resolved":
            self.resolutions[d["approval_id"]] = d["decision"]
            self.human_wait_s += max(0.0, ev["ts"] - self.pending_ts)
            self.pending = None
        elif t == "review_verdict":
            self.verdicts.append(d)
        elif t == "budget_warning":
            self.warned.add(f"{d['scope']}:{d['dimension']}")
        elif t == "budget_exceeded":
            self.stop = d
        elif t == "mission_finished":
            self.finished = d

    def metrics(self) -> dict[str, Any]:
        def m(r: RoleMeter) -> dict[str, Any]:
            return {"llm_calls": r.llm_calls, "tool_calls": r.tool_calls, "tokens": r.tokens,
                    "cost_usd": round(r.cost, 6), "active_ms": round(r.active_ms, 2), "denied": r.denied}
        return {"global": {**m(self.total), "review_rounds": len(self.verdicts),
                           "human_wait_s": round(self.human_wait_s, 3)},
                "by_role": {k: m(v) for k, v in sorted(self.roles.items())}}


@dataclass
class MissionLive:
    svc: Services
    gateway: Gateway
    approvals: dict[str, tuple[str, dict[str, Any]]] = field(default_factory=dict)  # en attente (I2)
    approved: dict[str, dict[str, Any]] = field(default_factory=dict)  # approuvées, pas encore exécutées
    artifacts: dict[str, Any] = field(default_factory=dict)
    node: str | None = None
    checkpoint_id: str | None = None


class TeamRuntime:
    def __init__(self, journal_dir: Path, tools: dict[str, ToolSpec] | None = None, policy: dict[str, Any] | None = None,
                 crew: dict[str, Any] | None = None, clock: Callable[[], float] = time.time,
                 limiter_clock: Callable[[], float] = time.monotonic, hmac_key: bytes | None = None,
                 anchor_dir: Path | None = None, idempotency: IdempotencyRegistry | None = None,
                 timeout_overrides: dict[str, float] | None = None):
        self.journal = Journal(journal_dir, anchor_dir=anchor_dir,
                               hmac_key=hmac_key if hmac_key is not None else hmac_key_from_env())
        self.tools = tools or load_tools()
        self.policy = policy or load_policy()
        self.crew = crew or load_crew()
        self.roles = {r["id"]: r for r in self.crew["roles"]}
        self.clock = clock
        self.timeout_overrides = timeout_overrides
        self.idempotency = idempotency or IdempotencyRegistry(default_state_dir(journal_dir) / "idempotency.json")
        self.pricing = self.policy["pricing_placeholder"]
        self.limiter = RateLimiter(self.policy["rate_limits"]["tool_calls_per_minute_per_key"], limiter_clock)
        self.graph_only = set(self.policy["mission"]["graph_only_tools"])
        self._live: dict[str, MissionLive] = {}
        self._locks: dict[str, threading.RLock] = {}
        self._guard = threading.Lock()

    # ------------------------------------------------------------ outils internes
    def lock(self, mid: str) -> threading.RLock:
        with self._guard:
            return self._locks.setdefault(mid, threading.RLock())

    def _emit(self, st: MissionState, type_: str, data: dict[str, Any], actor: str = RUNTIME) -> dict[str, Any]:
        assert type_ in TEAM_EVENTS, type_
        live = self._live.get(st.mission_id)
        meta = {"actor": actor, "node": live.node if live else None, "checkpoint_id": live.checkpoint_id if live else None}
        ev = self.journal.append(st.mission_id, type_, data, ts=self.clock(), meta=meta)
        st.apply(ev)
        return ev

    def state(self, mid: str) -> MissionState:
        if not MISSION_ID_RE.fullmatch(mid):
            raise KeyError(mid)
        events = self.journal.read(mid)
        if not events:
            raise KeyError(mid)
        st = MissionState(mid)
        for e in events:
            st.apply(e)
        return st

    def cost(self, tokens_in: int, tokens_out: int) -> float:
        return (tokens_in * self.pricing["in_usd_per_mtok"] + tokens_out * self.pricing["out_usd_per_mtok"]) / 1_000_000

    def is_live(self, mid: str) -> bool:
        return mid in self._live

    # ------------------------------------------------------------ ouverture
    def open_mission(self, mi: MissionInput, mission_id: str | None = None) -> str:
        mid = mission_id or new_mission_id()
        if not MISSION_ID_RE.fullmatch(mid):
            raise ValueError("mission_id invalide")
        if not DOSSIER_RE.fullmatch(mi.dossier_id):
            raise ValueError("dossier_id invalide (D-xxx attendu)")
        mpol = self.policy["mission"]
        budget, clamped = Budget.effective(mpol["budget_default"], mi.budget)
        role_budgets: dict[str, dict[str, Any]] = {}
        for rid, role in self.roles.items():
            over = dict(mi.role_budgets.get(rid, {}))
            rb, rclamped = Budget.effective(role["budget"], over)
            vals = rb.as_dict()
            for k, gv in budget.as_dict().items():  # un rôle ne dépasse jamais le global
                if vals[k] > gv:
                    if k in over:
                        raise ValueError(f"budget du rôle {rid} supérieur au budget global : {k}")
                    vals[k] = gv
            role_budgets[rid] = vals
            clamped += [f"{rid}.{k}" for k in rclamped]
        if set(mi.role_budgets) - set(self.roles):
            raise ValueError(f"rôles inconnus : {sorted(set(mi.role_budgets) - set(self.roles))}")
        rounds = mi.max_review_rounds or mpol["max_review_rounds"]
        if not (1 <= int(rounds) <= mpol["max_review_rounds"]):
            raise ValueError("max_review_rounds hors bornes")
        svc = Services.from_fixtures(mi.faults)
        gw = Gateway(self.tools, self.policy, build_impls(svc, self.idempotency), self.timeout_overrides, self.limiter)
        with self.lock(mid):
            if self.journal.read(mid):
                raise ValueError("mission_id déjà utilisé")
            self._live[mid] = MissionLive(svc, gw, node="init")
            st = MissionState(mid)
            self._emit(st, "mission_started", {
                "task": mi.task, "dossier_id": mi.dossier_id, "recipient": mi.recipient, "scopes": list(mi.scopes),
                "owner": mi.owner, "faults": mi.faults, "doubles": mi.doubles, "max_review_rounds": int(rounds),
                "budget": {"effective": budget.as_dict(), "clamped_to_default": clamped}, "role_budgets": role_budgets,
                "crew_version": self.crew["crew_version"], "fixtures_hash": fixtures_hash(),
            }, actor=mi.owner)
        return mid

    # ------------------------------------------------------------ nœuds et relais
    def enter_node(self, mid: str, node: str, checkpoint_id: str | None = None) -> None:
        with self.lock(mid):
            live = self._live.get(mid)
            st = self.state(mid)
            if st.finished:
                return
            if live is not None:
                live.node, live.checkpoint_id = node, checkpoint_id
            self._emit(st, "node_entered", {"node": node})

    def record(self, mid: str, type_: str, data: dict[str, Any], actor: str) -> None:
        if type_ not in {"handoff", "review_verdict", "agent_error"}:
            raise ValueError(f"type d'événement non autorisé : {type_}")
        with self.lock(mid):
            st = self.state(mid)
            if not st.finished:
                self._emit(st, type_, data, actor=actor)

    def put_artifact(self, mid: str, obj: Any) -> str:
        h = artifact_hash(obj)
        live = self._live.get(mid)
        if live is not None:
            live.artifacts[h] = json.loads(json.dumps(obj))
        return h

    def get_artifact(self, mid: str, ref: str) -> Any:
        live = self._live.get(mid)
        if live is None or ref not in live.artifacts:
            raise KeyError(ref)
        return json.loads(json.dumps(live.artifacts[ref]))

    # ------------------------------------------------------------ budget
    def _check(self, st: MissionState, actor: str, fn) -> Violation | None:
        rb = st.role_budgets.get(actor)
        if rb is not None:
            v = fn(rb, st.role(actor).meter())
            if v:
                return Violation(f"role:{actor}:{v.dimension}", v.limit, v.observed)
        assert st.budget is not None
        v = fn(st.budget, st.total.meter())
        return Violation(f"global:{v.dimension}", v.limit, v.observed) if v else None

    def _stop(self, st: MissionState, v: Violation, actor: str, proposed: dict[str, Any] | None) -> None:
        scope, *rest = v.dimension.split(":")
        role, dim = (rest[0], rest[1]) if scope == "role" else (None, rest[0])
        self._emit(st, "budget_exceeded", {"scope": scope, "role": role, "dimension": dim, "limit": v.limit,
                                           "observed": v.observed, "proposed_action_not_executed": proposed}, actor=actor)

    def stopped(self, mid: str) -> dict[str, Any] | None:
        return self.state(mid).stop

    def _warn(self, st: MissionState, actor: str) -> None:
        wr = self.policy["budget_on_exceed"]["warn_ratio"]
        pairs = [("global", st.budget, st.total)]
        if actor in st.role_budgets:
            pairs.append((f"role:{actor}", st.role_budgets[actor], st.role(actor)))
        for scope, b, m in pairs:
            for dim, r in ratios(b, m.meter()).items():  # type: ignore[arg-type]
                if r >= wr and f"{scope}:{dim}" not in st.warned:
                    self._emit(st, "budget_warning", {"scope": scope, "dimension": dim, "ratio": round(r, 3)}, actor=actor)

    def check_llm(self, mid: str, actor: str) -> Violation | None:
        """Avant chaque appel LLM d'un agent : plafond d'appels (steps) et temps, du rôle puis du global."""
        with self.lock(mid):
            st = self.state(mid)
            if st.stop or st.finished:
                return Violation("stopped", 0, 0)
            v = self._check(st, actor, check_pre_planner)
            if v:
                self._stop(st, v, actor, None)
            return v

    def charge_llm(self, mid: str, actor: str, usage: dict[str, int], latency_ms: float,
                   summary: dict[str, Any]) -> Violation | None:
        """Après chaque appel LLM : imputation (mission, rôle), puis plafonds de tokens et de coût."""
        with self.lock(mid):
            st = self.state(mid)
            if st.finished:
                return Violation("stopped", 0, 0)
            u = {"tokens_in": int(usage["tokens_in"]), "tokens_out": int(usage["tokens_out"])}
            u["cost_usd"] = self.cost(u["tokens_in"], u["tokens_out"])
            self._emit(st, "agent_output", {"usage": u, "latency_ms": latency_ms, "output": summary}, actor=actor)
            if st.stop:
                return Violation("stopped", 0, 0)
            v = self._check(st, actor, check_post_planner)
            if v:
                self._stop(st, v, actor, summary)
                return v
            self._warn(st, actor)
            return None

    # ------------------------------------------------------------ tools (seule porte de la crew)
    def _record(self, st: MissionState, actor: str, tool: str, args: dict[str, Any], status: str, dec: Decision, *,
                result: dict[str, Any] | None = None, error: dict[str, Any] | None = None, latency: float = 0.0,
                executed: bool = False, approval_id: str | None = None) -> dict[str, Any]:
        self._emit(st, "tool_call", {
            "step": len(st.tool_calls) + 1, "tool": tool, "args": args, "status": status,
            "gateway": {"verdict": dec.verdict, "code": dec.code, "reason": dec.reason, "checks": dec.checks},
            "result": result, "error": error, "latency_ms": latency, "approval_id": approval_id, "executed": executed,
        }, actor=actor)
        return {"status": "ok" if status == "ok" else ("denied" if status in {"denied", "rejected"} else "error"),
                "code": (error or {}).get("code"), "result": result, "message": (error or {}).get("message")}

    def _ctx(self, st: MissionState, scopes: tuple[str, ...], idem: str | None = None) -> CallContext:
        return CallContext(request_id=st.mission_id, step=len(st.tool_calls) + 1, scopes=scopes,
                           key_id=st.started["owner"], idem_key=idem)

    def role_scopes(self, st: MissionState, actor: str) -> tuple[str, ...]:
        role = self.roles.get(actor)
        if role is None:
            return ()
        return tuple(s for s in st.started["scopes"] if s in role["scopes"])  # mission ∩ rôle

    def propose(self, mid: str, actor: str, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        """ToolOutcome `{status: ok|denied|error, code, result, message}` : un refus est structuré et définitif."""
        with self.lock(mid):
            st = self.state(mid)
            live = self._live.get(mid)
            if st.finished or live is None:
                return {"status": "denied", "code": "MISSION_CLOSED", "result": None, "message": "mission close"}
            if st.stop:
                return {"status": "denied", "code": "BUDGET_EXCEEDED", "result": None, "message": "budget dépassé"}
            gw = live.gateway
            args = args if isinstance(args, dict) else {}
            scopes = self.role_scopes(st, actor)
            if tool in gw.forbidden or tool not in self.tools:
                dec = gw.evaluate(self._ctx(st, scopes), tool, args, call_counts=st.call_counts)
            elif actor not in self.roles or tool not in self.roles[actor]["tools"] or tool in self.graph_only:
                why = ("réservé au nœud d'exécution du graphe, après revue et approbation humaine"
                       if tool in self.graph_only else f"`{tool}` n'est pas un tool du rôle `{actor}`")
                dec = Decision("deny", "PERMISSION_DENIED", why, ["allowlist:pass", "role:fail"])
            else:
                idem = f"{mid}:{args.get('nc_id')}" if tool == "draft_finding" else None
                dec = gw.evaluate(self._ctx(st, scopes, idem), tool, args, call_counts=st.call_counts)
            if dec.verdict != "allow":
                if dec.verdict == "needs_approval":  # ne peut arriver qu'avec une spec modifiée : refus par défaut
                    dec = Decision("deny", "PERMISSION_DENIED", "action irréversible hors du nœud d'exécution", dec.checks)
                out = self._record(st, actor, tool, args, gw.status_for(dec.code or ""), dec,
                                   error={"code": dec.code, "message": dec.reason})
                lim = self.policy["mission"]["max_consecutive_denials"]
                if st.role(actor).consecutive_denied >= lim:
                    self._emit(st, "agent_error", {"code": "TOO_MANY_DENIALS", "limit": lim}, actor=actor)
                return out
            v = self._check(st, actor, check_pre_tool)
            if v:
                self._stop(st, v, actor, {"tool": tool, "args": args})
                return {"status": "denied", "code": "BUDGET_EXCEEDED", "result": None, "message": v.dimension}
            idem = f"{mid}:{args.get('nc_id')}" if tool == "draft_finding" else None
            res = gw.execute(self._ctx(st, scopes, idem), tool, args)
            out = self._record(st, actor, tool, args, "ok" if res.ok else "error", dec, result=res.result,
                               error=res.error, latency=res.latency_ms, executed=True)
            self._warn(st, actor)
            return out

    def denial_limit_reached(self, mid: str, actor: str) -> bool:
        return self.state(mid).role(actor).consecutive_denied >= self.policy["mission"]["max_consecutive_denials"]

    # ------------------------------------------------------------ approbation
    def request_approval(self, mid: str, actor: str, proposed: dict[str, Any]) -> dict[str, Any]:
        """Renvoie un ticket `{status: needs_approval, approval_id, args_hash, expires_at, ...}` ou un refus."""
        with self.lock(mid):
            st = self.state(mid)
            live = self._live.get(mid)
            if st.finished or live is None:
                return {"status": "denied", "code": "MISSION_CLOSED"}
            tool, args = proposed.get("tool"), proposed.get("args")
            if not isinstance(tool, str) or not isinstance(args, dict):
                return {"status": "error", "code": "INVALID_PROPOSAL"}
            if not st.verdicts or st.verdicts[-1]["verdict"] != "approve":  # O5 : relecture avant approbation
                dec = Decision("deny", "REVIEW_REQUIRED", "aucun verdict « approve » du relecteur", ["review:fail"])
                return self._record(st, actor, tool, args, "denied", dec, error={"code": dec.code, "message": dec.reason})
            if tool not in self.graph_only:
                dec = Decision("deny", "NOT_A_GRAPH_ACTION", "seules les actions irréversibles passent par l'approbation", [])
                return self._record(st, actor, tool, args, "denied", dec, error={"code": dec.code, "message": dec.reason})
            ctx = self._ctx(st, tuple(st.started["scopes"]))
            dec = live.gateway.evaluate(ctx, tool, args, call_counts=st.call_counts)
            if dec.verdict == "deny":  # un refus ne sollicite jamais d'humain
                return self._record(st, actor, tool, args, live.gateway.status_for(dec.code or ""), dec,
                                    error={"code": dec.code, "message": dec.reason})
            if dec.verdict != "needs_approval":
                return {"status": "error", "code": "UNEXPECTED_VERDICT"}
            h = args_hash(args)
            apr = "apr_" + hashlib.sha1(f"{mid}:{len(st.tool_calls) + 1}:{h}".encode()).hexdigest()[:12]
            live.approvals[apr] = (h, json.loads(json.dumps(args)))
            ttl = self.policy["approval"]["ttl_s"]
            data = {"approval_id": apr, "tool": tool, "args": args, "args_hash": h, "requested_by": st.started["owner"],
                    "proposed_by": actor, "expires_at": self.clock() + ttl,
                    "why": next((a["why"] for a in self.policy["approval_required"] if a["tool"] == tool), "")}
            self._emit(st, "approval_requested", data, actor=actor)
            return {"status": "needs_approval", **data}

    def resolve_approval(self, mid: str, approval_id: str, decision: str, approver_id: str,
                         approver_scopes: tuple[str, ...] | list[str], args_hash_seen: str | None) -> str:
        """Valide la décision humaine (ACSI) AVANT toute reprise du graphe. Renvoie la décision enregistrée."""
        if decision not in {"approve", "reject"}:
            raise ApprovalError("INVALID_DECISION", "decision doit valoir approve ou reject")
        pol = self.policy["approval"]
        with self.lock(mid):
            st = self.state(mid)
            if st.finished or not st.pending or st.pending["approval_id"] != approval_id:
                raise ApprovalError("NOT_PENDING", "aucune approbation en attente avec cet identifiant")
            if pol["approver_scope"] not in approver_scopes:
                raise ApprovalError("FORBIDDEN_SCOPE", f"la portée `{pol['approver_scope']}` est requise pour approuver")
            if pol["require_distinct_approver"] and approver_id == st.started["owner"]:
                raise ApprovalError("SELF_APPROVAL", "l'approbateur doit être distinct du demandeur")
            if decision == "approve" and args_hash_seen is None:
                raise ApprovalError("ARGS_HASH_REQUIRED", "renvoyer le `args_hash` affiché pour approuver")
            ok, why = self.journal.verify(mid)
            if not ok:
                raise ApprovalError("JOURNAL_TAMPERED", f"journal altéré ({why}) : approbation refusée")
            if self.clock() > st.pending["expires_at"]:
                self._expire(st)
                raise ApprovalError("EXPIRED", "l'approbation a expiré")
            pend = dict(st.pending)
            if args_hash_seen is not None and args_hash_seen != pend["args_hash"]:
                raise ApprovalError("ARGS_MISMATCH", "le hachage relu par l'approbateur ne correspond pas à la demande")
            live = self._live.get(mid)
            if live is None:
                self._emit(st, "approval_resolved", {"approval_id": approval_id, "decision": decision, "approver": approver_id},
                           actor=HUMAN)
                self.finish(mid, "failed", "Le runtime a redémarré pendant l'attente : rien n'a été exécuté.",
                            code="RUNTIME_RESTARTED")
                return "restarted"
            mem = live.approvals.get(approval_id)
            if (mem is None or mem[0] != pend["args_hash"] or args_hash(pend["args"]) != mem[0]
                    or args_hash(mem[1]) != mem[0]):
                raise ApprovalError("ARGS_MISMATCH", "arguments du journal différents de ceux gardés en mémoire")
            live.approvals.pop(approval_id)
            self._emit(st, "approval_resolved", {"approval_id": approval_id, "decision": decision, "approver": approver_id,
                                                 "args_hash": pend["args_hash"]}, actor=HUMAN)
            if decision == "approve":
                live.approved[approval_id] = {"tool": pend["tool"], "args": mem[1]}
            else:
                self._record(st, HUMAN, pend["tool"], pend["args"], "rejected",
                             Decision("needs_approval", None, "approbation refusée", ["approval:rejected"]),
                             error={"code": "APPROVAL_REJECTED", "message": f"refusé par {approver_id}"},
                             approval_id=approval_id)
            return decision

    def approval_decision(self, mid: str, approval_id: str) -> str | None:
        return self.state(mid).resolutions.get(approval_id)

    def execute_approved(self, mid: str, approval_id: str) -> dict[str, Any]:
        """Exécute une action approuvée. Idempotent : un nœud rejoué renvoie l'issue déjà enregistrée."""
        with self.lock(mid):
            st = self.state(mid)
            if approval_id in st.executed:
                d = st.executed[approval_id]
                return {"status": "ok" if d["status"] == "ok" else "error", "code": (d["error"] or {}).get("code"),
                        "result": d["result"], "message": (d["error"] or {}).get("message"), "replayed": True}
            if st.resolutions.get(approval_id) != "approve":
                return {"status": "denied", "code": "NOT_APPROVED", "result": None, "message": "aucune approbation ACSI"}
            live = self._live.get(mid)
            if live is None or approval_id not in live.approved:
                return {"status": "error", "code": "RUNTIME_RESTARTED", "result": None, "message": "état perdu"}
            act = live.approved.pop(approval_id)
            tool, args = act["tool"], act["args"]
            ctx = self._ctx(st, tuple(st.started["scopes"]))
            dec = live.gateway.evaluate(ctx, tool, args, call_counts=st.call_counts, approved=True)
            if dec.verdict != "allow":  # la politique a pu changer pendant l'attente
                return self._record(st, RUNTIME, tool, args, live.gateway.status_for(dec.code or ""), dec,
                                    error={"code": dec.code, "message": dec.reason}, approval_id=approval_id)
            assert st.budget is not None
            v = check_pre_tool(st.budget, st.total.meter())
            if v:
                self._stop(st, Violation(f"global:{v.dimension}", v.limit, v.observed), RUNTIME, {"tool": tool, "args": args})
                return {"status": "denied", "code": "BUDGET_EXCEEDED", "result": None, "message": v.dimension}
            res = live.gateway.execute(ctx, tool, args)
            return self._record(st, RUNTIME, tool, args, "ok" if res.ok else "error", dec, result=res.result,
                                error=res.error, latency=res.latency_ms, executed=True, approval_id=approval_id)

    # ------------------------------------------------------------ fin, balayage, cohérence
    def _expire(self, st: MissionState) -> None:
        p = st.pending or {}
        self._emit(st, "approval_resolved", {"approval_id": p.get("approval_id"), "decision": "expired", "approver": None},
                   actor=RUNTIME)
        self._finish(st, "expired", f"Approbation de `{p.get('tool')}` expirée : action non exécutée.", "APPROVAL_EXPIRED")

    def _finish(self, st: MissionState, status: str, answer: str, code: str | None) -> None:
        live = self._live.pop(st.mission_id, None)
        effects = live.svc.effects() if live else {}
        self._emit(st, "mission_finished", {"status": status, "code": code, "answer": answer, "effects": effects,
                                            "metrics": st.metrics()}, actor=RUNTIME)
        self.journal.seal(st.mission_id)
        with self._guard:
            self._locks.pop(st.mission_id, None)

    def finish(self, mid: str, status: str, answer: str, code: str | None = None) -> None:
        assert status in TERMINAL, status
        with self.lock(mid):
            st = self.state(mid)
            if not st.finished:
                try:
                    self._finish(st, status, answer, code)
                except (JournalTampered, JournalSealed):
                    # Journal altéré : on n'écrit rien (ce serait masquer la preuve), on libère la mission.
                    self._live.pop(mid, None)

    def refresh(self, mid: str) -> MissionState:
        with self.lock(mid):
            st = self.state(mid)
            if st.pending and not st.finished and self.clock() > st.pending["expires_at"]:
                self._expire(st)
            return st

    def sweep(self) -> list[str]:
        """Balayeur (EF-18) : approbations échues et missions orphelines (perdues au redémarrage)."""
        closed = []
        for mid in self.journal.list_ids("mis_"):
            anc = self.journal.anchor(mid)
            if anc and anc.get("sealed"):
                continue
            with self.lock(mid):
                st = self.state(mid)
                if st.finished:
                    continue
                if st.pending and self.clock() > st.pending["expires_at"]:
                    self._expire(st)
                    closed.append(mid)
                elif mid not in self._live:
                    self._finish(st, "failed", "Le runtime a redémarré : mission orpheline clôturée.", "RUNTIME_RESTARTED")
                    closed.append(mid)
        return closed

    def consistent(self, mid: str, checkpoint_seq: int) -> tuple[bool, str]:
        """ENF-04 : le checkpoint LangGraph est un état dérivé. Le journal doit être intègre et au moins aussi
        avancé que ce que le checkpoint a vu ; sinon la reprise est refusée (CHECKPOINT_DIVERGENCE)."""
        ok, why = self.journal.verify(mid)
        if not ok:
            return False, why
        seq, _ = self.journal.head(mid)
        if seq < checkpoint_seq:
            return False, f"journal ({seq}) en retard sur le checkpoint ({checkpoint_seq})"
        return True, "ok"

    def effects(self, mid: str) -> dict[str, int]:
        live = self._live.get(mid)
        if live:
            return live.svc.effects()
        st = self.state(mid)
        return (st.finished or {}).get("effects", {})

    def public(self, mid: str) -> dict[str, Any]:
        st = self.refresh(mid)
        if st.finished:
            status, code, answer = st.finished["status"], st.finished.get("code"), st.finished["answer"]
        else:
            status, code, answer = ("awaiting_approval" if st.pending else "running"), None, ""
        pending = None
        if st.pending and not st.finished:
            p = st.pending
            pending = {k: p[k] for k in ("approval_id", "tool", "args", "args_hash", "expires_at", "requested_by", "proposed_by")}
        return {"mission_id": mid, "status": status, "code": code, "answer": answer, "owner": st.started.get("owner"),
                "dossier_id": st.started.get("dossier_id"), "pending_approval": pending, "effects": self.effects(mid),
                "review_verdicts": [v["verdict"] for v in st.verdicts], "metrics": st.metrics(),
                "budget_exceeded": st.stop,
                "tool_calls": [{"step": c["step"], "actor": c["actor"], "tool": c["tool"], "status": c["status"],
                                "code": (c["error"] or {}).get("code")} for c in st.tool_calls]}
