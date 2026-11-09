"""Graphe LangGraph du cycle de vie d'une mission (P2).

    init → analyse → rédaction → revue ─┬─ approve ─→ proposer ─┬─ needs_approval → approbation (interrupt) → exécution → clôture
                        ↑                │                       └─ deny (0 humain) ───────────────────────────────────→ clôture
                        └ request_changes (tour < max)           block / tours épuisés / budget / erreur ──────────────→ clôture

Règles (cahier des charges, section 7) :
- LangGraph n'appelle AUCUN tool : il ne parle qu'au `TeamRuntime` (ACSI) ;
- chaque nœud vérifie d'abord la cohérence journal ↔ checkpoint (ENF-04) puis le budget ;
- le nœud d'approbation ne contient que `interrupt()` : la décision humaine est validée par ACSI
  (`TeamRuntime.resolve_approval`) AVANT la reprise, et le nœud d'exécution redemande à ACSI si
  l'approbation existe ; un `Command(resume=…)` forgé n'exécute donc rien ;
- l'exécution est un nœud séparé et idempotent (un nœud rejoué ne renvoie pas le mail).
"""
from __future__ import annotations

from typing import Any, Callable, TypedDict

from langgraph.errors import GraphBubbleUp
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from acsi_agent.team import TeamRuntime

from .doubles import Brain
from .models import OUTPUT_MODELS
from .phases import PhaseError, PhaseRunner


class GraphState(TypedDict, total=False):
    mission_id: str
    dossier_id: str
    round: int
    max_rounds: int
    analysis_ref: str
    drafts_ref: str
    verdict: str
    feedback: list[dict[str, Any]]
    proposal_ref: str | None
    approval_id: str | None
    args_hash: str | None
    status: str | None
    code: str | None
    answer: str | None
    last_error: str | None
    journal_seq: int


BrainFor = Callable[[str, dict[str, Any]], Brain]


def _terminal(status: str, code: str | None, answer: str) -> dict[str, Any]:
    return {"status": status, "code": code, "answer": answer}


def build_graph(rt: TeamRuntime, runner: PhaseRunner, brains: BrainFor) -> StateGraph:
    def doubles(mid: str) -> dict[str, Any]:
        return rt.state(mid).started.get("doubles", {})

    def node(name: str, fn: Callable[[GraphState], dict[str, Any]]):
        def inner(state: GraphState, config=None) -> dict[str, Any]:
            mid = state["mission_id"]
            ok, why = rt.consistent(mid, state.get("journal_seq", 0))
            if not ok:
                upd = _terminal("failed", "CHECKPOINT_DIVERGENCE", f"Journal et checkpoint divergent : {why}.")
                return {**upd, "journal_seq": state.get("journal_seq", 0)}
            st = rt.state(mid)
            if st.finished:  # close par ailleurs (expiration, balayeur, redémarrage)
                return {**_terminal(st.finished["status"], st.finished.get("code"), st.finished["answer"]),
                        "journal_seq": rt.journal.head(mid)[0]}
            conf = (config or {}).get("configurable") or {}
            # Le nœud s'exécute depuis le checkpoint parent : son id est dans `checkpoint_map` (clé "").
            cp = conf.get("checkpoint_id") or (conf.get("checkpoint_map") or {}).get("")
            rt.enter_node(mid, name, cp)
            try:
                upd = fn(state) if not rt.stopped(mid) else {}
            except GraphBubbleUp:
                raise
            except PhaseError as e:
                rt.record(mid, "agent_error", {"code": "PLANNER_ERROR", "role": e.role,
                                               "error": f"{type(e.cause).__name__}: {e.cause}"[:300]}, actor=e.role)
                upd = {**_terminal("failed", "PLANNER_ERROR", f"Erreur du rôle `{e.role}` : mission arrêtée sans autre action."),
                       "last_error": str(e)[:300]}
            stop = rt.stopped(mid)
            if stop and upd.get("status") in (None, "completed"):
                who = f"rôle {stop['role']}" if stop["scope"] == "role" else "global"
                upd = _terminal("budget_exceeded", "BUDGET_EXCEEDED",
                                f"Budget dépassé ({who}, {stop['dimension']} : {stop['observed']} > {stop['limit']}). "
                                "L'action proposée n'a pas été exécutée.")
            upd["journal_seq"] = rt.journal.head(mid)[0]
            return upd
        return inner

    def phase(mid: str, role: str, inputs: dict[str, Any]):
        res = runner.run(mid, role, inputs, brains(role, doubles(mid)))
        if res.stopped:
            if res.reason == "denials":
                return None, _terminal("refused", "TOO_MANY_DENIALS",
                                       f"Le rôle `{role}` a cumulé trop de refus de la gateway : mission arrêtée.")
            return None, {}
        try:
            out = OUTPUT_MODELS[role].model_validate(res.output).model_dump()
        except Exception as e:  # noqa: BLE001 — EF-12 : sortie invalide d'un agent
            raise PhaseError(role, f"sortie invalide : {e}") from e
        return out, None

    def handoff(mid: str, src: str, dst: str, ref: str, reason: str) -> None:
        rt.record(mid, "handoff", {"from": src, "to": dst, "artifact": ref, "reason": reason}, actor=src)

    # ------------------------------------------------------------ nœuds
    def n_init(s: GraphState) -> dict[str, Any]:
        st = rt.state(s["mission_id"])
        return {"dossier_id": st.started["dossier_id"], "round": 1, "max_rounds": st.started["max_review_rounds"],
                "feedback": [], "status": None}

    def n_analyse(s: GraphState) -> dict[str, Any]:
        mid = s["mission_id"]
        st = rt.state(mid)
        out, term = phase(mid, "analyst", {"dossier_id": s["dossier_id"], "task": st.started["task"]})
        if term is not None or out is None:
            return term or {}
        if out["error"]:
            return _terminal("failed", out["error"], f"Analyse impossible ({out['error']}).")
        ref = rt.put_artifact(mid, out)
        handoff(mid, "analyst", "writer", ref, "analyse terminée")
        return {"analysis_ref": ref}

    def n_draft(s: GraphState) -> dict[str, Any]:
        mid = s["mission_id"]
        st = rt.state(mid)
        inputs = {"analysis": rt.get_artifact(mid, s["analysis_ref"]), "round": s["round"],
                  "feedback": s.get("feedback", []), "recipient": st.started.get("recipient")}
        out, term = phase(mid, "writer", inputs)
        if term is not None or out is None:
            return term or {}
        if out["error"]:
            return _terminal("failed", out["error"], f"Rédaction impossible ({out['error']}).")
        ref = rt.put_artifact(mid, out)
        handoff(mid, "writer", "reviewer", ref, f"brouillons du tour {s['round']}")
        return {"drafts_ref": ref}

    def n_review(s: GraphState) -> dict[str, Any]:
        mid = s["mission_id"]
        drafts = rt.get_artifact(mid, s["drafts_ref"])
        allowed = sorted({d for r in rt.policy["argument_rules"] for d in r.get("allowed_domains", [])})
        inputs = {"analysis": rt.get_artifact(mid, s["analysis_ref"]), "drafts": drafts, "round": s["round"],
                  "allowed_domains": allowed}
        out, term = phase(mid, "reviewer", inputs)
        if term is not None or out is None:
            return term or {}
        rt.record(mid, "review_verdict", {"round": s["round"], "verdict": out["verdict"], "reasons": out["reasons"],
                                          "drafts": s["drafts_ref"]}, actor="reviewer")
        if out["verdict"] == "block":
            return _terminal("refused", "REVIEW_BLOCKED", "Le relecteur conformité a bloqué : " + "; ".join(out["reasons"]))
        if out["verdict"] == "request_changes":
            if s["round"] >= s["max_rounds"]:
                return _terminal("failed", "REVIEW_ROUNDS_EXCEEDED",
                                 f"Désaccord persistant après {s['round']} tour(s) de revue : rien n'est proposé.")
            handoff(mid, "reviewer", "writer", s["drafts_ref"], "corrections demandées")
            return {"verdict": "request_changes", "round": s["round"] + 1, "feedback": out["feedback"]}
        prop = drafts.get("proposed_action")
        if not prop:
            n = len(drafts["drafts"])
            return _terminal("completed", None, f"{n} brouillon(s) rédigé(s) et validé(s) par le relecteur ; aucun envoi demandé.")
        return {"verdict": "approve", "proposal_ref": rt.put_artifact(mid, prop)}

    def n_propose(s: GraphState) -> dict[str, Any]:
        mid = s["mission_id"]
        prop = rt.get_artifact(mid, s["proposal_ref"])
        t = rt.request_approval(mid, prop["proposed_by"], {"tool": prop["tool"], "args": prop["args"]})
        if t["status"] == "needs_approval":
            return {"approval_id": t["approval_id"], "args_hash": t["args_hash"]}
        code = t.get("code") or "DENIED"
        return _terminal("refused", code, f"Action `{prop['tool']}` refusée par la gateway ACSI ({code}) : aucun humain sollicité.")

    def n_approval(s: GraphState) -> dict[str, Any]:
        # Rien d'autre que la pause : LangGraph relance ce nœud depuis son début à la reprise.
        resp = interrupt({"mission_id": s["mission_id"], "approval_id": s["approval_id"], "args_hash": s["args_hash"]})
        return {"last_error": None if isinstance(resp, dict) else "réponse d'approbation invalide"}

    def n_execute(s: GraphState) -> dict[str, Any]:
        mid, apr = s["mission_id"], s["approval_id"]
        decision = rt.approval_decision(mid, apr)  # ACSI, pas la valeur de reprise
        if decision == "reject":
            return _terminal("refused", "APPROVAL_REJECTED", "Transmission refusée par l'approbateur : rien n'a été envoyé.")
        if decision != "approve":
            return _terminal("refused", "NOT_APPROVED", "Aucune approbation validée par ACSI : rien n'a été exécuté.")
        out = rt.execute_approved(mid, apr)
        if out["status"] == "ok":
            r = out["result"]
            return _terminal("completed", None, f"Dossier {r['dossier_id']} transmis à {r['recipient']} "
                                                f"({r['drafts_submitted']} brouillon(s)), réf. {r['submission_id']}.")
        status = "budget_exceeded" if out["code"] == "BUDGET_EXCEEDED" else ("refused" if out["status"] == "denied" else "failed")
        return _terminal(status, out["code"], f"Exécution non aboutie ({out['code']}).")

    def n_close(s: GraphState) -> dict[str, Any]:
        mid = s["mission_id"]
        if not rt.state(mid).finished:
            rt.finish(mid, s.get("status") or "failed", s.get("answer") or "", s.get("code"))
        return {}

    # ------------------------------------------------------------ arêtes
    def after(nxt: str | Callable[[GraphState], str]):
        def route(s: GraphState) -> str:
            if s.get("status"):
                return "close"
            return nxt(s) if callable(nxt) else nxt
        return route

    g = StateGraph(GraphState)
    for name, fn in (("init", n_init), ("analyse", n_analyse), ("redaction", n_draft), ("revue", n_review),
                     ("proposer", n_propose), ("approbation", n_approval), ("execution", n_execute)):
        g.add_node(name, node(name, fn))
    g.add_node("cloture", n_close)
    g.add_edge(START, "init")
    edges = {
        "init": after("analyse"), "analyse": after("redaction"), "redaction": after("revue"),
        "revue": after(lambda s: "redaction" if s.get("verdict") == "request_changes" else "proposer"),
        "proposer": after("approbation"), "approbation": after("execution"), "execution": after("close"),
    }
    targets = {"analyse": "analyse", "redaction": "redaction", "revue": "revue", "proposer": "proposer",
               "approbation": "approbation", "execution": "execution", "close": "cloture"}
    for src, route in edges.items():
        g.add_conditional_edges(src, route, targets)
    g.add_edge("cloture", END)
    return g
