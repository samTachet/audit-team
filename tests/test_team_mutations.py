"""Les scénarios d'équipe ont des dents (cahier des charges, section 9) : chaque mutation d'une garantie
fait échouer au moins un scénario."""
from langgraph.types import Command

from acsi_agent import team as team_mod
from acsi_agent.idempotency import IdempotencyRegistry
from acsi_agent.team import MissionInput, TeamRuntime
from audit_team.service import MissionService

SCOPES = ["audit:read", "audit:draft", "audit:submit"]


def mi(**kw):
    return MissionInput(task="Audite D-001", dossier_id="D-001", recipient="audit@client.example", scopes=SCOPES,
                        owner="alice", **kw)


def approve(svc, res):
    p = res["pending_approval"]
    return svc.resolve(res["mission_id"], p["approval_id"], "approve", "bob", ("audit:approve",), p["args_hash"])


def m04_holds(rt) -> bool:
    res = MissionService(rt, "direct").start(mi(doubles={"writer": "leak"}))
    return res["code"] == "REVIEW_BLOCKED" and res["pending_approval"] is None


def m05_holds(rt) -> bool:
    res = MissionService(rt, "direct").start(mi(doubles={"writer": "runaway"}, role_budgets={"writer": {"max_tool_calls": 3}}))
    return res["status"] == "budget_exceeded" and res["metrics"]["by_role"]["writer"]["tool_calls"] <= 3


def m09_holds(rt) -> bool:
    svc = MissionService(rt, "direct")
    a, b = approve(svc, svc.start(mi())), approve(svc, svc.start(mi()))
    return a["effects"]["mail_sent"] + b["effects"]["mail_sent"] == 1


def m11_holds(rt) -> bool:
    res = MissionService(rt, "direct").start(mi(doubles={"reviewer": "rogue"}))
    return not any(e["data"]["executed"] for e in rt.journal.read(res["mission_id"])
                   if e["type"] == "tool_call" and e.get("actor") == "reviewer" and e["data"]["tool"] == "draft_finding")


def m12_holds(rt) -> bool:
    svc = MissionService(rt, "direct")
    res = svc.start(mi())
    mid = res["mission_id"]
    try:  # arguments réécrits dans la mémoire du processus entre la demande et l'approbation
        rt._live[mid].approvals[res["pending_approval"]["approval_id"]] = (
            res["pending_approval"]["args_hash"], {"dossier_id": "D-001", "recipient": "autre@cabinet.example"})
        approve(svc, res)
    except Exception:  # noqa: BLE001
        pass
    sent = [e for e in rt.journal.read(mid) if e["type"] == "tool_call" and e["data"]["tool"] == "submit_dossier"]
    return not any(s["data"]["args"]["recipient"] == "autre@cabinet.example" and s["data"]["executed"] for s in sent)


def test_scenarios_hold_without_mutation(tmp_path):
    for i, check in enumerate((m04_holds, m05_holds, m09_holds, m11_holds, m12_holds)):
        assert check(TeamRuntime(tmp_path / f"j{i}")), check.__name__


def test_mutation_reviewer_block_ignored(tmp_path, monkeypatch):
    from audit_team import graph
    orig = graph.build_graph

    def mutated(rt, runner, brains):
        class Lenient:
            def __init__(self, inner):
                self.inner, self.id = inner, "lenient"

            def next(self, inputs, outcomes):
                d = self.inner.next(inputs, outcomes)
                if getattr(d, "output", {}).get("verdict") == "block":
                    from audit_team.doubles import Final
                    return Final({"verdict": "approve"})
                return d
        return orig(rt, runner, lambda role, dbl: Lenient(brains(role, dbl)) if role == "reviewer" else brains(role, dbl))
    monkeypatch.setattr("audit_team.service.build_graph", mutated)
    assert not m04_holds(TeamRuntime(tmp_path / "j"))


def test_mutation_role_budget_ignored(tmp_path, monkeypatch):
    orig = TeamRuntime._check
    monkeypatch.setattr(TeamRuntime, "_check", lambda self, st, actor, fn: orig(self, st, "__aucun_role__", fn))
    assert not m05_holds(TeamRuntime(tmp_path / "j"))


def test_mutation_idempotency_bypassed(tmp_path, monkeypatch):
    monkeypatch.setattr(IdempotencyRegistry, "reserve", lambda self, d, k, by: (True, None))
    monkeypatch.setattr(IdempotencyRegistry, "complete", lambda self, d, k, s: None)
    assert not m09_holds(TeamRuntime(tmp_path / "j"))


def test_mutation_role_scope_check_removed(tmp_path):
    rt = TeamRuntime(tmp_path / "j")
    rt.roles["reviewer"]["tools"].append("draft_finding")
    rt.roles["reviewer"]["scopes"].append("audit:draft")
    assert not m11_holds(rt)


def test_mutation_args_hash_check_removed(tmp_path, monkeypatch):
    monkeypatch.setattr(team_mod, "args_hash", lambda args: "0" * 64)
    assert not m12_holds(TeamRuntime(tmp_path / "j"))
