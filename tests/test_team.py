"""Scénarios multi-agents M01–M16 (cahier des charges, section 10), joués avec les deux runners :
`direct` (P1/P2 : graphe LangGraph, sans CrewAI) et `crewai` (P3 : crew CrewAI derrière la gateway)."""
import math
import socket

import pytest

from acsi_agent.agent import ApprovalError
from acsi_agent.team import MissionInput, TeamRuntime
from audit_team.doubles import BrainFactory, WriterDouble
from audit_team.replay import replay_mission
from audit_team.service import MissionService, sqlite_checkpointer

SUBMIT_SCOPES = ["audit:read", "audit:draft", "audit:submit"]
APPROVER = ("audit:read", "audit:approve")


@pytest.fixture(params=["direct", "crewai"])
def runner(request):
    return request.param


@pytest.fixture
def rt(tmp_path):
    return TeamRuntime(tmp_path / "journal", hmac_key=b"cle-de-test")


@pytest.fixture
def svc(rt, runner):
    return MissionService(rt, runner)


def mi(dossier="D-001", recipient="audit@client.example", doubles=None, scopes=None, **kw):
    return MissionInput(task=f"Audite le dossier {dossier} et prépare sa transmission.", dossier_id=dossier,
                        recipient=recipient, scopes=scopes or list(SUBMIT_SCOPES), owner="alice",
                        doubles=doubles or {}, **kw)


def approve(svc, res, approver="bob", decision="approve"):
    p = res["pending_approval"]
    return svc.resolve(res["mission_id"], p["approval_id"], decision, approver, APPROVER, p["args_hash"])


def events(rt, res, type_=None):
    ev = rt.journal.read(res["mission_id"])
    return [e for e in ev if type_ is None or e["type"] == type_]


def first_index(evs, pred):
    return next((i for i, e in enumerate(evs) if pred(e)), None)


# ---------------------------------------------------------------- M01 nominal
def test_m01_nominal_mission(svc, rt):
    res = svc.start(mi())
    assert res["status"] == "awaiting_approval" and res["effects"]["mail_sent"] == 0
    assert res["pending_approval"]["proposed_by"] == "writer" and res["review_verdicts"] == ["approve"]
    done = approve(svc, res)
    assert done["status"] == "completed" and done["effects"]["mail_sent"] == 1
    evs = events(rt, done)
    assert all(e.get("actor") for e in evs)  # chaque événement a un acteur
    assert {e["actor"] for e in evs} >= {"alice", "analyst", "writer", "reviewer", "human", "acsi"}
    # O3 : tout tool exécuté l'a été après un verdict allow de la gateway
    assert all(e["data"]["gateway"]["verdict"] == "allow" for e in evs if e["type"] == "tool_call" and e["data"]["executed"])
    # O4 : l'effet externe suit une approbation humaine au même args_hash
    sub = first_index(evs, lambda e: e["type"] == "tool_call" and e["data"]["tool"] == "submit_dossier")
    res_i = first_index(evs, lambda e: e["type"] == "approval_resolved")
    assert res_i < sub and evs[res_i]["data"]["decision"] == "approve"
    assert evs[res_i]["data"]["args_hash"] == res["pending_approval"]["args_hash"]
    # O5 : un verdict approve du relecteur précède la demande d'approbation
    assert first_index(evs, lambda e: e["type"] == "review_verdict") < first_index(evs, lambda e: e["type"] == "approval_requested")
    assert [e["data"]["to"] for e in evs if e["type"] == "handoff"] == ["writer", "reviewer"]
    assert rt.journal.verify(done["mission_id"]) == (True, "ok") and rt.journal.anchor(done["mission_id"])["sealed"]
    assert not rt.is_live(done["mission_id"])


def test_no_role_can_touch_irreversible_tools(svc, rt):
    res = svc.start(mi())
    mid = res["mission_id"]
    for role in ("analyst", "writer", "reviewer", "intrus"):
        for tool in ("submit_dossier", "delete_draft"):
            out = rt.propose(mid, role, tool, {"dossier_id": "D-001", "recipient": "audit@client.example"})
            assert out["status"] == "denied" and out["code"] == "PERMISSION_DENIED", (role, tool)
    assert rt.effects(mid)["mail_sent"] == 0
    from audit_team.crew import tools_for
    for role in ("analyst", "writer", "reviewer"):
        names = {t.name for t in tools_for(rt, mid, role, [])}
        assert names == set(rt.roles[role]["tools"]) and not names & {"submit_dossier", "delete_draft"}


def test_approval_requires_a_prior_review_approve(rt):
    mid = rt.open_mission(mi())
    t = rt.request_approval(mid, "writer", {"tool": "submit_dossier", "args": {"dossier_id": "D-001", "recipient": "audit@client.example"}})
    assert t["status"] == "denied" and t["code"] == "REVIEW_REQUIRED"
    assert rt.state(mid).pending is None


def test_mission_without_recipient_completes_without_any_approval(svc, rt):
    res = svc.start(mi(recipient=None))
    assert res["status"] == "completed" and res["effects"] == {"mail_sent": 0, "drafts_created": 2, "drafts_deleted": 0}
    assert events(rt, res, "approval_requested") == []


# ---------------------------------------------------------------- M02–M04 revue
def test_m02_disagreement_then_agreement(svc, rt):
    res = svc.start(mi(doubles={"writer": "sloppy_first"}))
    assert res["review_verdicts"] == ["request_changes", "approve"]
    done = approve(svc, res)
    assert done["status"] == "completed" and done["effects"]["mail_sent"] == 1
    assert done["effects"]["drafts_created"] == 2  # le tour 2 révise les brouillons, il ne les duplique pas
    assert len(events(rt, done, "review_verdict")) == 2


def test_m03_persistent_disagreement(svc, rt):
    res = svc.start(mi(doubles={"writer": "always_sloppy"}, max_review_rounds=2))
    assert res["status"] == "failed" and res["code"] == "REVIEW_ROUNDS_EXCEEDED"
    assert events(rt, res, "approval_requested") == [] and res["effects"]["mail_sent"] == 0


def test_m04_reviewer_blocks(svc, rt):
    res = svc.start(mi(doubles={"writer": "leak"}))
    assert res["status"] == "refused" and res["code"] == "REVIEW_BLOCKED"
    assert events(rt, res, "approval_requested") == [] and res["effects"]["mail_sent"] == 0


# ---------------------------------------------------------------- M05–M06 budget
def test_m05_role_budget_exhausted(svc, rt):
    res = svc.start(mi(doubles={"writer": "runaway"}, role_budgets={"writer": {"max_tool_calls": 3}}))
    assert res["status"] == "budget_exceeded"
    stop = res["budget_exceeded"]
    assert (stop["scope"], stop["role"], stop["dimension"]) == ("role", "writer", "tool_calls")
    assert stop["proposed_action_not_executed"]["tool"] == "draft_finding"
    assert res["metrics"]["by_role"]["writer"]["tool_calls"] == 3 and res["effects"]["mail_sent"] == 0


def test_m06_global_budget_exhausted(svc, rt):
    res = svc.start(mi(budget={"max_tool_calls": 4}))
    assert res["status"] == "budget_exceeded"
    assert (res["budget_exceeded"]["scope"], res["budget_exceeded"]["role"]) == ("global", None)
    by_role = res["metrics"]["by_role"]
    assert all(m["tool_calls"] < 4 for m in by_role.values()) and res["metrics"]["global"]["tool_calls"] == 4


def test_role_llm_budget_and_warnings(svc, rt):
    res = svc.start(mi(role_budgets={"analyst": {"max_steps": 2}}))
    assert res["status"] == "budget_exceeded" and res["budget_exceeded"]["dimension"] == "steps"
    assert res["budget_exceeded"]["role"] == "analyst"
    assert any(e["data"]["scope"] == "role:analyst" for e in events(rt, res, "budget_warning"))


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_m14_non_finite_budgets_are_refused(rt, bad):
    for kw in ({"budget": {"max_cost_usd": bad}}, {"role_budgets": {"writer": {"max_steps": bad}}}):
        with pytest.raises(ValueError):
            rt.open_mission(mi(**kw))
    assert rt.journal.list_ids("mis_") == []


def test_role_budget_cannot_exceed_global(rt):
    with pytest.raises(ValueError):
        rt.open_mission(mi(budget={"max_tool_calls": 5}, role_budgets={"writer": {"max_tool_calls": 8}}))
    mid = rt.open_mission(mi(budget={"max_tool_calls": 5}))  # défaut de rôle ramené au global
    assert rt.state(mid).role_budgets["writer"].max_tool_calls == 5


# ---------------------------------------------------------------- M07–M08 crash et reprise
class SimulatedCrash(BaseException):
    """Arrêt brutal du worker (ni Exception ni erreur du modèle)."""


def test_m07_crash_during_drafting_then_resume(rt, runner):
    class CrashOnce(WriterDouble):
        crashed = False

        def next(self, inputs, outcomes):
            if len(outcomes) == 1 and not CrashOnce.crashed:
                CrashOnce.crashed = True
                raise SimulatedCrash()
            return super().next(inputs, outcomes)
    saver = sqlite_checkpointer(rt.journal.dir.parent / "checkpoints.sqlite")
    svc = MissionService(rt, runner, checkpointer=saver, brains=BrainFactory({"writer": CrashOnce()}))
    mid = rt.open_mission(mi())
    with pytest.raises(SimulatedCrash):
        svc._invoke({"mission_id": mid, "journal_seq": 0}, mid)
    assert rt.public(mid)["status"] == "running" and rt.effects(mid)["drafts_created"] == 1
    svc2 = MissionService(rt, runner, checkpointer=saver, brains=BrainFactory({"writer": CrashOnce()}))  # nouveau worker
    assert svc2.snapshot(mid)["next"] == ["redaction"]  # reprise au checkpoint : l'analyse n'est pas refaite
    res = svc2.resume(mid)
    assert res["status"] == "awaiting_approval"
    assert res["effects"]["drafts_created"] == 2  # brouillon du premier passage réutilisé, pas dupliqué
    assert sum(1 for e in events(rt, res, "node_entered") if e["data"]["node"] == "analyse") == 1
    done = approve(svc2, res)
    assert done["status"] == "completed" and done["effects"]["mail_sent"] == 1
    assert rt.journal.verify(mid) == (True, "ok")


def test_m08_crash_while_awaiting_approval(tmp_path, runner):
    saver = sqlite_checkpointer(tmp_path / "checkpoints.sqlite")
    rt1 = TeamRuntime(tmp_path / "journal")
    res = MissionService(rt1, runner, checkpointer=saver).start(mi())
    assert res["status"] == "awaiting_approval"
    rt2 = TeamRuntime(tmp_path / "journal")  # redémarrage du processus : mémoire (services, args_hash) perdue
    svc2 = MissionService(rt2, runner, checkpointer=saver)
    done = approve(svc2, rt2.public(res["mission_id"]))
    assert done["status"] == "failed" and done["code"] == "RUNTIME_RESTARTED" and done["effects"] == {}
    assert not any(e["data"]["executed"] for e in events(rt2, done, "tool_call") if e["data"]["tool"] == "submit_dossier")


def test_sweeper_closes_orphan_missions(tmp_path):
    rt1 = TeamRuntime(tmp_path / "journal")
    res = MissionService(rt1, "direct").start(mi())
    rt2 = TeamRuntime(tmp_path / "journal")
    assert rt2.sweep() == [res["mission_id"]]
    assert rt2.public(res["mission_id"])["code"] == "RUNTIME_RESTARTED"


def test_expired_approval_is_never_executed(tmp_path):
    t = [1000.0]
    rt = TeamRuntime(tmp_path / "journal", clock=lambda: t[0])
    svc = MissionService(rt, "direct")
    res = svc.start(mi())
    t[0] += rt.policy["approval"]["ttl_s"] + 1
    with pytest.raises(ApprovalError) as e:
        approve(svc, res)
    assert e.value.code in {"EXPIRED", "NOT_PENDING"}
    assert rt.public(res["mission_id"])["status"] == "expired" and rt.effects(res["mission_id"])["mail_sent"] == 0


# ---------------------------------------------------------------- M09 double soumission
def test_m09_double_submission_across_missions(svc, rt):
    a = approve(svc, svc.start(mi()))
    b = approve(svc, svc.start(mi()))
    assert a["status"] == "completed" and a["effects"]["mail_sent"] == 1
    assert b["status"] == "failed" and b["code"] == "ALREADY_SUBMITTED" and b["effects"]["mail_sent"] == 0


# ---------------------------------------------------------------- M10–M11 injection et tool hors rôle
def test_m10_injection_targeting_a_role(svc, rt, runner):
    res = svc.start(mi(dossier="D-002", recipient=None, doubles={"analyst": "injected", "reviewer": "naive"}))
    assert res["status"] == "refused" and res["code"] == "RECIPIENT_NOT_ALLOWED"
    assert events(rt, res, "approval_requested") == [] and res["effects"]["mail_sent"] == 0
    calls = [e["data"] for e in events(rt, res, "tool_call")]
    assert not any(c["executed"] and c["tool"] in {"shell_exec", "submit_dossier"} for c in calls)
    if runner == "direct":
        assert any(c["tool"] == "shell_exec" and c["error"]["code"] == "TOOL_FORBIDDEN" for c in calls)
    else:  # CrewAI ne connaît même pas le tool : la tentative est tracée, rien n'atteint un service
        assert any(e["data"].get("tool") == "shell_exec" for e in events(rt, res, "agent_error"))


def test_m10_normal_doubles_ignore_the_injection(svc, rt):
    res = svc.start(mi(dossier="D-002", recipient="audit@client.example"))
    assert res["status"] == "awaiting_approval"
    assert res["pending_approval"]["args"]["recipient"] == "audit@client.example"


def test_m11_tool_outside_role(svc, rt, runner):
    res = svc.start(mi(doubles={"reviewer": "rogue"}))
    assert res["status"] == "awaiting_approval"  # la tentative est refusée, la mission continue
    rev_drafts = [e for e in events(rt, res, "tool_call") if e["actor"] == "reviewer" and e["data"]["tool"] == "draft_finding"]
    assert not any(e["data"]["executed"] for e in rev_drafts)
    if runner == "direct":
        assert rev_drafts and rev_drafts[0]["data"]["error"]["code"] == "PERMISSION_DENIED"
    out = rt.propose(res["mission_id"], "reviewer", "draft_finding",
                     {"dossier_id": "D-001", "nc_id": "NC-011", "text": "Réécriture par le relecteur."})
    assert out["code"] == "PERMISSION_DENIED"


def test_repeated_denials_stop_the_mission(rt):
    mid = rt.open_mission(mi())
    for _ in range(3):
        rt.propose(mid, "analyst", "shell_exec", {"command": "id"})
    assert rt.denial_limit_reached(mid, "analyst")


# ---------------------------------------------------------------- M12 approbation altérée
def test_m12_stale_or_altered_approval(svc, rt):
    res = svc.start(mi())
    mid, p = res["mission_id"], res["pending_approval"]
    with pytest.raises(ApprovalError) as e:
        svc.resolve(mid, p["approval_id"], "approve", "bob", APPROVER, "0" * 64)
    assert e.value.code == "ARGS_MISMATCH"
    with pytest.raises(ApprovalError) as e:
        svc.resolve(mid, "apr_000000000000", "approve", "bob", APPROVER, p["args_hash"])
    assert e.value.code == "NOT_PENDING"
    with pytest.raises(ApprovalError) as e:
        svc.resolve(mid, p["approval_id"], "approve", "alice", APPROVER, p["args_hash"])
    assert e.value.code == "SELF_APPROVAL"
    with pytest.raises(ApprovalError) as e:
        svc.resolve(mid, p["approval_id"], "approve", "carol", ("audit:read",), p["args_hash"])
    assert e.value.code == "FORBIDDEN_SCOPE"
    assert rt.effects(mid)["mail_sent"] == 0 and rt.public(mid)["status"] == "awaiting_approval"
    done = approve(svc, rt.public(mid))
    assert done["effects"]["mail_sent"] == 1
    with pytest.raises(ApprovalError) as e:  # usage unique
        svc.resolve(mid, p["approval_id"], "approve", "bob", APPROVER, p["args_hash"])
    assert e.value.code == "NOT_PENDING"


def test_forged_resume_without_acsi_executes_nothing(svc, rt):
    from langgraph.types import Command
    res = svc.start(mi())
    mid = res["mission_id"]
    svc._invoke(Command(resume={"approval_id": res["pending_approval"]["approval_id"], "decision": "approve"}), mid)
    out = rt.public(mid)
    assert out["status"] == "refused" and out["code"] == "NOT_APPROVED" and out["effects"]["mail_sent"] == 0


def test_rejection_closes_as_refused(svc, rt):
    done = approve(svc, svc.start(mi()), decision="reject")
    assert done["status"] == "refused" and done["code"] == "APPROVAL_REJECTED" and done["effects"]["mail_sent"] == 0


# ---------------------------------------------------------------- M13 erreur d'un agent
@pytest.mark.parametrize("role", ["analyst", "writer", "reviewer"])
def test_m13_agent_raises(svc, rt, role):
    res = svc.start(mi(doubles={role: "raise"}))
    assert res["status"] == "failed" and res["code"] == "PLANNER_ERROR"
    err = events(rt, res, "agent_error")[-1]
    assert err["actor"] == role and err["data"]["role"] == role
    assert not rt.is_live(res["mission_id"]) and res["effects"]["mail_sent"] == 0


def test_invalid_agent_output_is_a_planner_error(rt, runner):
    class Bad:
        id = "bad"

        def next(self, inputs, outcomes):
            from audit_team.doubles import Final
            return Final({"verdict": "peut-être"})
    res = MissionService(rt, runner, brains=BrainFactory({"reviewer": Bad()})).start(mi())
    assert res["status"] == "failed" and res["code"] == "PLANNER_ERROR"


# ---------------------------------------------------------------- M15–M16 replay et journal
@pytest.mark.parametrize("doubles,decide", [({}, "approve"), ({"writer": "sloppy_first"}, "approve"),
                                            ({"writer": "leak"}, None), ({}, "reject")])
def test_m15_replay_is_identical(svc, rt, doubles, decide):
    res = svc.start(mi(doubles=doubles))
    if decide:
        res = approve(svc, res, decision=decide)
    rep = replay_mission(rt, res["mission_id"])
    assert rep["replayable"] and rep["identical"], rep


def test_replay_detects_a_changed_gateway(svc, rt):
    res = approve(svc, svc.start(mi()))
    rt.policy = {**rt.policy, "argument_rules": [{**rt.policy["argument_rules"][0], "allowed_domains": ["cabinet.example"]}]}
    rep = replay_mission(rt, res["mission_id"])
    assert not rep["identical"]


def test_m16_truncated_journal_after_sending(svc, rt):
    res = svc.start(mi())
    p = res["pending_approval"]
    done = approve(svc, res)
    assert done["effects"]["mail_sent"] == 1
    mid = done["mission_id"]
    path = rt.journal.path(mid)
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    cut = next(i for i, line in enumerate(lines) if '"type":"approval_requested"' in line) + 1
    path.write_text("".join(lines[:cut]), encoding="utf-8")
    rt.journal.forget(mid)
    ok, why = rt.journal.verify(mid)
    assert not ok and "tronqué" in why
    with pytest.raises(ApprovalError) as e:
        svc.resolve(mid, p["approval_id"], "approve", "bob", APPROVER, p["args_hash"])
    assert e.value.code == "JOURNAL_TAMPERED"


def test_checkpoint_divergence_is_detected(svc, rt):
    res = svc.start(mi())
    mid = res["mission_id"]
    services = rt._live[mid].svc
    path = rt.journal.path(mid)
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    path.write_text("".join(lines[:-3]), encoding="utf-8")  # journal ramené en arrière derrière le checkpoint
    rt.journal.forget(mid)
    from langgraph.types import Command
    svc._invoke(Command(resume={"approval_id": res["pending_approval"]["approval_id"], "decision": "approve"}), mid)
    assert svc.snapshot(mid)["values"]["code"] == "CHECKPOINT_DIVERGENCE"
    assert services.effects()["mail_sent"] == 0 and not rt.is_live(mid)
    assert len(rt.journal.read(mid)) == len(lines) - 3  # rien n'a été ajouté par-dessus la troncature


# ---------------------------------------------------------------- hermétisme
def test_crewai_mission_makes_no_network_call(rt, monkeypatch):
    def no_net(*a, **k):
        raise AssertionError(f"appel réseau interdit : {a[1:] if len(a) > 1 else a}")
    monkeypatch.setattr(socket.socket, "connect", no_net)
    monkeypatch.setattr(socket, "create_connection", no_net)
    svc = MissionService(rt, "crewai")
    assert approve(svc, svc.start(mi()))["status"] == "completed"
