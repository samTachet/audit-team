"""P0 — correctifs des garanties v1 (rapport d'analyse : B1, I1, I2, I3, I4, I5, N1, N2, N4, N6)."""
import hashlib
import json
import math
import os

import pytest
from fastapi.testclient import TestClient

from acsi_agent import eval as ev
from acsi_agent.agent import Runtime, TaskInput, args_hash
from acsi_agent.api import create_app
from acsi_agent.budget import Budget
from acsi_agent.journal import GENESIS, Journal, JournalSealed, _canon
from acsi_agent.planners import RuleBasedPlanner, ScriptedPlanner
from acsi_agent.specs import load_policy, load_scenarios
from acsi_agent.types import ToolAction, new_request_id

LEAD = ("audit:read", "audit:draft", "audit:submit", "audit:delete", "audit:approve")
SUBMIT = "Soumets le dossier D-001 à audit@client.example."
KEYS = {"k-alice": {"id": "alice", "role": "lead_auditor"}, "k-bob": {"id": "bob", "role": "lead_auditor"}}
H = lambda k: {"X-API-Key": k}  # noqa: E731


def submit_task(owner="alice"):
    return TaskInput(task=SUBMIT, scopes=["audit:read", "audit:submit"], owner=owner)


def approve(rt, res, approver="bob"):
    p = res.pending_approval
    return rt.resolve_approval(res.request_id, p["approval_id"], "approve", approver, LEAD, args_hash_seen=p["args_hash"])


# ================================================================ B1 : budget fini
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, 0, -1, True, "3"])
def test_budget_rejects_non_finite_and_non_positive(bad):
    with pytest.raises(ValueError):
        Budget.effective(load_policy()["budget_default"], {"max_steps": bad})


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_budget_dataclass_refuses_non_finite_values(bad):
    d = dict(load_policy()["budget_default"], max_cost_usd=bad)
    with pytest.raises(ValueError):
        Budget(**d)


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_runtime_refuses_runaway_task_with_non_finite_budget(runtime, bad):
    ti = TaskInput(task="Relis l'article 5 du RGPD.", planner={"type": "runaway", "tool": "lookup_article",
                                                                 "args": {"regulation": "RGPD", "article": "5"}},
                   budget={"max_steps": bad, "max_cost_usd": bad})
    with pytest.raises(ValueError):
        runtime.start(ti)


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_api_rejects_non_finite_budget_with_422_invalid_budget(tmp_path, literal):
    client = TestClient(create_app(Runtime(tmp_path / "j"), KEYS))
    body = '{"task": "Relis l\'article 5 du RGPD.", "budget": {"max_steps": %s}}' % literal
    r = client.post("/tasks", content=body.encode(), headers={**H("k-alice"), "content-type": "application/json"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "INVALID_BUDGET"
    assert client.app.state.runtime.journal.list_ids() == []  # aucune tâche créée


# ================================================================ I4 : planner qui lève
class Boom:
    id = "boom-planner"

    def plan(self, ctx):
        raise ConnectionError("quota dépassé")


class Garbage:
    id = "garbage-planner"

    def plan(self, ctx):
        return {"tool": "lookup_article"}  # ni ToolAction ni Finish


@pytest.mark.parametrize("planner", [Boom(), Garbage()])
def test_planner_error_closes_task_as_failed_and_frees_live_entry(runtime, planner):
    res = runtime.start(TaskInput(task="Relis l'article 5 du RGPD."), planner=planner)
    assert res.status == "failed"
    fin = next(e for e in res.events if e["type"] == "task_finished")["data"]
    assert fin["code"] == "PLANNER_ERROR"
    assert res.request_id not in runtime._live and res.request_id not in runtime._locks
    assert runtime.journal.verify(res.request_id) == (True, "ok")
    assert any(e["type"] == "planner_error" for e in res.events)


def test_planner_error_after_a_step_keeps_effects_listed(runtime):
    class Late:
        id = "late"
        n = 0

        def plan(self, ctx):
            self.n += 1
            if self.n == 1:
                return ToolAction("lookup_article", {"regulation": "RGPD", "article": "32"})
            raise ValueError("JSON invalide")
    res = runtime.start(TaskInput(task="Relis l'article 32 du RGPD."), planner=Late())
    assert res.status == "failed" and len(res.steps) == 1 and res.steps[0].status == "ok"


# ================================================================ I1 : journal ancré, HMAC, scellé
def _rewrite(path, events):
    path.write_text("".join(_canon(e) + "\n" for e in events), encoding="utf-8")


def _rechain(events):
    prev = GENESIS
    for i, e in enumerate(events, start=1):
        e.pop("mac", None)
        e["seq"], e["prev"] = i, prev
        core = {k: v for k, v in e.items() if k != "hash"}
        e["hash"] = hashlib.sha256((prev + _canon(core)).encode()).hexdigest()
        prev = e["hash"]
    return events


def test_truncated_tail_is_detected_after_execution(runtime):
    res = approve(runtime, runtime.start(submit_task()))
    assert res.effects["mail_sent"] == 1
    p, events = runtime.journal.path(res.request_id), runtime.journal.read(res.request_id)
    cut = next(i for i, e in enumerate(events) if e["type"] == "approval_requested") + 1
    _rewrite(p, events[:cut])
    ok, why = runtime.journal.verify(res.request_id)
    assert not ok and "tronqué" in why


def test_full_rewrite_with_recomputed_chain_is_detected_by_anchor(runtime):
    res = runtime.start(TaskInput(task="Relis l'article 32 du RGPD."))
    events = runtime.journal.read(res.request_id)
    events[0]["data"]["owner"] = "mallory"
    _rewrite(runtime.journal.path(res.request_id), _rechain(events))
    ok, why = runtime.journal.verify(res.request_id)
    assert not ok and "tête ancrée" in why


def test_hmac_detects_rewrite_of_journal_and_anchor_without_key(tmp_path):
    j = Journal(tmp_path / "j", hmac_key=b"cle-operateur-hors-volume")
    rid = new_request_id()
    j.append(rid, "a", {"x": 1})
    j.append(rid, "b", {"x": 2})
    assert j.verify(rid) == (True, "ok")
    events = _rechain(j.read(rid))
    events[1]["data"]["x"] = 3
    events = _rechain(events)
    _rewrite(j.path(rid), events)
    anc = j.anchor(rid)
    (j.anchor_dir / f"{rid}.head.json").write_text(json.dumps({**anc, "hash": events[-1]["hash"]}))
    ok, why = j.verify(rid)
    assert not ok and "HMAC" in why


def test_finished_task_journal_is_sealed(runtime):
    res = runtime.start(TaskInput(task="Relis l'article 32 du RGPD."))
    assert runtime.journal.anchor(res.request_id)["sealed"] is True
    with pytest.raises(JournalSealed):
        runtime.journal.append(res.request_id, "approval_resolved", {"forged": True})


def test_append_is_constant_time_with_in_memory_head(tmp_path, monkeypatch):
    j = Journal(tmp_path / "j")
    rid = new_request_id()
    j.append(rid, "a", {})
    calls = []
    monkeypatch.setattr(j, "read", lambda r: calls.append(r) or [])
    j.append(rid, "b", {})
    assert calls == []  # la tête vient de la mémoire, pas d'une relecture du fichier


# ================================================================ I2 : hachage gardé en mémoire
def test_rewritten_args_with_recomputed_chain_and_anchor_are_refused(runtime):
    """Accès disque complet (journal + ancre, sans HMAC) : la copie mémoire empêche quand même l'envoi détourné."""
    res = runtime.start(submit_task())
    rid = res.request_id
    events = runtime.journal.read(rid)
    req = next(e for e in events if e["type"] == "approval_requested")
    req["data"]["args"]["recipient"] = "autre@cabinet.example"  # domaine autorisé : la gateway ne l'arrêterait pas
    req["data"]["args_hash"] = args_hash(req["data"]["args"])
    events = _rechain(events)
    _rewrite(runtime.journal.path(rid), events)
    runtime.journal._write_anchor(rid, len(events), events[-1]["hash"])
    runtime.journal.forget(rid)
    assert runtime.journal.verify(rid) == (True, "ok")  # sans HMAC, la falsification complète passe la chaîne…
    from acsi_agent.agent import ApprovalError
    with pytest.raises(ApprovalError) as e:  # … mais pas la comparaison avec la mémoire
        runtime.resolve_approval(rid, req["data"]["approval_id"], "approve", "bob", LEAD,
                                 args_hash_seen=req["data"]["args_hash"])
    assert e.value.code == "ARGS_MISMATCH"
    assert runtime._live[rid].svc.effects()["mail_sent"] == 0


def test_approver_hash_must_match(runtime):
    from acsi_agent.agent import ApprovalError
    res = runtime.start(submit_task())
    with pytest.raises(ApprovalError) as e:
        runtime.resolve_approval(res.request_id, res.pending_approval["approval_id"], "approve", "bob", LEAD,
                                 args_hash_seen="f" * 64)
    assert e.value.code == "ARGS_MISMATCH"
    assert approve(runtime, runtime.result(res.request_id)).effects["mail_sent"] == 1


# ================================================================ I3 : idempotence entre tâches
def test_same_dossier_twice_across_tasks_sends_one_mail(runtime):
    first = approve(runtime, runtime.start(submit_task()))
    second = approve(runtime, runtime.start(submit_task()))
    assert first.status == "completed" and first.effects["mail_sent"] == 1
    assert second.effects["mail_sent"] == 0 and second.steps[-1].error["code"] == "ALREADY_SUBMITTED"


def test_idempotency_survives_a_restart(tmp_path):
    rt1 = Runtime(tmp_path / "j")
    approve(rt1, rt1.start(submit_task()))
    rt2 = Runtime(tmp_path / "j")  # « redémarrage » : nouveau processus, même stockage
    res = approve(rt2, rt2.start(submit_task()))
    assert res.effects["mail_sent"] == 0 and res.steps[-1].error["code"] == "ALREADY_SUBMITTED"


def test_unknown_outcome_blocks_any_retry_until_reconciled(tmp_path):
    rt = Runtime(tmp_path / "j", timeout_overrides={"submit_dossier": 0.05})
    slow = TaskInput(task=SUBMIT, scopes=["audit:read", "audit:submit"], owner="alice", faults={"mail_latency_s": 0.3})
    res = approve(rt, rt.start(slow))
    assert res.steps[-1].error["code"] == "TIMEOUT" and res.steps[-1].error["outcome_unknown"]
    entry = rt.idempotency.get("D-001")
    assert entry["state"] in {"in_flight", "submitted"}  # l'envoi a pu partir après le TIMEOUT
    again = approve(rt, rt.start(submit_task()))
    assert again.steps[-1].error["code"] in {"OUTCOME_UNKNOWN", "ALREADY_SUBMITTED"} and again.effects["mail_sent"] == 0
    import time
    time.sleep(0.5)  # le thread lent finit : l'envoi tardif est enregistré, il ne repartira jamais
    assert rt.idempotency.get("D-001")["state"] == "submitted"
    assert approve(rt, rt.start(submit_task())).steps[-1].error["code"] == "ALREADY_SUBMITTED"


def test_reconcile_releases_an_unknown_outcome(runtime):
    runtime.idempotency.reserve("D-001", "k", "req_x")  # envoi interrompu par un crash : issue inconnue
    assert approve(runtime, runtime.start(submit_task())).steps[-1].error["code"] == "OUTCOME_UNKNOWN"
    runtime.idempotency.reconcile("D-001", sent=False, note="vérifié auprès du destinataire")
    assert approve(runtime, runtime.start(submit_task())).effects["mail_sent"] == 1


def test_external_failure_releases_reservation(runtime):
    res = approve(runtime, runtime.start(TaskInput(task=SUBMIT, scopes=["audit:read", "audit:submit"], owner="alice",
                                                    faults={"mail": "unavailable"})))
    assert res.steps[-1].error["code"] == "EXTERNAL_UNAVAILABLE"
    assert runtime.idempotency.get("D-001") is None  # rien n'est parti : une nouvelle tentative (approuvée) est possible


# ================================================================ I5 / N1 / N2 / N4 / N6
def test_api_refuses_to_start_with_several_workers(tmp_path, monkeypatch):
    monkeypatch.setenv("WEB_CONCURRENCY", "2")
    with pytest.raises(RuntimeError):
        create_app(Runtime(tmp_path / "j"), KEYS)


def test_rate_token_is_reserved_at_allow_decision():
    from acsi_agent.backends import Services
    from acsi_agent.gateway import Gateway
    from acsi_agent.ratelimit import RateLimiter
    from acsi_agent.specs import load_tools
    from acsi_agent.tools_impl import build_impls
    from acsi_agent.types import CallContext
    lim = RateLimiter(1, clock=lambda: 0.0)
    g = Gateway(load_tools(), load_policy(), build_impls(Services.from_fixtures()), limiter=lim)
    c = CallContext("req_" + "0" * 32, 1, ("audit:read",), "k")
    args = {"regulation": "RGPD", "article": "5"}
    assert g.evaluate(c, "lookup_article", args).verdict == "allow"
    assert g.evaluate(c, "lookup_article", args).code == "RATE_LIMITED"  # deux décisions concurrentes : une seule passe


def test_sweeper_closes_orphans_and_expired_approvals(tmp_path):
    t = [1000.0]
    rt = Runtime(tmp_path / "j", clock=lambda: t[0])
    pending = rt.start(submit_task())
    assert rt.sweep() == []
    t[0] += load_policy()["approval"]["ttl_s"] + 1
    assert rt.sweep() == [pending.request_id]
    assert rt.result(pending.request_id).status == "expired"
    other = rt.start(submit_task())
    rt2 = Runtime(tmp_path / "j", clock=lambda: t[0])  # redémarrage : plus de Live
    assert rt2.sweep() == [other.request_id]
    fin = next(e for e in rt2.journal.read(other.request_id) if e["type"] == "task_finished")["data"]
    assert fin["code"] == "RUNTIME_RESTARTED" and fin["effects"] == {}


def test_eval_fails_when_a_scenario_is_skipped(monkeypatch, tmp_path):
    from acsi_agent import agent as agent_mod
    monkeypatch.setattr(agent_mod, "CAPABILITY_PHASE", 2)
    assert ev.main(["--journal-dir", str(tmp_path / "j")]) == 1


def test_mutation_without_pre_tool_budget_check_is_caught(runtime, monkeypatch):
    """N6 : la ligne « Budget dur » du README est maintenant couverte par une mutation."""
    from acsi_agent import agent as agent_mod
    monkeypatch.setattr(agent_mod, "check_pre_tool", lambda b, m: None)
    sc = next(s for s in load_scenarios() if s["id"] == "S14_budget_tool_calls")
    assert not ev.run_scenario(sc, runtime).success
