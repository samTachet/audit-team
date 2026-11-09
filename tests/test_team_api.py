import pytest
from fastapi.testclient import TestClient

from acsi_agent.agent import Runtime
from acsi_agent.team import TeamRuntime
from audit_team.api import create_app
from audit_team.service import MissionService

KEYS = {
    "k-alice": {"id": "alice", "role": "lead_auditor"},
    "k-bob": {"id": "bob", "role": "lead_auditor"},
    "k-carol": {"id": "carol", "role": "viewer"},
    "k-dave": {"id": "dave", "role": "auditor"},
}
H = lambda k: {"X-API-Key": k}  # noqa: E731
BODY = {"task": "Audite le dossier D-001 et transmets-le.", "dossier_id": "D-001", "recipient": "audit@client.example"}


@pytest.fixture(params=["direct", "crewai"])
def client(tmp_path, request):
    svc = MissionService(TeamRuntime(tmp_path / "j"), request.param)
    return TestClient(create_app(svc, KEYS, Runtime(tmp_path / "j")))


def test_mission_full_flow_over_http(client):
    r = client.post("/missions", json=BODY, headers=H("k-alice"))
    assert r.status_code == 201, r.text
    m = r.json()
    assert m["status"] == "awaiting_approval" and m["effects"]["mail_sent"] == 0
    url = f"/missions/{m['mission_id']}/approvals/{m['pending_approval']['approval_id']}"
    assert client.post(url, json={"decision": "approve", "args_hash": m["pending_approval"]["args_hash"]},
                       headers=H("k-alice")).json()["error"]["code"] == "SELF_APPROVAL"
    r = client.post(url, json={"decision": "approve"}, headers=H("k-bob"))
    assert r.status_code == 422 and r.json()["error"]["code"] == "ARGS_HASH_REQUIRED"
    r = client.post(url, json={"decision": "approve", "args_hash": "a" * 64}, headers=H("k-bob"))
    assert r.status_code == 409 and r.json()["error"]["code"] == "ARGS_MISMATCH"
    r = client.post(url, json={"decision": "approve", "args_hash": m["pending_approval"]["args_hash"]}, headers=H("k-bob"))
    assert r.status_code == 200 and r.json()["status"] == "completed" and r.json()["effects"]["mail_sent"] == 1
    mid = m["mission_id"]
    traj = client.get(f"/missions/{mid}/trajectory", headers=H("k-alice")).json()
    assert traj["chain_ok"] and all(e.get("actor") for e in traj["events"])
    assert "FIN completed" in client.get(f"/missions/{mid}/timeline", headers=H("k-alice")).text
    rep = client.post(f"/missions/{mid}/replay", headers=H("k-alice")).json()
    assert rep["identical"]
    assert client.get(f"/missions/{mid}/replay", headers=H("k-alice")).status_code == 405  # POST seulement


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_m14_api_rejects_non_finite_budgets(client, literal):
    for field in ('"budget": {"max_cost_usd": %s}', '"role_budgets": {"writer": {"max_steps": %s}}'):
        body = ('{"task": "Audite D-001", "dossier_id": "D-001", ' + field % literal + "}").encode()
        r = client.post("/missions", content=body, headers={**H("k-alice"), "content-type": "application/json"})
        assert r.status_code == 422 and r.json()["error"]["code"] == "INVALID_BUDGET"


def test_role_budget_above_global_is_invalid_budget(client):
    r = client.post("/missions", json={**BODY, "budget": {"max_tool_calls": 3}, "role_budgets": {"writer": {"max_tool_calls": 5}}},
                    headers=H("k-alice"))
    assert r.status_code == 422 and r.json()["error"]["code"] == "INVALID_BUDGET"


def test_mission_scopes_and_visibility(client):
    assert client.post("/missions", json=BODY, headers=H("k-dave")).status_code == 403  # auditor : pas d'audit:submit
    m = client.post("/missions", json={**BODY, "recipient": None}, headers=H("k-dave")).json()
    assert m["status"] == "completed"
    assert client.get(f"/missions/{m['mission_id']}", headers=H("k-carol")).status_code == 404
    assert client.get(f"/missions/{m['mission_id']}", headers=H("k-bob")).status_code == 200
    assert client.get("/missions/../../etc/passwd", headers=H("k-alice")).status_code == 404
    assert client.post("/missions", json={**BODY, "scopes": ["audit:read", "audit:approve"]}, headers=H("k-alice")).status_code == 422


def test_v1_tasks_still_served(client):
    r = client.post("/tasks", json={"task": "Liste les non-conformités du dossier D-001."}, headers=H("k-alice"))
    assert r.status_code == 201 and r.json()["status"] == "completed" and r.json()["request_id"].startswith("req_")
