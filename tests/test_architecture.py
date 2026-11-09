"""ENF-01 / ENF-02 / ENF-07 / EF-03 : une seule porte, frameworks verrouillés, spec d'équipe vérifiée."""
import ast
import copy
import os
from pathlib import Path

import pytest

from acsi_agent import spec_check
from acsi_agent.specs import load_policy, load_tools
from acsi_agent.team import MissionInput, TeamRuntime, load_crew

TEAM_PKG = Path(__file__).resolve().parents[1] / "src" / "audit_team"
FORBIDDEN_MODULES = {"acsi_agent.backends", "acsi_agent.tools_impl", "acsi_agent.gateway"}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            out.add(n.module)
            out |= {f"{n.module}.{a.name}" for a in n.names}
    return out


def test_team_package_never_imports_backends_tools_or_gateway():
    files = sorted(TEAM_PKG.glob("*.py"))
    assert files
    for f in files:
        bad = _imports(f) & FORBIDDEN_MODULES
        assert not bad, f"{f.name} importe {bad}"
        src = f.read_text(encoding="utf-8")
        assert "Services" not in src and "build_impls" not in src, f.name


def test_team_package_only_reaches_tools_through_team_runtime():
    """Aucun accès aux attributs internes du runtime (services vivants, gateway) depuis le paquet de l'équipe."""
    for f in TEAM_PKG.glob("*.py"):
        src = f.read_text(encoding="utf-8")
        for attr in ("._live", ".gateway.", ".svc.", "impls"):
            assert attr not in src, (f.name, attr)


def test_specs_including_crew_are_consistent():
    assert spec_check.check() == []


@pytest.mark.parametrize("mutate,needle", [
    (lambda c: c["roles"][2]["tools"].append("submit_dossier"), "irréversible"),
    (lambda c: c["roles"][0]["tools"].append("shell_exec"), "hors allowlist"),
    (lambda c: c["roles"][1]["scopes"].remove("audit:draft"), "portée audit:draft manque"),
    (lambda c: c["framework_settings"].update(allow_delegation=True), "allow_delegation"),
    (lambda c: c["framework_settings"].update(native_tools=["SerperDevTool"]), "tool natif"),
    (lambda c: c.update(process="hierarchical"), "sequential"),
    (lambda c: c["roles"][0]["budget"].update(max_steps=float("nan")), "plafond max_steps invalide"),
    (lambda c: c["roles"][0]["budget"].update(max_tool_calls=999), "supérieur au budget global"),
])
def test_spec_check_refuses_unsafe_roles(mutate, needle):
    crew = copy.deepcopy(load_crew())
    mutate(crew)
    errs = spec_check.check_crew(crew, load_tools(), load_policy())
    assert any(needle in e for e in errs), errs


def test_crewai_agents_and_crew_are_locked_down(tmp_path):
    from audit_team.crew import GatewayTool, build_team_crew
    from audit_team.doubles import BrainFactory
    rt = TeamRuntime(tmp_path / "j")
    mid = rt.open_mission(MissionInput(task="t", dossier_id="D-001", owner="alice"))
    f = BrainFactory()
    crew = build_team_crew(rt, mid, {r: f(r, {}) for r in ("analyst", "writer", "reviewer")},
                           {r: {"dossier_id": "D-001"} for r in ("analyst", "writer", "reviewer")})
    from crewai import Process
    assert crew.process == Process.sequential and not crew.memory and not crew.cache
    assert [a.role for a in crew.agents] == ["Analyste de dossier", "Rédacteur de constats", "Relecteur conformité"]
    for a in crew.agents:
        assert a.allow_delegation is False and a.allow_code_execution is False and a.cache is False
        assert a.max_retry_limit == 0
        assert a.tools and all(isinstance(t, GatewayTool) for t in a.tools)
        assert not {t.name for t in a.tools} & {"submit_dossier", "delete_draft"}
        assert all(t.cache_function(None, None) is False for t in a.tools)


def test_gateway_tool_never_serves_from_cache(tmp_path):
    from audit_team.crew import GatewayTool
    rt = TeamRuntime(tmp_path / "j")
    mid = rt.open_mission(MissionInput(task="t", dossier_id="D-001", owner="alice"))
    sink = []
    tool = GatewayTool.build(rt, mid, "analyst", "lookup_article", sink)
    for _ in range(2):
        tool.run(regulation="RGPD", article="32")
    calls = [e for e in rt.journal.read(mid) if e["type"] == "tool_call"]
    assert len(calls) == 2 and all(c["data"]["gateway"]["verdict"] == "allow" for c in calls)


def test_gateway_tool_schema_forbids_extra_fields(tmp_path):
    from audit_team.crew import args_model
    from pydantic import ValidationError
    m = args_model("lookup_article", load_tools()["lookup_article"].input_schema)
    with pytest.raises(ValidationError):
        m(regulation="RGPD", article="32", recipient="x@evil.example")


def test_telemetry_and_tracing_are_disabled_by_default():
    import audit_team  # noqa: F401
    assert os.environ["CREWAI_DISABLE_TELEMETRY"] == "true" and os.environ["OTEL_SDK_DISABLED"] == "true"
    assert os.environ["CREWAI_TRACING_ENABLED"] == "false"
    from crewai.telemetry.telemetry import Telemetry
    assert Telemetry._is_telemetry_disabled()  # contrôle de CrewAI lui-même, pas seulement nos variables
