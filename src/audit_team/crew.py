"""Intégration CrewAI (P3) : agents verrouillés, tools = adaptateurs vers la gateway ACSI, LLM gouverné.

- `GatewayTool(BaseTool)` : un adaptateur par tool ACSI autorisé au rôle. `args_schema` dérivé de
  `specs/tools.json` (champs supplémentaires interdits) ; `_run` délègue à `TeamRuntime.propose` ; cache
  neutralisé (`cache_function` ne met jamais en cache). `submit_dossier` et `delete_draft` n'ont PAS d'adaptateur.
- `GovernedLLM(BaseLLM)` : enveloppe le « cerveau » d'un rôle (double déterministe aujourd'hui, LLM réel en P4).
  Avant chaque appel : `check_llm` (budget ACSI) ; après : `charge_llm`. Il produit du texte ReAct
  (`Action:` / `Final Answer:`) que l'exécuteur de CrewAI interprète.
- Agents : `allow_delegation=False`, `allow_code_execution=False`, `cache=False`, mémoire désactivée,
  `max_retry_limit=0`, aucun tool natif ; `max_iter` large : c'est le budget ACSI qui arrête, pas CrewAI.
- Crew : `Process.sequential`, `memory=False`, `cache=False`. Le graphe lance une phase = une crew d'un rôle ;
  `build_team_crew` assemble la crew séquentielle des trois rôles (même verrouillage) pour inspection.
- Télémétrie : coupée par `audit_team/__init__.py` (`CREWAI_DISABLE_TELEMETRY`, `OTEL_SDK_DISABLED`,
  `CREWAI_TRACING_ENABLED=false`) avant l'import de crewai.
"""
from __future__ import annotations

import json
import re
from typing import Any

from crewai import Agent, Crew, Process, Task
from crewai.llms.base_llm import BaseLLM
from crewai.tools import BaseTool
from pydantic import BaseModel, ConfigDict, PrivateAttr, create_model

from acsi_agent.team import TeamRuntime

from .doubles import Act, Brain, Final
from .phases import PhaseError, PhaseResult, decide


def _never_cache(_args: Any = None, _result: Any = None) -> bool:
    return False


def args_model(tool_name: str, input_schema: dict[str, Any]) -> type[BaseModel]:
    """Schéma Pydantic fermé dérivé de la spec ACSI (la validation fine reste dans la gateway)."""
    req = set(input_schema.get("required", []))
    fields: dict[str, Any] = {}
    for k in input_schema.get("properties", {}):
        fields[k] = (str, ...) if k in req else (str | None, None)
    return create_model(f"{tool_name}_args", __config__=ConfigDict(extra="forbid"), **fields)


class GatewayTool(BaseTool):
    """Adaptateur CrewAI → ACSI. Aucune référence aux services ni aux implémentations des tools."""
    _rt: TeamRuntime = PrivateAttr()
    _mid: str = PrivateAttr()
    _actor: str = PrivateAttr()
    _sink: list = PrivateAttr()

    @classmethod
    def build(cls, rt: TeamRuntime, mid: str, actor: str, tool: str, sink: list) -> "GatewayTool":
        spec = rt.tools[tool]
        t = cls(name=tool, description=spec.purpose, args_schema=args_model(tool, spec.input_schema),
                cache_function=_never_cache)
        t._rt, t._mid, t._actor, t._sink = rt, mid, actor, sink
        return t

    def _run(self, **kwargs: Any) -> str:
        args = {k: v for k, v in kwargs.items() if v is not None}
        out = self._rt.propose(self._mid, self._actor, self.name, args)
        self._sink.append({"tool": self.name, "args": args, **out})
        return json.dumps(out, ensure_ascii=False)


def tools_for(rt: TeamRuntime, mid: str, role: str, sink: list) -> list[GatewayTool]:
    return [GatewayTool.build(rt, mid, role, t, sink) for t in rt.roles[role]["tools"] if t not in rt.graph_only]


class GovernedLLM(BaseLLM):
    """« LLM » CrewAI dont chaque appel est imputé et plafonné par ACSI."""

    def __init__(self, rt: TeamRuntime, mid: str, role: str, brain: Brain, inputs: dict[str, Any], sink: list):
        super().__init__(model=f"governed:{getattr(brain, 'id', 'brain')}")
        self._rt, self._mid, self._role, self._brain, self._inputs, self._sink = rt, mid, role, brain, inputs, sink
        self._emitted = 0
        self._last: Act | None = None
        self.error: PhaseError | None = None
        self.abort: str | None = None

    def supports_function_calling(self) -> bool:
        return False

    def supports_stop_words(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 64000

    def get_input_tokens_count(self, messages, tools=None, response_model=None) -> int:
        # Même estimation que CrewAI (1 jeton / 4 caractères), sans l'avertissement : l'imputation ACSI
        # réelle se fait dans `decide` (usage_for), pas ici.
        from crewai.utilities.message_utils import message_content_text
        text = messages if isinstance(messages, str) else "".join(message_content_text(m) for m in messages)
        return (len(text) + 3) // 4

    def _final(self, obj: dict[str, Any], thought: str) -> str:
        return f"Thought: {thought}\nFinal Answer: {json.dumps(obj, ensure_ascii=False)}"

    def call(self, messages, tools=None, callbacks=None, available_functions=None, from_task=None, from_agent=None,
             response_model=None, **kwargs) -> str:
        if self.error or self.abort:
            return self._final({"aborted": self.abort or "error"}, "Arrêt.")
        if self._last is not None and len(self._sink) < self._emitted:
            # L'exécuteur CrewAI n'a pas appelé le tool (absent du rôle, ou réutilisation refusée) : rien n'a
            # atteint la gateway. On le trace et on le rend visible au cerveau.
            self._sink.append({"tool": self._last.tool, "args": self._last.args, "status": "denied",
                               "code": "TOOL_NOT_AVAILABLE", "result": None})
            self._rt.record(self._mid, "agent_error", {"code": "TOOL_NOT_AVAILABLE", "tool": self._last.tool,
                                                       "where": "crewai"}, actor=self._role)
        if self._rt.denial_limit_reached(self._mid, self._role):
            self.abort = "denials"
            return self._final({"aborted": "denials"}, "Trop de refus.")
        try:
            d = decide(self._rt, self._mid, self._role, self._brain, self._inputs, list(self._sink))
        except PhaseError as e:
            self.error = e
            return self._final({"aborted": "error"}, "Erreur du modèle.")
        if d is None:
            self.abort = "budget"
            return self._final({"aborted": "budget"}, "Budget ACSI atteint.")
        if isinstance(d, Final):
            return self._final(d.output, d.thought or "J'ai terminé.")
        self._last = d
        self._emitted = len(self._sink) + 1
        return f"Thought: {d.thought or 'Je continue.'}\nAction: {d.tool}\nAction Input: {json.dumps(d.args, ensure_ascii=False)}"


def _settings(rt: TeamRuntime) -> dict[str, Any]:
    return rt.crew["framework_settings"]


def build_agent(rt: TeamRuntime, mid: str, role: str, llm: BaseLLM, sink: list) -> Agent:
    spec, fs = rt.roles[role], _settings(rt)
    return Agent(role=spec["role"], goal=spec["goal"], backstory=spec["backstory"], llm=llm,
                 tools=tools_for(rt, mid, role, sink), allow_delegation=fs["allow_delegation"],
                 allow_code_execution=fs["allow_code_execution"], cache=fs["cache"], memory=fs["memory"],
                 max_retry_limit=fs["max_retry_limit"], max_iter=fs["max_iter"], verbose=False)


def build_task(rt: TeamRuntime, role: str, inputs: dict[str, Any], agent: Agent) -> Task:
    spec = rt.roles[role]
    return Task(description=f"{spec['goal']}\n\nEntrées (JSON) :\n{json.dumps(inputs, ensure_ascii=False, sort_keys=True)}",
                expected_output=f"Un objet JSON `{spec['output']}`, sans texte autour.", agent=agent)


def build_crew(agents: list[Agent], tasks: list[Task], rt: TeamRuntime) -> Crew:
    fs = _settings(rt)
    assert rt.crew["process"] == "sequential"  # Process.hierarchical exclu en v2.0
    return Crew(agents=agents, tasks=tasks, process=Process.sequential, memory=fs["memory"], cache=fs["cache"],
                verbose=False)


def build_team_crew(rt: TeamRuntime, mid: str, brains: dict[str, Brain], inputs: dict[str, dict[str, Any]]) -> Crew:
    """Crew séquentielle des trois rôles (analyste → rédacteur → relecteur), même verrouillage que les phases."""
    agents, tasks = [], []
    for role in ("analyst", "writer", "reviewer"):
        sink: list = []
        a = build_agent(rt, mid, role, GovernedLLM(rt, mid, role, brains[role], inputs[role], sink), sink)
        agents.append(a)
        tasks.append(build_task(rt, role, inputs[role], a))
    return build_crew(agents, tasks, rt)


_JSON_RE = re.compile(r"\{.*\}", re.S)


class CrewAIPhaseRunner:
    """Une phase du graphe = une crew CrewAI d'un rôle, `Process.sequential`."""
    id = "crewai-runner-v1"

    def __init__(self, rt: TeamRuntime):
        self.rt = rt

    def run(self, mission_id: str, role: str, inputs: dict[str, Any], brain: Brain) -> PhaseResult:
        sink: list[dict[str, Any]] = []
        llm = GovernedLLM(self.rt, mission_id, role, brain, inputs, sink)
        agent = build_agent(self.rt, mission_id, role, llm, sink)
        crew = build_crew([agent], [build_task(self.rt, role, inputs, agent)], self.rt)
        result = crew.kickoff()
        if llm.error is not None:
            raise llm.error
        if llm.abort == "denials":
            return PhaseResult(None, stopped=True, reason="denials", outcomes=sink)
        if llm.abort or self.rt.stopped(mission_id):
            return PhaseResult(None, stopped=True, reason="budget", outcomes=sink)
        raw = getattr(result, "raw", str(result))
        m = _JSON_RE.search(raw or "")
        try:
            out = json.loads(m.group(0) if m else raw)
        except (TypeError, ValueError) as e:
            raise PhaseError(role, f"sortie non JSON : {str(raw)[:120]}") from e
        return PhaseResult(out, outcomes=sink)
