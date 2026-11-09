"""Service de missions : ouvre une mission ACSI, pilote le graphe, valide les approbations, reprend après crash.

Checkpointer :
- tests et défaut : `InMemorySaver` (mémoire du processus) ;
- déploiement : `SqliteSaver` (`sqlite_checkpointer(path)`), invoqué avec `durability="sync"` pour qu'aucun
  checkpoint ne manque après un crash. `thread_id` = `mission_id`.

Reprise (`resume`) : relance le graphe depuis son dernier checkpoint (`invoke(None, config)`). Le nœud
interrompu repart de son début ; les tools d'écriture sont idempotents (brouillon par mission et NC,
envoi par registre) et chaque nœud recontrôle le journal (ENF-04).
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from acsi_agent.team import MissionInput, TeamRuntime

from .doubles import BrainFactory
from .graph import build_graph
from .phases import DirectPhaseRunner, PhaseRunner


def sqlite_checkpointer(path: Path):
    from langgraph.checkpoint.sqlite import SqliteSaver
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(str(path), check_same_thread=False))


def make_runner(kind: str, rt: TeamRuntime) -> PhaseRunner:
    if kind == "direct":
        return DirectPhaseRunner(rt)
    if kind == "crewai":
        from .crew import CrewAIPhaseRunner
        return CrewAIPhaseRunner(rt)
    raise ValueError(f"runner inconnu : {kind}")


class MissionService:
    def __init__(self, rt: TeamRuntime, runner: PhaseRunner | str = "crewai", checkpointer: Any = None,
                 brains: BrainFactory | None = None):
        self.rt = rt
        self.runner = make_runner(runner, rt) if isinstance(runner, str) else runner
        self.brains = brains or BrainFactory()
        self.checkpointer = checkpointer if checkpointer is not None else InMemorySaver()
        self.graph = build_graph(rt, self.runner, self.brains).compile(checkpointer=self.checkpointer)
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    @staticmethod
    def config(mid: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": mid}}

    def _lock(self, mid: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(mid, threading.Lock())

    def _invoke(self, payload: Any, mid: str) -> None:
        # Un seul pilotage du graphe à la fois par mission (ENF-08). Verrou distinct de celui du runtime :
        # CrewAI exécute les tools dans ses propres threads, qui prennent le verrou ACSI à chaque appel.
        with self._lock(mid):
            self.graph.invoke(payload, self.config(mid), durability="sync")

    def start(self, mi: MissionInput, mission_id: str | None = None) -> dict[str, Any]:
        mid = self.rt.open_mission(mi, mission_id)
        self._invoke({"mission_id": mid, "journal_seq": 0}, mid)
        return self.rt.public(mid)

    def resolve(self, mid: str, approval_id: str, decision: str, approver: str, scopes: tuple[str, ...] | list[str],
                args_hash: str | None) -> dict[str, Any]:
        """ACSI valide d'abord (portée, approbateur distinct, TTL, usage unique, args_hash) ; puis seulement le
        graphe reprend avec `Command(resume=…)`. Une ApprovalError laisse le graphe en pause."""
        d = self.rt.resolve_approval(mid, approval_id, decision, approver, scopes, args_hash)
        if d in {"approve", "reject"}:
            self._invoke(Command(resume={"approval_id": approval_id, "decision": d}), mid)
        return self.rt.public(mid)

    def resume(self, mid: str) -> dict[str, Any]:
        """Reprise après un crash pendant un nœud : relance depuis le dernier checkpoint."""
        snap = self.graph.get_state(self.config(mid))
        if snap.next and not any(t.interrupts for t in snap.tasks):
            self._invoke(None, mid)
        return self.rt.public(mid)

    def snapshot(self, mid: str) -> dict[str, Any]:
        snap = self.graph.get_state(self.config(mid))
        return {"next": list(snap.next), "values": dict(snap.values),
                "checkpoint_id": (snap.config or {}).get("configurable", {}).get("checkpoint_id")}
